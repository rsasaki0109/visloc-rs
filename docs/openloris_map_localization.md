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
  office rearranged, the VIO corrected into the map frame gives a map pose
  for 94–99.8% of the frames at 2.8–7.6 cm RMSE. This holds although only
  0–37% of the images localize on their own. office1-7, none of whose
  images reaches the 30 inliers a fix needs, gets its map pose from
  [weak fixes](basalt_map_localization.md#weak-fixes-too-few-inliers-but-three-agree).
- **Wheel odometry:** correcting the robot's wheel odometry instead gives
  3.5–5.1 cm, once the point the wheel odometry reports and the level of
  its frame are calibrated from the VIO
  ([below](#where-the-wheel-odometry-is-measured)). The dataset's
  `base_link` sits 5–8 cm from that point and leans 1.9°, which made
  in-place turns look like up to 17 cm of drift (7–12 cm RMSE).
- **Scale:** the T265 calibration shipped with the dataset makes the maps
  1–3.5% long. With its stereo baseline corrected by a factor the wheel
  odometry gives (0.986), and close weak fixes used too
  (`--weak-gate-cm 5`), the VIO + map error falls to 1.1–2.4 cm
  ([below](#correcting-the-scale)).
- **Daylight:** the two sessions recorded in daylight with the ceiling
  lights off never localize in maps built at night from the raw images:
  ORB features do not survive that change. With the map's and the
  queries' local contrast equalized
  ([below](#across-lighting-local-contrast-equalization)), the daylight
  session office1-4 gets a map pose for 33% of its frames at 2.6 cm. The
  dawn session office1-5 still gets none.
- **Grainy images:** blurring before ORB (`--smooth-features 1.5`) and a
  lower detector threshold (`--detector-quality 0.003`) localize many more
  images across sessions: office1-7 186 times instead of 5, at 2.66 cm
  instead of 6.86 cm, and corridor1-1's VIO + map is 13 cm off at the
  median instead of 71 cm ([below](#grainy-images-blur-before-orb)).
- **A long run:** over the corridor's 284 s and 232 m, the wheels'
  heading drifts 38° in two minutes. Once each few seconds of the wheel
  calibration's fit get their own heading, its offset stays at
  (−4…3, 7…12) cm, near the office's (1–3, 6–8) cm; before, it wandered
  by a metre. Across those sessions only 2% of the images localize, so
  the run cannot score the map poses ([below](#over-a-long-run-corridor)).

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

| Map | From | SLAM ATE | Scale, SLAM / ground truth | Keyframe images / landmarks |
|---|---|---|---|---|
| A | office1-1 | 2.75 cm | 1.017 | 332 / 4,027 |
| B | office1-2 | 1.58 cm | 1.009 | 356 / 3,969 |
| — | office1-3 | 0.94 cm | 1.036 | 170 / 1,298 |

The SLAM comes out 1–3.5% long in scale on every sequence. This points to
the T265 calibration shipped with the dataset (`trans_matrix.yaml`).
Because of it, a fix is 3–4 cm off on average however well it matches.
Correcting the calibration's stereo baseline from the wheel odometry
removes most of this ([below](#correcting-the-scale)).

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

- Localization of every image until the first map pose, then every 5th
  (the defaults with `--localize-every 5`). The VIO runs on every frame;
  the wheel odometry is interpolated at every image.
- Projection matching (`--projection-radius-px 20`, the default), with the
  prediction used only within 5 s and 2 m of the newest fix.
- Scoring with the map's fixed alignment to the mocap frame: the
  "fixed" mode of `scripts/evaluate_map_localization.py`.

"Alone" is each odometry's own error after a rigid fit to the ground truth.
It is in the odometry's own frame, not the map's. The wheel odometry's
reference point and level are calibrated from office1-3's VIO
([below](#where-the-wheel-odometry-is-measured)).

| Query → map | Images that localize alone | VIO + map: frames, RMSE / max | Wheel odometry + map: frames, RMSE / max | VIO alone | Wheel alone |
|---|---|---|---|---|---|
| office1-3 → A (minutes later) | 36.7% | 99.7%, 2.83 / 3.7 cm | 99.2%, 3.45 / 4.0 cm | 0.83 cm | 0.50 cm |
| office1-6 → A (2.8 days, moved furniture) | 6.7% | 99.8%, 3.08 / 4.3 cm | 99.7%, 4.87 / 8.3 cm | 3.24 cm | 1.31 cm |
| office1-6 → merged | 2.9% | 99.8%, 3.30 / 4.7 cm | 99.7%, 5.08 / 8.6 cm | | |
| office1-7 → B | 0% | 99.4%, 6.86 / 13.7 cm | 99.3%, 5.08 / 10.8 cm | 5.29 cm | 1.45 cm |
| office1-7 → merged | 0% | 93.9%, 7.62 / 15.0 cm | 93.8%, 4.13 / 9.2 cm | | |
| office1-7 → merged, `--min-inliers 15` | | 94.9%, 4.06 / 8.2 cm | 94.8%, 2.05 / 3.6 cm | | |
| office1-4 → A, B or merged (daylight) | 0% | no map pose | no map pose | | 4.26 cm |
| office1-5 → merged (dawn) | 0% | no map pose, also at 15 inliers | no map pose | 5.20 cm | 2.59 cm |

What this shows:

- **The correction keeps the robot in the map frame although few images
  localize.** The robot needs map-frame poses across sessions, which VIO
  alone does not give: it has its own frame each time. On office1-6, 2.8
  days later with the furniture moved, fewer than 7% of the images
  localize alone. The corrected VIO still gets a map pose for 99.8% of the
  frames, about as accurate as the VIO's own rigid-fit error. On these
  6 m paths the VIO barely drifts, so the map's own few-cm error (its
  scale above) is what bounds the result.
- **Projection matching.** Matching around the corrected VIO's prediction
  raises office1-6's fixes from 16 to 128 of its 216 attempts (every 5th
  image from the start, before weak fixes). The VIO + map RMSE falls from
  3.93 to 3.15 cm.
- **Trying every image until the first map pose matters on the merged
  map.** Before weak fixes, with every 5th image from the start
  (`--initial-localize-every 5`), office1-6's first map pose came 2.0 s in
  and office1-7's (at 15 inliers) 7.8 s in; with every image, 0.5 s and
  1.9 s. Frames with a map pose rose from 94.4% to 98.5% and from 79.4% to
  94.9%. Where the map matches none of the images, it only adds attempts:
  every image for 30 s, then every 5th (`--initial-window-s`).
- **The wheel odometry follows the VIO once its frame is calibrated.**
  Where fixes keep coming (office1-3, office1-6), the corrected VIO stays
  ahead: 2.8–3.3 cm against 3.5–5.1 cm. Where they are few (office1-7), the
  calibrated wheel odometry is the better partner: 4.1–5.1 cm against
  6.9–7.6 cm by default, and 2.05 cm against 4.06 cm at 15 inliers. Before
  calibration it was 7–12 cm everywhere
  ([below](#where-the-wheel-odometry-is-measured)).
- **Merging both directions costs some matches.** office1-6 localizes
  alone in 2.9% of its images against the merged map, against 6.7% in A.
  The many near-identical landmarks make the ratio test reject more
  matches (`--duplicate-radius-m 0.1` did not change this run). Once the
  VIO has a map pose, projection matching makes up for it.
- **office1-7 gets its map pose from weak fixes.** Its best map-wide match
  has 21 inliers against B and 17 against the merged map, below the 30 a
  fix needs. On EuRoC 20 inliers already admits poses metres off, so 15 is
  not a safe default. Instead, localizations with 15–29 inliers are
  [weak fixes](basalt_map_localization.md#weak-fixes-too-few-inliers-but-three-agree):
  three that agree set the correction, and after that they are not used.
  Against B the correction is set 0.2 s in, and against the merged map 2.3
  s in. Few later images reach 30 inliers, even around the predicted
  projections: 5 and 13 fixes over the 38 s. So the VIO carries the pose
  for most of the run, at 6.9–7.6 cm. At `--min-inliers 15` every
  15-inlier localization is a fix, and none is more than 10.3 cm off,
  which gives 4.06 cm. `--weak-gate-cm 5` keeps the default threshold and
  uses only the weak fixes within 5 cm of the corrected VIO: 3.35 cm
  against B and 3.80 cm against the merged map
  ([more](#correcting-the-scale)).
- **Lighting is the hard limit.** office1-4 and office1-5 are recorded
  in daylight with the ceiling lights off. Their best images reach at
  most 12 inliers against maps recorded at night. No map, threshold or
  odometry gives them map poses: ORB features on these raw images do not
  match across that change. Equalizing the local contrast changes this
  for office1-4 ([below](#across-lighting-local-contrast-equalization)).

## Where the wheel odometry is measured

`--odometry` turns each odometry pose into the IMU's with `--imu-from-body`,
here the dataset's `base_link` in the T265 IMU frame. The wheel odometry,
though, reports another point, near the axle's middle, about which the robot
turns in place. If the extrinsic puts the body a few centimetres away from
that point, every turn swings the computed IMU pose by up to twice that
distance. In office1-3's final 150° in-place turn, the wheel odometry and
the mocap put `base_link` 16.7 cm apart. That looked like drift, and no fix
corrected it, since the camera turns away from the mapped view in that turn.
Fitted against the ground truth, `base_link` sits (1.7–4.7, 4.6–6.7) cm
(forward, left) from the wheel odometry's point in office1-1, -2, -3, -6
and -7, 5–8 cm away.

`scripts/calibrate_wheel_offset.py` fits that offset without ground truth,
from a run with both the VIO and the wheel odometry. It fits the VIO's body
path to the wheel odometry's, with the offset in the body frame, a
rotation, a translation and a scale (the VIO's scale differs by a few
percent). Turns in place are what reveal the offset. From office1-3's VIO
run it finds (1.3, 6.2) cm, within that spread, and leaves 0.40 cm RMS
(1.88 cm without the offset).

The extrinsic's tilt matters as much. The T265 sits 0.89 m above
`base_link`, and the wheel odometry keeps its body level, so a tilt of the
body frame in the extrinsic moves the computed IMU sideways: 1.6 cm per
degree. Against the VIO's gravity, the dataset's `base_link` leans by
1.9° (1.94, 1.86 and 1.97° on office1-3, -6 and -7), about 3 cm at the
IMU. Assuming a level floor, the script also levels the body frame: it
turns it by the smallest rotation that makes the VIO's mean up its z axis
(`--keep-tilt` skips this). It prints the corrected `--imu-from-body`, with
the body moved to the wheel odometry's point and levelled. With that one
calibration from office1-3 in every run:

| Wheel odometry + map | `base_link` as given | Point calibrated | ... and levelled |
|---|---|---|---|
| office1-3 → A | 11.6 / 19.6 cm | 6.23 / 9.7 cm | **3.45 / 4.0 cm** |
| office1-6 → A | 7.17 / 15.7 cm | 5.88 / 11.6 cm | **4.87 / 8.3 cm** |
| office1-6 → merged | 7.17 / 15.8 cm | 5.84 / 11.6 cm | **5.08 / 8.6 cm** |
| office1-7 → B | 11.9 / 23.3 cm | **5.00 / 11.4 cm** | 5.08 / 10.8 cm |
| office1-7 → merged | 11.6 / 21.4 cm | 4.42 / 9.2 cm | **4.13 / 9.2 cm** |
| office1-7 → merged, `--min-inliers 15` | 6.89 / 16.0 cm | 2.27 / 4.6 cm | **2.05 / 3.6 cm** |
| Wheel alone, rigid fit: office1-3, -6, -7, -4, -5 | 0.97, 4.80, 7.89, 8.34, 9.39 cm | 1.52, 1.08, 2.67, 4.63, 3.98 cm | 0.50, 1.31, 1.45, 4.26, 2.59 cm |

(RMSE / max. The frames with a map pose do not change.)

- **The calibration carries over between sessions.** It is fitted on
  office1-3 and holds on office1-6 and office1-7, recorded 2.8 days later.
  Calibrated and levelled, the rigid-fit error of the wheel odometry alone
  falls by a factor of 1.9–5.4 from the dataset's extrinsic on all five
  sessions, and wheel odometry + map by 1.4–3.4 times.
- **Levelling helps on most runs, not all.** Alone it helps on four of the
  five sessions (office1-6: 1.08 to 1.31 cm), and with the map on five of
  the six runs (office1-7 → B: 5.00 to 5.08 cm).
- **The fit uses the VIO only as a reference path.** The VIO's error over
  the run (0.83 cm on office1-3) sets how well the offset is found.
  Ground-truth fits on the five sessions spread by about 3 cm, so some of
  the error stays whichever value is used.
- **On a robot**, put the body frame (the odometry's `child_frame_id`) at
  the wheel odometry's point in the URDF. `visloc_map_localize_node`'s
  `extrinsics tf` then reads the corrected camera position from
  `/tf_static`.
- **Combining the two odometries did not help once corrected into the
  map.** Taking the VIO's attitude and the wheels' travel (each step's
  displacement in the wheel odometry's frame, turned by the VIO's heading)
  gives the best odometry alone on four of the five sessions: 0.39, 1.63,
  1.39 and 1.98 cm (rigid fit) on office1-3, -6, -7 and -4 (with the point
  calibrated, before levelling). Corrected
  into the map, though, it does no better than the better of the two: 3.37,
  5.03 and 5.42 cm on office1-3 → A, office1-6 → A and office1-7 → B. The
  point the robot turns about varies by about 3 cm between sessions, so
  the wheels' travel in a turn is off by that much, which a rigid fit over
  the whole run hides.

## Correcting the scale

A stereo VIO takes its metric scale mostly from the distance between its
two cameras. The calibration's baseline is 63.98 mm. Shrinking it changes
the SLAM's scale almost one to one:

| Baseline factor | office1-1: scale, ATE | office1-2: scale, ATE |
|---|---|---|
| 1 (as shipped) | 1.017, 2.75 cm | 1.010, 1.58 cm |
| 0.99 | 1.007, 1.22 cm | **0.999, 0.55 cm** |
| 0.98 | **0.997, 0.78 cm** | 0.989, 1.75 cm |
| 0.97 | 0.987, 2.14 cm | 0.979, 3.30 cm |

(Scale of the SLAM trajectory against the ground truth; ATE after a rigid
fit. With a scale fit too, the ATE is 0.5–0.6 cm at every factor.)

The ground truth is not needed to pick the factor. The wheel odometry's
scale is set by its wheels, and against the ground truth it is within 2%
on every session but office1-3, which mostly turns in place.
`scripts/calibrate_wheel_offset.py` fits the SLAM trajectory to it and
prints the scale: 1.013 on office1-1 and 1.015 on office1-2 (1.017 and
1.010 against the ground truth). Their mean gives a factor of 0.986, which
`scripts/scale_stereo_baseline.py` applies to the calibration, for the
maps and the queries alike. Maps A and B rebuilt with it are 1.003 and
0.995 of the ground truth's scale, at 0.76 and 0.91 cm ATE (2.75 and
1.58 cm before).

VIO + map, the same queries (frames with a map pose, RMSE / max):

| Query → map | As shipped | Baseline × 0.986 | ... and `--weak-gate-cm 5` |
|---|---|---|---|
| office1-3 → A | 99.7%, 2.83 / 3.7 cm | 99.7%, 1.19 / 1.5 cm | 99.7%, **1.08 / 1.5 cm** |
| office1-6 → A | 99.8%, 3.08 / 4.3 cm | 99.8%, 1.87 / 2.8 cm | 99.8%, **1.27 / 2.0 cm** |
| office1-7 → B | 99.4%, 6.86 / 13.7 cm | 98.6%, 5.36 / 10.6 cm | 98.6%, **2.36 / 4.7 cm** |

- **The fixes get more accurate.** With the scale right, the fixes are 1.1
  to 1.7 cm off (RMSE) instead of 3.5–3.9 cm, and the VIO alone drifts
  less: 1.15 instead of 3.24 cm on office1-6, 3.00 instead of 5.29 cm on
  office1-7 (rigid fit).
- **Weak fixes close to the corrected VIO help most where full fixes are
  rare.** With `--weak-gate-cm 5`, once the VIO has a map pose, a
  localization with 15–29 inliers (map-wide or by projection) within 5 cm
  and 2° of where the correction puts the VIO is used as well. office1-7
  gets 116 of them next to 22 full fixes. Keeping the correction close
  keeps the predicted projections close, so more images reach 30 inliers
  too (office1-6: 120 full fixes instead of 84). On EuRoC it does not
  help (changes of -0.6 to +0.5 cm), so it stays off by default
  ([numbers](basalt_map_localization.md#weak-fixes-too-few-inliers-but-three-agree)).
- **The wheel odometry no longer gains on the VIO.** On the corrected maps
  it gives 1.96, 5.30 and 4.89 cm (1.70, 2.38 and 3.66 cm with the weak
  gate) on the same three queries: its turns are off by the variation in
  the turning point ([above](#where-the-wheel-odometry-is-measured)).
- **One factor does not fit every session exactly.** With it, the VIO of
  the three queries is 1.006–1.018 of the ground truth's scale. The best
  factor per session ranges from 0.97 to 0.99.

## Across lighting: local contrast equalization

ORB compares the brightness of nearby pixels. A whole image getting
brighter or darker does not affect it, but a change of lighting across the
scene does: by day the ceiling lights are off and the windows light the
room, so the shading moves and the darker parts lose contrast.
`basalt_euroc_online_slam_demo --equalize-local-contrast` evens out the
local contrast of each keyframe image before ORB extraction
(`visloc_basalt::contrast::LocalContrast`: contrast-limited adaptive
histogram equalization over 8 × 8 tiles, clip limit 2). `map.json` records
it, and `basalt_euroc_localize_demo` and `visloc_map_localize_node` then
equalize every query image the same way. The VIO tracks the raw images.

Maps A (office1-1) and B (office1-2) rebuilt this way, against the maps
from the raw images. Same settings as above. "15 inliers" is
`--min-inliers 15`, which leaves no room for weak fixes. The last two rows
were measured before weak fixes:

| Query → map | Raw map | Equalized map |
|---|---|---|
| office1-4 → B (daylight) | no map pose (no image above 12 inliers) | **33.4%, 2.62 / 4.2 cm** |
| office1-4 → B (daylight), 15 inliers | no map pose | **33.6%, 2.46 / 4.3 cm** |
| office1-5 → A (dawn), 15 inliers | no map pose | no map pose (no image above 11 inliers) |
| office1-7 → B | 99.4%, 6.86 / 13.7 cm | 99.6%, 6.72 / 13.5 cm |
| office1-7 → B, 15 inliers | 99.6%, 3.36 / 6.9 cm | 99.7%, 2.30 / 4.4 cm |
| office1-6 → A | 99.7%, 3.15 / 4.3 cm | 99.9%, 3.24 / 4.7 cm |
| office1-3 → A | 99.7%, 2.83 / 3.7 cm | 99.7%, 2.92 / 3.7 cm |

(VIO + map: frames with a map pose, RMSE / max.)

- **It lets a daylight session localize in a night map.** Against the
  equalized B, 24 of office1-4's 870 images localize on their own at 15
  inliers, all within 3.8 cm of the ground truth (none against the raw
  B). The VIO then has a map pose from the first two agreeing fixes, 17 s
  in, to the end. The other direction of its drive, which only A covers,
  stays without fixes.
- **Weak fixes are enough to start.** Its best daylight image has 23
  inliers against the map, so the default 30 admits none.
  Three weak fixes with 15–17 inliers set the correction 19.3 s in. After
  that, matching around the predicted projections finds 30–48 inliers, so
  the 19 fixes that follow are full ones: 33.4%, 2.62 / 4.2 cm, about as
  at `--min-inliers 15`. The calibrated wheel odometry in place of the VIO
  gives 33.4%, 4.52 / 10.0 cm.
- **The dawn session stays out of reach.** office1-5 is darker still; its
  best image reaches 11 inliers against the equalized A as against the
  raw one. Equalizing the images the VIO tracks too, not just the ORB
  frontend's, gave it 7 fixes and a map pose for 18.9% of its frames in
  one trial, but that was the only run that did.
- **A daylight session in the map does not bridge the change either.**
  office1-4's own equalized map (242 keyframe images, 3.94 cm SLAM ATE)
  registers in neither equalized A nor B: none of its keyframes localizes
  in them. Against that daylight map, neither the dawn office1-5 nor the
  night office1-6 localizes a single image (at most 10 and 12 inliers).
- **Within the same lighting it does little.** More images localize on
  their own (office1-6: 174 fixes over the flight instead of 131), and the
  maps have more landmarks (A 5,358 instead of 4,027), but the VIO + map
  error moves by at most 0.1 cm, except office1-7 at 15 inliers (3.36 to
  2.30 cm).
- **It stays off by default.** On EuRoC it changes little, and V2_03 gets
  worse ([numbers](basalt_map_localization.md#local-contrast-equalization)).
  `basalt_map_merge` refuses to merge an equalized map with a raw one,
  whose descriptors would not match.

## Grainy images: blur before ORB

On the corridor ([below](#over-a-long-run-corridor)) few images localize
across sessions, and not because of the lighting. Within 1 m of
corridor1-5's path and facing within 30° of it, 6.3% of corridor1-1's 898
tries localize; corridor1-5's own images, against its map, 33.5%. Two
near-identical views from the two sessions (6–9 cm and under 5° apart,
equally bright) share 22 geometrically consistent ORB matches per pair.
The T265's images are grainy, and ORB's descriptor compares single
pixels: noise flips enough of those comparisons that the same corner's
descriptors fall too far apart. OpenCV's ORB blurs before its descriptor
for this reason; the Basalt frontend this repository ports does not.

`basalt_euroc_online_slam_demo --smooth-features SIGMA` blurs each keyframe
image (after any equalization) before ORB detection and description.
`map.json` records it, and `basalt_euroc_localize_demo` and
`visloc_map_localize_node` blur their images the same way. At sigma 1.5
the pairs share 38 consistent matches:

| corridor1-1 → corridor1-5 | Raw map | Blurred map (1.5) |
|---|---|---|
| Map landmarks | 7,036 | 9,620 |
| corridor1-5's own images that localize (every 5th) | 33.5% | 52.1% |
| corridor1-1, VIO + map: localizations; RMSE against the map around it | 57; 1.55 m | 75; 1.35 m |
| ... wheel odometry with the gyro's heading + map | 58; 0.95 m | 76; 0.93 m |
| Time per image (median, ORB included) | 43.5 ms | 54.1 ms |

- **The localizations still come at the same two places**, more of them.
  Elsewhere the corridor yields too few matches either way (a median of
  14 with the blur).
- **No other session fills the quarter of the path the map misses.**
  corridor1-3 covers most of it, but none of its images (every 5th, 421)
  localizes in corridor1-5's map, so `basalt_map_merge` cannot register it.
- **Local contrast equalization does not help here:** 5,138 landmarks,
  32% of corridor1-5's own images, 57 localizations of corridor1-1.

In the office it localizes more images at about the same error (VIO +
map, `--localize-every 5`, the maps as shipped and rebuilt with the blur):

| Query → map | Raw map: RMSE / max, localizations | Blurred map |
|---|---|---|
| office1-3 → A | 2.83 / 3.7 cm, 34 | 2.97 / 3.8 cm, 43 |
| office1-6 → A | 3.08 / 4.3 cm, 130 | 3.26 / 4.5 cm, 180 |
| office1-7 → B | 6.86 / 13.7 cm, 5 | 6.61 / 13.2 cm, 52 |

On EuRoC: [basalt_map_localization.md](basalt_map_localization.md#blurring-before-orb).
It stays off by default.

### And a lower detector threshold

The detector keeps corners stronger than 1% of the image's strongest
(OpenCV's quality level). In the corridor a few bright corners (lamps,
door frames against the windows) set that bar, and the walls' weaker
texture falls below it: the blurred images give 201 corners each.
`basalt_euroc_online_slam_demo --detector-quality Q` sets the level for a
map, `map.json` records it, and localization detects the same way. On the
near-identical pairs, blurred, 0.3% gives 421 corners and 60 consistent
matches per pair, 0.1% 756 and 82 (1%: 201 and 38). Splitting the image
into cells, each with its share of the corner budget, gave fewer corners
for the same matches and was dropped.

| corridor1-1 → corridor1-5 | Blurred | Blurred, quality 0.003 | Blurred, quality 0.001 |
|---|---|---|---|
| Map landmarks | 9,620 | 15,815 | 19,042 |
| corridor1-5's own images that localize (every 5th) | 52.1% | 73.4% | 76.4% |
| corridor1-1, VIO + map: localizations (their RMSE); RMSE / median | 75 (2.9 cm); 1.35 m / 62 cm | 360 (4.9 cm); **0.94 m / 13 cm** | 130 (3.5 cm); 1.43 m / 54 cm |
| ... wheel odometry with the gyro's heading + map | 76 (2.9 cm); 0.93 m / 64 cm | 164 (3.9 cm); 0.66 m / 48 cm | 254 (5.3 cm); 0.67 m / 28 cm |
| Time per image (median) | 54 ms | 74 ms | 103 ms |

- **At 0.003 corridor1-1 localizes along much more of the path.** Its VIO +
  map is 13 cm off at the median, from 71 cm with the raw map; what is
  left comes from the stretches the map does not cover.
- **0.001 takes in noise too.** More corners, but the VIO + map gets fewer
  localizations (130) and the wheel odometry's are less accurate.

In the office, with both (VIO + map, `--localize-every 5`, the maps as
shipped and rebuilt):

| Query → map | Raw map: RMSE / max, localizations | Blurred | Blurred, quality 0.003 |
|---|---|---|---|
| office1-3 → A | 2.83 / 3.7 cm, 34 | 2.97 / 3.8 cm, 43 | 2.76 / 3.7 cm, 56 |
| office1-6 → A | 3.08 / 4.3 cm, 130 | 3.26 / 4.5 cm, 180 | 3.15 / 4.5 cm, 213 |
| office1-7 → B | 6.86 / 13.7 cm, 5 | 6.61 / 13.2 cm, 52 | **2.66** / 4.7 cm, 186 |

office1-7, whose images never reached the 30 inliers a localization needs,
now localizes 186 times and is 2.66 cm off. For the T265, build maps with
`--smooth-features 1.5 --detector-quality 0.003`. Both stay off by default.

### Two more scenes: cafe and home

The same comparison in two other scenes, each scene's map built from its
first session and the second session localized in it (VIO + map,
`--localize-every 5`). Their ground truth comes from a 2D laser (z = 0,
level). The maps themselves are 17–19 cm from it after a rigid fit, almost
all of it horizontal: the VIO drifts that much over 30–40 m with no loop
closed. So each localization is scored with the local fit used in the
corridor, on the map poses within 3 m.

| | Raw map | Blurred, quality 0.003 |
|---|---|---|
| cafe1-1's map: landmarks | 12,422 | 14,072 |
| cafe1-2's localizations | 180 | 207 |
| ... their error, local fit: RMSE / median (scored) | 5.8 / 4.3 cm (146) | 5.7 / 3.8 cm (167) |
| home1-1's map: landmarks | 13,856 | 24,250 |
| home1-2's localizations | 29 | **177** |
| ... their error, local fit: RMSE / median (scored) | 9.5 cm (3) | 15.1 / 12.9 cm (26) |
| Time per image (median), cafe / home | 40 / 22 ms | 56 / 58 ms |

In the café the settings add 15% more localizations at the same error; in
the home, six times as many. Home's ground truth has a pose every 0.1 s at
best, so few of them get scored, and those few are less accurate than in
the café. The market (41.5 GB to download) was not tried.

## On ROS 2

The same runs, through the nodes instead of the demo: each session as a
rosbag2 (`tools/euroc_to_rosbag2.py --odometry`, with the stereo images, the
IMU and the wheel odometry on `/wheel/odom`), played at real time with `ros2
bag play`, on 4 cores. Two `static_transform_publisher`s give `base_link ->
imu0` (the calibrated and levelled extrinsic above) and `imu0 -> cam0`.
`visloc_map_localize_node` runs with `extrinsics:=tf` and `weak_gate_m:=0.05`
against the scale-corrected maps ([above](#correcting-the-scale)), and its
odometry is either `visloc_vio_node` (`base_frame:=base_link`) or the
bag's wheel odometry. `map_pose` (of `base_link`, moved to the IMU) is
scored like the demo's:

| Query → map | VIO node: first map pose, RMSE / max | Demo | Wheel odometry: first map pose, RMSE / max | Demo |
|---|---|---|---|---|
| office1-6 → A | 0.2 s, 1.51 / 2.7 cm | 1.27 / 2.0 cm | 0.2 s, 2.47 / 4.4 cm | 2.38 / 4.3 cm |
| office1-7 → B | 0.9 s, 2.22 / 5.0 cm | 2.36 / 4.7 cm | 0.7 s, 3.25 / 8.8 cm | 3.66 / 9.5 cm |

- **The nodes reproduce the demo** within 0.4 cm, at real time, with the
  wheel odometry at its own 20 Hz and the camera's position from
  `/tf_static`. `map_pose` comes for every odometry message from the first
  map pose on.
- **The extrinsic's level shows most here.** The node corrects
  `base_link`'s pose, 0.9 m below the camera, so a lean in `base_link ->
  imu0` moves every fix against the level wheel odometry. With it only
  moved to the wheel odometry's point, not levelled, office1-7's wheel
  odometry was 4.67 cm instead of 3.25 cm.
- **One run of six lost `/tf_static`.** In one office1-7 run,
  `visloc_vio_node` never received the transient-local `base_link -> imu0`
  and published no odometry. A probe with the nodes' reader
  (`ros2/visloc-ros2/examples/tf_static_probe.rs`), started 200 times next
  to freshly started `static_transform_publisher`s, had nothing after 30 s
  in 4 cases. A second reader in the same node gets them, so both nodes
  now subscribe again while a transform they need is missing; with that,
  300 of 300 probes had both transforms.

### Calibrating the wheel odometry online

`visloc_wheel_calibration_node` fits the wheel odometry's point, the lean of
`base_link` and the map's scale while the robot drives, and publishes the
wheel odometry as `base_link`'s pose
([node README](../ros2/visloc-ros2/README.md#visloc_wheel_calibration_node)).
It takes the offset from the VIO, which follows every turn, and the lean and
the scale from the map node's localizations (`map_fix`), which carry the
map's own gravity and scale; without the VIO, the localizations give the
offset too, held near the saved one (or zero) where their turns do not show
it. It refits every 5 s.

Which reference should give what was first measured offline, in the demo:
each calibration applied to the whole wheel odometry from its first pose
(RMSE of `basalt_euroc_localize_demo`'s map poses):

| Calibration of the wheel odometry | office1-6 → A | office1-7 → B |
|---|---|---|
| None | 7.17 cm | 11.96 cm |
| The session's own VIO fit: offset, lean and scale | 4.18 cm | 6.39 cm |
| ... with the wheels' scale | 3.97 cm | 5.04 cm |
| ... with the map's lean and scale (from its localizations) | 4.10 cm | **3.30 cm** |
| office1-3's offline calibration (wheels' scale) | 4.89 cm | 5.07 cm |
| ... with the map's lean and scale | 4.88 cm | 3.64 cm |
| The map's localizations alone: offset, lean and scale | 5.05 cm | — |
| ... lean and scale only | 6.09 cm | 8.23 cm |

- **A VIO session's scale is not the map's.** office1-7's VIO is 3.6% long
  against its wheels, map B 1.1%; applying the VIO's scale costs 1.3 cm
  there. The localizations give the map's scale directly (1.011 against
  the wheels, the same on both maps), so the node uses only theirs.
- **The map's gravity is what the map node sees.** The lean measured by
  the localizations (1.4° on map B) differs from the VIO's (1.9°), and on
  office1-7 it is worth another 0.8 cm.
- **The VIO is better at the offset.** The localizations come where the
  camera sees the map, rarely through a turn in place: office1-6's span
  118° of the wheels' turn, office1-7's none.

Through ROS 2 (both rosbag2s at real time, the dataset's own
`base_link -> imu0` on `/tf_static`, the maps as shipped, the map node's
defaults), RMSE / max of `map_pose`, moved to the IMU:

| Wheel odometry into the map node | office1-6 → A | office1-7 → B |
|---|---|---|
| As is | 8.01 / 14.0 cm | 12.10 / 22.8 cm |
| Calibrated in the run from the map's localizations alone | **5.49 / 9.5 cm** | 10.89 / 21.6 cm |
| ... from the VIO and the localizations | 5.46 / 9.4 cm | 10.88 / 21.5 cm |
| ... with the gyro's heading (`imu`), from the localizations alone | 5.34 / 9.4 cm | 10.00 / 17.2 cm |
| ... from the VIO and the localizations | 5.26 / 7.3 cm | 10.01 / 17.2 cm |
| ... without the scale (`apply_scale:=false`) | 5.40 / 9.0 cm | — |
| From office1-3's offline calibration, saved | 4.92 / 8.3 cm | 4.66 / 10.4 cm |
| ... refined in the run from the localizations | 5.01 / 9.3 cm | 4.66 / 10.5 cm |
| ... refined from the VIO and the localizations | 5.16 / 9.7 cm | 4.89 / 10.9 cm |
| From the other session's VIO calibration, refined | 5.07 / 9.2 cm | 9.79 / 19.5 cm |

- **The gyro's heading helps through ROS 2 too**
  ([below](#the-wheels-heading-from-a-gyro)). With `-r imu:=/imu0`, the
  node takes the bag's T265 IMU (bias 5.4 and −9.3°/min); run again without
  it, the rows above come out within 0.08 cm. On 4 cores the VIO node
  drops a different share of frames from run to run (123–410 of about
  1,100), which moves the VIO rows by a few millimetres either way.
- **Without a VIO or a saved calibration, the map calibrates the wheels.**
  office1-6's 131 localizations give (0.7, 4.8) cm, a 1.87° lean and a
  scale of 1.018 within the run, and the error falls from 8.01 to 5.49 cm.
  The scale changes it by 0.1 cm: with fixes this frequent, the correction
  absorbs it.
- **office1-7 is too short for it.** Its 24 localizations come in the first
  11 s and span 0.5 m of the wheels' path, enough for the lean only; the
  VIO turns 180° only at the end of the 38 s session.
- **A saved calibration stays the best start.** In sessions this short,
  refining office1-3's calibration does not improve on it, and the VIO's
  offset from office1-7's last seconds costs 0.2 cm. On a long run the
  refits keep following the robot (over the last 120 s of pairs;
  [below](#over-a-long-run-corridor)).
- **The output stays continuous.** Each refit continues the published pose
  with a yaw and a translation, as at the first estimate (a step of 1.03 cm
  where the wheels moved 1.05 cm).

### Over a long run: corridor

The office sessions last 12–53 s. OpenLORIS-Scene's corridor sessions are
longer: corridor1-1 drives 232 m in 284 s along a loop of corridors, turning
2,026° in all by its wheel odometry. Their ground truth comes from a 2D laser
SLAM, not motion capture. The map is corridor1-5's (146 s, 126 m; 1,836
keyframe images, 7,036 landmarks), built as above. Its SLAM closes no loop:
it is 1.33 m RMSE and up to 3.9 m off the ground truth, and 3% long. It
cannot close one. corridor1-5 drives each corridor out and back, and every
place it passes twice (921 pairs of its poses within 1 m and over 30 s
apart) it passes facing the other way, so its forward cameras never see
the same view twice. Fitting a scale too leaves 1.17 m: the error is
mostly the VIO's heading drift. So the localizations are scored against
the map around them: each is put in the ground truth's frame through the
stretch of the map's own trajectory within 5 m of it
(`evaluate_map_localization.py --mode local`). Scored that way,
the map's own SLAM poses are 3.2 cm off at the median; the laser ground
truth jumps by up to 0.7 m in places.

- **Few images localize across these sessions.** 57 of the 2,426 images
  corridor1-1 tries localize, at two places (66–68 s and 206–214 s). Each
  is 3.1 cm RMSE from the map around it. A quarter of corridor1-1's path
  lies more than 1 m from corridor1-5's, up to 22 m. Between those two
  places the VIO + map drifts up to 3.8 m (71 cm at the median). The wheel
  odometry gets 12 localizations and drifts by metres. This run therefore
  does not measure the wheel odometry + map. It measures whether the
  calibration keeps up.
- **The wheels slip in turns.** Their heading drifts 15° off the VIO's in
  one 350° turn in place, and 38° after 120 s. The VIO stays within 1.3–5°
  of the ground truth.

With one heading and position between the wheel odometry and the
reference over the whole 120 s window, the offset took up that drift and
wandered by a metre. The fit now gives each 5 s of VIO pairs (`segment_s`),
and each 30 s of map localizations (`fix_segment_s`), a heading and a
position of their own, and shares the offset and the scale. Replaying
corridor1-1's wheel odometry, VIO and localizations through it
(`wheel_calibration_replay`, the node's calibration without ROS), with an
estimate every 5 s:

| corridor1-1 | One heading per 120 s window | Each 5 s / 30 s its own |
|---|---|---|
| Offset from the VIO, forward / left, over the run | −117 … 29 / −64 … 71 cm | −4.0 … 2.8 / 6.7 … 12.1 cm |
| Fit residual at the end (RMS) | 62 cm | 2.3 cm |
| Map's scale from all 57 localizations (`window_s:=0`) | 0.485 | 1.0315 |

- **The offset is the office's.** Over the whole run it is (0.7, 8.2) cm.
  The office VIO runs give (1–3, 6–8) cm, and the office's motion capture
  (2.7–4.7, 6.8–7.4) cm. The corridor's laser ground truth puts it at
  (15, 0) cm, so its `base_link` is a different point and it cannot check
  the offset.
- **The scale is the map's.** 1.0315 is the 3% the map is long. With one
  heading for all localizations, the 8 at 66 s and the 49 at 206 s, 140 s
  and 40° of wheel drift apart, gave 0.485.
- **The short office sessions stay where they were.** Through the same
  replay, office1-6's and office1-7's wheel odometry + map in the demo
  stays at 5.11 cm and goes from 11.70 to 11.69 cm, with the VIO and the
  localizations.

### The wheels' heading from a gyro

The wheels' travel holds up over the corridor run: it is 1.1% short of the
ground truth. Their heading does not. The wheels slip in turns in place,
and their heading drifts 44.5° from the ground truth over corridor1-1's
272 s, 15° of it in its first 350° turn. Between two localizations a
robot's pose rests on that odometry, and on the corridor localizations are
minutes apart.

`visloc_localization::GyroWheelOdometry` keeps the wheels' travel and turns
it by the heading a gyro integrates about the vertical. The accelerometer
gives the vertical, so the IMU may sit anywhere on the robot in any
orientation; here it is the T265's. Where the wheels stand still, the
heading holds, and the gyro's rate about the vertical is taken as its bias
(corridor1-1: −2.7°/min, from its first second).
`visloc_wheel_calibration_node` uses it with an `imu` topic, and
`wheel_calibration_replay` with `--imu`.

Odometry alone on corridor1-1, from every second of the run: how far off
its end point is after 10, 30 and 60 s, with the heading aligned at the
start. The ground truth's own heading is 2.3° off its direction of travel,
and is corrected for that.

| After | Wheels: median / p90 / max | Wheels with the gyro's heading |
|---|---|---|
| 10 s | 0.22 / 0.41 / 0.71 m | 0.15 / 0.29 / 0.52 m |
| 30 s | 0.82 / 3.1 / 4.7 m | 0.37 / 0.53 / 0.76 m |
| 60 s | 2.7 / 6.8 / 15.9 m | 0.81 / 1.06 / 1.37 m |
| Heading over the whole run | −44.5° … 3.2° | −5.6° … 2.7° |

- **The map finds the robot again.** In the demo, wheel odometry + map on
  corridor1-1 (calibrated from the localizations alone, through the
  replay) gets 58 localizations instead of 12: with the heading held, the
  prediction still points the projection search at the right landmarks
  140 s after the last localization. Scored against the map around each
  pose, its RMSE falls to 0.95 m (median 0.68 m, max 2.9 m). Without the
  gyro it is 3.5 m (median 2.2 m, max 8.5 m) where it is scored at all:
  more than half its poses drift over 5 m from the map. The VIO + map gets
  1.55 m: with no localization over the last 70 s, its own drift reaches
  3.8 m.
- **What remains is mostly the wheels' scale.** 60 s of driving is about
  50 m, and 1.1% of it is 0.55 m.
- **The short office sessions gain too.** Through the replay (VIO and
  localizations into the calibration, then the demo), office1-6's wheel
  odometry + map goes from 5.11 to 4.79 cm and office1-7's from 11.69 to
  10.60 cm; with the localizations alone, from 5.11 to 4.79 cm and from
  11.96 to 10.78 cm.

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

# Map with local contrast equalization (for office1-4 against office1-2's map)
target/release/examples/basalt_euroc_online_slam_demo --euroc-dir data/office1-2 \
  --calibration data/office1-2/calibration.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir runs/o12_slam_eq --save-map runs/o12_map_eq --equalize-local-contrast
target/release/examples/basalt_euroc_localize_demo --map runs/o12_map_eq \
  --euroc-dir data/office1-4 --out-dir runs/o14_vio_eq --vio \
  --calibration data/office1-4/calibration.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --localize-every 5

# Query: VIO corrected into the map frame
target/release/examples/basalt_euroc_localize_demo --map runs/o11_map \
  --euroc-dir data/office1-3 --out-dir runs/o13_vio --vio \
  --calibration data/office1-3/calibration.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --localize-every 5

# Where the wheel odometry is measured, from that VIO run. --imu-from-body
# is base_link in the T265 IMU frame, the inverse of T_base_imu in
# data/office1-3/extrinsics.json. Prints the corrected --imu-from-body
# (here -0.0314,-0.8938,-0.2587,-0.5059,-0.5042,-0.4958,0.4940).
python3 scripts/calibrate_wheel_offset.py --vio runs/o13_vio/trajectory_vio.tum \
  --odometry data/office1-3/odom.tum \
  --imu-from-body=0.0306,-0.8937,-0.2452,-0.5030,-0.4927,-0.5074,0.4967

# Query: the wheel odometry corrected into the map frame
target/release/examples/basalt_euroc_localize_demo --map runs/o11_map \
  --euroc-dir data/office1-6 --out-dir runs/o16_wheel \
  --odometry data/office1-6/odom.tum \
  --imu-from-body -0.0314,-0.8938,-0.2587,-0.5059,-0.5042,-0.4958,0.4940 \
  --localize-every 5

# The scale: the SLAM trajectory of each mapping session against its wheel
# odometry ("VIO scale vs wheel odometry": 1.013 and 1.015), then the
# baseline divided by their mean, for every sequence's calibration
python3 scripts/calibrate_wheel_offset.py --vio runs/o11_slam/trajectory_online.tum \
  --odometry data/office1-1/odom.tum \
  --imu-from-body=-0.0314,-0.8938,-0.2587,-0.5059,-0.5042,-0.4958,0.4940
python3 scripts/scale_stereo_baseline.py --calibration data/office1-1/calibration.json \
  --scale 0.986 --out data/office1-1/calibration_scaled.json
# ... then the map and the queries as above, with calibration_scaled.json
# (and --weak-gate-cm 5 for the last column)

# On ROS 2: a bag with the wheel odometry, then the nodes (source ROS 2 Jazzy)
python3 ros2/visloc-ros2/tools/euroc_to_rosbag2.py data/office1-7 o17_bag \
  --odometry data/office1-7/odom.tum
ros2 run tf2_ros static_transform_publisher --x 0.276788 --y 0.028308 --z 0.888469 \
  --qx 0.505898 --qy 0.504198 --qz 0.495798 --qw 0.493998 \
  --frame-id base_link --child-frame-id imu0 &   # the inverse of the --imu-from-body above
ros2 run tf2_ros static_transform_publisher --x 0.010699 --y -0.000053 --z -0.000119 \
  --qx 0.005548 --qy 0.001361 --qz 0.999981 --qw 0.002460 \
  --frame-id imu0 --child-frame-id cam0 &        # T_imu_cam0 of calibration.json
visloc_map_localize_node --map runs/o12_map_scaled --ros-args -p extrinsics:=tf \
  -p weak_gate_m:=0.05 -r image:=/cam0/image_raw -r odom:=/wheel/odom &
ros2 bag play o17_bag --delay 3

# The corridor: corridor1-5's map, corridor1-1 the query. corridor1-1.7z
# (13.9 GB) is a file of its own; corridor1-5.7z (6.9 GB) is a member of
# corridor1-2_5-package.tar, bytes 12802381312 .. +6892693271
curl -L -o corridor1-1.7z $U/corridor1-1.7z
curl -L -r 12802381312-19695074582 -o corridor1-5.7z $U/corridor1-2_5-package.tar
python3 scripts/openloris_to_euroc.py --sequence corridor1-1.7z --out data/corridor1-1
python3 scripts/openloris_to_euroc.py --sequence corridor1-5.7z --out data/corridor1-5
# ... runs/c15_slam and runs/c15_map from corridor1-5, runs/c11_vio as the
# VIO query above. Then the calibration node's fit, replayed (--imu-from-body:
# the inverse of T_base_imu in data/corridor1-1/extrinsics.json):
cargo run --release --manifest-path ros2/visloc-ros2/Cargo.toml \
  --example wheel_calibration_replay -- --wheel data/corridor1-1/odom.tum \
  --imu-from-body 0.030639,-0.893715,-0.245211,-0.503044,-0.492681,-0.507444,0.496703 \
  --vio runs/c11_vio/trajectory_vio.tum --fixes runs/c11_vio/localization.tum \
  --out runs/c11_wheel.tum --estimate runs/c11_wheel.json
# ... and scored against the map around each pose
python3 scripts/evaluate_map_localization.py --mode local \
  --estimate runs/c11_vio/localization.tum \
  --reference data/corridor1-1/mav0/state_groundtruth_estimate0/data.csv \
  --align-estimate runs/c15_slam/trajectory_online.tum \
  --align-ground-truth data/corridor1-5/mav0/state_groundtruth_estimate0/data.csv

# Score against the ground truth, through the map's own alignment
python3 scripts/evaluate_map_localization.py --mode fixed \
  --estimate runs/o16_wheel/trajectory_map.tum \
  --reference data/office1-6/mav0/state_groundtruth_estimate0/data.csv \
  --align-estimate runs/o11_slam/trajectory_online.tum \
  --align-ground-truth data/office1-1/mav0/state_groundtruth_estimate0/data.csv \
  --queries 1080
```

The byte ranges of the other sequences are in the tar's headers. The
dataset is CC BY-ND 4.0: keep the converted images out of the repository.

## Limits

- **One scene, short paths.** Seven sequences, all in one office, each at
  most 11 m. They test change over days, not large areas. On the one long
  run, in the corridor, few images localize across sessions
  ([above](#over-a-long-run-corridor)).
- **ORB only.** Across the day/night change no raw image localizes.
  Equalizing the local contrast gets one of the two daylight sessions a
  map pose, through weak fixes; the dawn session still has none.
- **The T265's shipped calibration.** It leaves the maps 1–3.5% long in
  scale, so every map pose carries a few cm of error. The baseline
  correction above removes most of it; the scale still varies by up to
  2% between sessions, which only a proper calibration would remove.
- **The wheel odometry is planar and its frame is assumed level.** The
  office floor is level, which this needs.
