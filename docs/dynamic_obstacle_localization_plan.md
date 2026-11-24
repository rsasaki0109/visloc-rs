# Map-based localization among moving people and objects (plan)

Status: **plan only, nothing implemented yet** (2026-11-24).

Goal: keep localizing against a saved visual map when people and other
movable things are in view, **indoors first** (robots in offices, cafés,
warehouses, homes), on **embedded hardware such as a Jetson**, where every
extra millisecond and megabyte counts.

## Where things stand

- **Map localization** (`basalt_euroc_localize_demo`, `visloc_map_localize_node`)
  matches the image's ORB keypoints to the map's landmarks by Hamming
  distance, around the predicted projections when a prediction is fresh,
  then P3P RANSAC. A pose with at least `min_inliers` inliers is a fix.
  `OdomCorrection` gates the fixes (15 cm / 3°, widened with travel),
  averages 5 s of them into `map -> odom`, and replaces the transform when
  three rejected fixes in a row agree.
- **Nothing on the map-localization side knows about moving things.** The
  only handling is in the VIO: `visloc_basalt::dynamic_objects` drops the
  tracks inside detector boxes judged moving from their residuals
  ([object_metric_slam_plan.md](object_metric_slam_plan.md), Phase 2).
  It gave no gain on LaMAria R_08 (outdoor walk), where moving objects
  are a small share of the view. Indoors, a person 1 m from the camera can
  fill half of it.
- **What has been seen so far** (the 3DGS street sim, outdoor,
  [navigation.md](navigation.md)): a cyclist and a dog walker passing in
  front of the camera left 18 s without a single localization; eight
  people cost 4% of the localizations; the run-to-run spread (4–16 cm
  RMSE on the same street) is larger than most single-lever effects, so
  one run per setting decides nothing.

## Failure modes

Each lever below is aimed at one of these, and Phase 0 measures each one
separately.

| # | Mode | What happens |
| --- | --- | --- |
| F1 | Occlusion of the query | People hide the mapped structure: fewer matches, no fix, the odometry carries the pose for longer |
| F2 | Wrong matches on movers | Keypoints on a person or a chair match map landmarks by chance (repetitive clothing, similar texture) and pull the RANSAC pose |
| F3 | A dirty map | People, chairs, boxes or carts present while mapping are landmarks; at query time they are gone or moved |
| F4 | Dirty odometry | A large mover drags the VIO; handled in the VIO only when boxes are supplied |
| F5 | The correction fooled | Three fixes locked onto a mover agree with each other and replace `map -> odom` |

Indoors F1 and F3 are expected to dominate: people stand close, and
furniture moves between sessions (OpenLORIS office already showed moved
furniture).

## Resource budget (Jetson)

The reference target is a Jetson Orin Nano / Orin NX class device (6–8
ARM cores, shared CPU/GPU memory). Until it is on hand, the 4-core
development container is the stand-in and Jetson numbers are a
"needs hardware" item.

- **Detector-free first.** Phase 1 uses only what localization already
  computes (matches, inliers, the predicted pose). Its added cost must stay
  under **2 ms per localized image** on the 4-core container (today's
  median is 31–75 ms) and add no per-frame allocation in the hot loop.
- **The detector is opt-in and paced** (Phase 2). YOLOX-nano at 416 px
  takes about 85 ms on the development CPU, which is too much per frame.
  - Run it only on the images that are localized (`--localize-every N`),
    not on every VIO frame.
  - Run it only on demand: when Phase 1's signals say the view looks
    disturbed (low inlier spread, high occlusion ratio, rejected fixes).
  - Between detections, carry the boxes forward with ByteTrack's Kalman
    prediction (0.03 ms).
  - On the Jetson, run it on the GPU through `onnx-cuda` (TensorRT execution
    provider as a follow-up), at a lower input size if needed.
  - Budget: at most **one detection per second on average** and **no more
    than 15% extra CPU** at the default settings, measured on the
    container; Jetson measured once hardware is available.
- **Map side costs nothing at query time.** Landmark stability (Phase 1)
  is a per-landmark `u8`/`f32` stored with the map. Map-time masking
  (Phase 2) runs offline at `--save-map`.
- Every new switch defaults to off; with it off, outputs stay bit-identical
  (the EuRoC and OpenLORIS localization runs, as in the existing parity
  checks).

## Phases

### Phase 0: an indoor benchmark with moving people (before any lever)

Data, indoor only:

- **TUM RGB-D fr3 dynamic sequences.** Map from `fr3/sitting_static` or
  `fr3/long_office_household` (`tum_rgbd_save_map`), query
  `fr3/walking_{xyz,static,halfsphere,rpy}` (two people walking right in
  front of the camera) and `fr3/sitting_*` (people moving a little).
  Mocap ground truth.
- **Bonn RGB-D Dynamic.** Maps from its static sequences, queries from the
  crowd, balloon, moving-box and person-tracking sequences. Ground truth
  from mocap.
- **OpenLORIS-Scene cafe** (people moving around a café; the cafe1-1 map
  already exists, [openloris_map_localization.md](openloris_map_localization.md))
  and, if the 41.5 GB download is worth it, market.
- **Gazebo apartment and Depot** with walking actors (Gazebo's animated
  actors on scripted paths), so the robot loop and Nav2 can be scored with
  people in the way. The sim also writes per-pixel actor masks (segmentation
  camera), so the share of inliers on people is known exactly.
  - Scenarios: 0 / 2 / 5 / 10 people; one person walking just ahead of
    the robot; a person standing in a doorway; and **a map built with
    boxes and chairs that are moved or gone at query time** (F3).

Replay offline (`basalt_euroc_localize_demo --vio` or `--odometry`) with
several seeds, so Nav2's timing does not add to the spread.

Metrics per run:

- share of images localized, and of frames with a map pose;
- longest stretch without a fix;
- pose error RMSE / median / max;
- **wrong fixes accepted** (> 10 cm off) and resets of the correction;
- inliers on movers (from the masks), inlier spread over the image;
- added time per localized image, and peak memory.

Deliverable: a table of today's code on these scenarios, with each failure
put down to F1–F5. Levers are chosen from that table, not before it.

### Phase 1: detector-free levers

1. **Inlier spread.** Score a fix by how many cells of a coarse image grid
   (for example 4 × 4) hold inliers, not only by the inlier count. A fix
   whose inliers sit in one blob becomes a weak fix (it needs agreeing
   fixes to be used). Against F2 and F5.
2. **Matches that disagree with the prediction, dropped as a group.** With
   a fresh prediction, matches whose reprojection error is large and that
   cluster in the image (a person-sized blob) are left out before RANSAC.
   A detector-free mover mask, from what is already computed. Against F2.
3. **Occluded, not lost.** Count the landmarks the predicted camera should
   see that found no match, and where in the image they are. When the
   missing ones are concentrated (something stands in front) while the rest
   match, keep the correction on the odometry rather than treating it as
   lost or resetting; lower the bar for a weak fix from the visible part.
   Against F1.
4. **Landmark stability in the map.** A score per landmark from the
   keyframes and the time span that observed it and the sessions it appears
   in (with `basalt_map_merge` appending sessions). Localization runs then
   decay it when the landmark is predicted visible and unoccluded but is
   never matched. Low-stability landmarks get less weight in RANSAC scoring
   or are dropped from the search. Against F3; the map gets cleaner with use.
5. **A stricter replacement rule.** Three agreeing rejected fixes may
   replace `map -> odom` only if they are spread over the image and over
   stable landmarks. Against F5.

Each lever is one switch, off by default, measured alone and then
together on the Phase 0 table.

### Phase 2: detector boxes (opt-in, `onnx-inference` / `onnx-cuda`)

- `visloc-objects` (YOLOX-nano + ByteTrack), paced as in the budget above.
- **Query side:** drop keypoints in the boxes of movable classes (person;
  indoors also chair, cart, bag where the detector knows them). Unlike the
  VIO filter, every movable box is a candidate, not only the moving ones.
  Keypoints in a box that match high-stability landmarks are kept (a chair
  that never moves).
- **Map side:** the same boxes at `--save-map`, so people present while
  mapping never become landmarks. Offline, so cost is no issue there.
- **Odometry:** pass the same boxes to the VIO (`set_object_boxes`), which
  costs nothing extra once they exist.
- Measure what the boxes add over Phase 1 alone. If Phase 1 already closes
  most of the gap, the detector stays a map-building tool and the query
  side stays detector-free on the Jetson.

### Phase 3: on the robot

- `visloc_map_localize_node`: the Phase 1 switches as parameters, and an
  optional box topic (from an external detector node, so a Jetson can run
  the detector on the GPU with TensorRT outside visloc).
- Nav2 in the Gazebo apartment and Depot with actors: goals reached,
  contacts, pose error, against the empty runs.
- A Jetson run of the node with and without each lever: time per image,
  CPU, memory (needs hardware).
- Results into [navigation.md](navigation.md) /
  [openloris_map_localization.md](openloris_map_localization.md), and the
  README sensor table.

## Targets (to be fixed once Phase 0 has the numbers)

On the indoor scenarios with the most people, against today's code:

- halve the loss in the share of images localized;
- **no fix more than 10 cm off accepted**;
- halve the longest stretch without a fix;
- no regression on EuRoC and OpenLORIS office (bit-identical with the
  switches off; within the run-to-run spread with them on);
- within the resource budget above.

## Risks

- **The spread hides the effect.** Several seeds per setting, medians, and
  offline replay rather than live runs.
- **TUM and Bonn are handheld RGB-D, not a robot.** Their maps come from
  RGB + depth, not from the stereo-inertial SLAM; the matching and
  correction under test are the same, the odometry differs.
- **Static-looking movers.** A person standing still for minutes, or a
  chair, looks like structure. Stability over sessions (Phase 1.4) is the
  answer, not the per-frame tests.
- **The detector on the Jetson.** If even paced detection does not fit,
  the detector stays offline (map hygiene only).
