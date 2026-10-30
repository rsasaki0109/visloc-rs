# 2D object detection and tracking

`visloc-objects` (`pipelines/objects`, also `visloc_rs::objects`) finds and
tracks objects in image space. It is Phase 1 of the
[metric-object plan](object_metric_slam_plan.md). The later phases turn
these tracks into metric 3D objects with the VIO.

- **Detection:** `ObjectDetector` is the boundary.
  - `YoloxOnnxDetector` runs an official YOLOX model with ONNX Runtime
    (feature `onnx-inference`).
  - `ReplayDetector` returns stored boxes: ground truth, a detector's saved
    output, or synthetic boxes in tests.
- **Tracking:** `ByteTracker` is ByteTrack with class-aware association; see
  the crate documentation.
- **Scoring:** `MotAccumulator` computes CLEAR MOT (MOTA, ID switches,
  misses, false positives, mean IoU).

## Examples

- `cargo run --example track_objects_dummy` tracks three synthetic people.
  Two of them cross, and the hidden one's score drops to 0.3. The example
  checks that nothing is missed or switched (MOTA 1.0) and writes
  MOTChallenge-format tracks. CI runs it (`scripts/run_examples.sh`).
- `yolox_track_demo` runs YOLOX and ByteTrack on a directory of frames. It
  writes `tracks.txt` (`frame,id,x,y,w,h,score,class,-1,-1`) and, with
  `--draw`, every frame with its track boxes and IDs:

  ```sh
  curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
  ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo run --release \
    --features image-io,onnx-inference --example yolox_track_demo -- \
    --model yolox_s.onnx --images frames/ --out-dir target/yolox_track --draw
  ```

  For `yolox_nano` and `yolox_tiny`, add `--input-size 416`. `ort` loads ONNX
  Runtime at run time: set `ORT_DYLIB_PATH`, for example to the
  `libonnxruntime.so` that the `onnxruntime` Python wheel ships.

## YOLOX

The models are Apache-2.0 and are not bundled. The two that were checked are:

| Model | Input | sha256 |
| --- | --- | --- |
| `yolox_nano.onnx` | 416 × 416 | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| `yolox_s.onnx` | 640 × 640 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

Both come from release `0.1.1rc0`. Pre- and post-processing are in
`visloc_objects::yolox` and follow YOLOX's `demo/ONNXRuntime`:

- **Pre-processing:** a letterbox (aspect ratio kept, top left, grey 114),
  BGR, values 0–255.
- **Post-processing:** decode the grid, score = objectness × class
  probability, then per-class NMS at IoU 0.45.

The score threshold is 0.1, low on purpose, because ByteTrack's second
association uses the low-score boxes.

The results match a NumPy re-implementation of that pipeline running on the
Python `onnxruntime` 1.29, on YOLOX's `assets/dog.jpg` (decoded once to PNG,
so both sides see the same pixels):

| Model | Detections (Rust / Python) | Largest box difference | Largest score difference |
| --- | --- | --- | --- |
| yolox_s | 7 / 7 | 4.8e-4 px | 4.0e-7 |
| yolox_nano | 8 / 8 | 5.0e-4 px | 4.0e-7 |

The ignored test `pipelines/objects/tests/yolox_onnx.rs` runs the detector on
that image. Its header has the command, and it checks that the dog, the
bicycle and the vehicle are found with a score above 0.5. On 40 frames panned
across the same image, `yolox_track_demo` with `yolox_s` keeps one ID each for
the bicycle, the dog and the car through all 40 frames, with no ID switches.
YOLOX also reports the car as a truck in 34 frames, as a separate track
because association is class-aware. On the development container's CPU,
detection takes about 85 ms per frame and tracking 0.03 ms.

## Moving objects in the VIO

`visloc_basalt::dynamic_objects` leaves the tracks of moving objects out of
the Basalt VIO (Phase 2 of the plan). It takes camera-0 boxes with stable
object IDs, per frame timestamp. Each box is only a candidate:
- **Scoring:** right after the IMU prediction of the frame, every camera-0
  observation is scored in pixels. A track with a landmark gets its
  reprojection error. A track without one gets the epipolar error against its
  oldest observation still in the window.
- **Moving:** a box whose median error is above 2 px and three times the
  median outside all boxes. Its tracks are dropped from that frame in every
  camera.
- **Hold:** a moving object stays dropped for 0.5 s after its last moving
  decision.
- **Parked objects:** they score like the background, so they are kept.

Producing the boxes and using them:

```sh
# boxes of people, vehicles and animals over the sequence's camera-0 frames
ORT_DYLIB_PATH=... cargo run --release --features image-io,onnx-inference \
  --example yolox_track_demo -- --model yolox_s.onnx \
  --images SEQ/mav0/cam0/data --out-dir target/seq_objects --boxes-csv boxes.csv
# the VIO with the filter
cargo run --release --example basalt_euroc_vio_demo -- --euroc-dir SEQ \
  --calibration CALIB --config CONFIG --out-dir target/seq_vio --object-boxes boxes.csv
```

`summary.txt` then has the counts: `object_box_frames`, `object_boxes`,
`moving_object_boxes`, `undecided_object_boxes` and
`moving_object_dropped_observations`.

Without `--object-boxes` nothing changes. On TUM-VI room1, `trajectory.tum`
has the same sha256 as on `main` in stereo, `--mono` and `--pipeline`. With
the boxes `yolox_s` finds on room1, the only movable class is one false
"person" at the bottom edge for 5 frames. It has too few points to decide,
so it is left in and the trajectory is again unchanged.

`pipelines/basalt/tests/dynamic_objects_synthetic.rs` checks the filter on a
monocular-inertial replay. A camera moves through a static scene, and a rigid
object of 40 points stays about 3.5 m in front of it (someone walking
alongside, or a car ahead at the same speed). That is the hardest case: the
object barely moves in the image. Its points agree with one another but not
with the scene. The results are for 6 s at 20 Hz:

| Run | RMS position error |
| --- | ---: |
| Static scene | 7.8 mm |
| With the object | 0.58 m |
| With the object and its boxes | 12 mm |
| Static scene with a box over static points | 7.8 mm (bit-identical) |

With the filter, the object is caught from the third frame. The remaining
4 mm come from those first frames, and from keyframe decisions that change
when its tracks are left out.

The first version scored against the previous frame only. It caught the
object much later, at an RMS error of 0.13 m. Under pure translation, a
point moving with the camera looks like a point at infinity, so only the
accumulated rotation reveals it.

### On real data: LaMAria R_08_hard

LaMAria's training sequence R_08_hard is a 746 m walk through Zurich's
streets, with its pseudo-dense ground truth. Both runs used the driver's
default setup: `scripts/run_lamaria_test_submission.py --split training`,
the multi-camera big-window config, `--pipeline`, and the binary from
`e9b5ea3`.

- **Detection:** Aria's SLAM cameras are mounted sideways, so YOLOX needs
  `--rotate 90`. Without it, it finds people in 1 of 24 sampled frames; with
  it, in 9. On the whole sequence, `yolox_s` gives boxes of movable classes
  in 3895 of 12328 frames. That is 5843 boxes, mostly cars (201 tracks),
  people (189) and bicycles (30).
- **Result:**

| Run | SE3 ATE | Sim3 ATE | Median error |
| --- | ---: | ---: | ---: |
| Without boxes | 0.922 m | 0.918 m | 0.561 m |
| With the boxes | 0.938 m | 0.935 m | 0.586 m |

- **The filter changed little, and slightly for the worse.**
  - Of the 5843 boxes it judged 289 moving and left 4267 undecided, too
    small for 5 scored points. It dropped 3101 observations.
  - The trajectories are identical up to pose 2996 and differ by at most
    0.27 m after.
  - On this walk, moving objects are a small share of the features, and
    Basalt's robust losses already cope with them. The filter pays off when
    they dominate the view, as in the synthetic test.
  - Telling whether its 289 moving decisions were right would need a look
    at those frames. A false positive on a static object only costs
    features, but the result does not show a gain to pay for that.

## Metric objects (static)

`visloc_objects::metric` turns object tracks and metric camera poses into
metric 3D objects. This is the start of Phase 3 of the plan: the headline
that the IMU makes object sizes metric.

- **Triangulation:** `triangulate` intersects a feature's observation rays
  (camera pose plus unit bearing) by least squares. It rejects tracks with
  less than 2° of parallax, a point behind a camera, or a ray more than 0.3°
  off the point.
- **Box fit:** `fit_upright_box` fits a gravity-aligned box (world z up, as
  in Basalt). It drops points more than 3 MADs from the median point. The
  heading is the one whose 2–98 % quantile rectangle has the smallest area,
  because the main axis of the scatter is unstable for near-square
  footprints.
- **Scale source:** each `ObjectEstimate` records it
  (`ScaleSource::Triangulated`).
- **Class priors:** `plausible_size` holds rough COCO size ranges. They
  only flag an estimate and never set a scale.
- **Tests:**
  - a chair-sized box walked past at 3 m with 0.5 px noise comes out within
    5 cm;
  - the same walk with its trajectory doubled gives a box twice as large,
    which the prior flags.

`objects_metric_offline` runs this on a VIO run. Its inputs are the dumped
feature tracks (`basalt_euroc_vio_demo --dump-tracks`), the trajectory, the
calibration and the 2D object boxes (`yolox_track_demo --boxes-csv FILE
--all-classes`).
- A track belongs to the object whose camera-0 boxes contain it at least 3
  times.
- Objects of the same class whose boxes lie within 0.5 m are merged and
  fitted again, because the 2D tracker gives an object a new ID each time it
  comes back into view.
- Objects with fewer than 12 points are not reported.

### TUM-VI room1

All runs use the same YOLOX boxes and the same dumped feature tracks, and
`--trajectory-is-camera` for the visual-only one. YOLOX's largest steady
object in the room is a wall-mounted whiteboard, which it calls "tv". The 2D
tracker split it into 10 IDs, and the merge puts them back together.

| VIO | Whiteboard, length × depth × height | Centre (m) |
| --- | --- | --- |
| Mono + IMU | 2.30 × 0.30 × 1.77 m | (1.93, 0.24, 0.01) |
| Stereo + IMU (an independent metric scale) | 2.21 × 0.25 × 1.76 m | (1.96, 0.27, 0.00) |
| Mono, no IMU (`--mono-vo`) | 0.60 × 0.37 × 0.80 (no scale) | (−0.65, −0.03, −0.15) in its own frame |

- **Mono + IMU against stereo + IMU:** they agree to 4 % in length, 0.6 % in
  height and 4 cm in position. On room1 the mono-inertial trajectory's scale
  is 0.4 % off the motion capture (see `mono_inertial_vio.md`).
- **Without the IMU:** the same board comes out at about a third of its
  size, at whatever scale the visual-only bootstrap picked.
- **Depth:** the box is 0.25–0.30 m deep where a flat board has none. Some
  wall points beside the board pass as inliers.
- **Class prior:** the "tv" range of `plausible_size` is 0.3–2.0 m, so the
  2.3 m board is flagged. That is right for a TV, and the board is not one.
  The flag does not change the estimate.
- **Not reported:** the only other objects with enough boxes, two chairs
  half hidden under a desk, have 6 points each.

## Metric objects on the ground (moving)

A moving object cannot be triangulated: while the camera moves, so does the
object, and its rays do not meet. `visloc_objects::ground` places such an
object on the ground instead.

- **The ground:** `PathGround` uses the triangulated scene points (static,
  so metric) in the gravity-aligned world.
  - The camera moves over the ground, so each point's height is measured
    from the camera where it passed nearest. On a road that climbs or falls,
    the ground then stays one sharp layer below the path, at the camera's
    height above it.
  - That height is the densest 15 cm band at least 0.3 m down, taken for
    each 20 m stretch of path either way. A camera whose height changes (a
    hand lowering it) is followed too.
  - The ground under a spot is that height below the nearest point of the
    path.
  - `estimate_ground_height`, one flat plane from the densest 5 cm layer
    below the camera, is the simpler version. On KITTI 0084 (4.5 m down in
    40 s) it smeared the road over many layers and put cars 33 % off in
    height (below).
- **The foot:** `BoxRays::from_box` unprojects the middles of the 2D box's
  edges and its corners into world rays. The edge-middle ray that points
  lowest is the bottom of the box, so this works for a camera mounted
  sideways (Aria) as well. Where it meets the ground is the object's foot
  (`place_on_ground`).
- **The size:** the box's rays cut the vertical plane through the foot,
  facing the camera. Their top gives the height above the ground and their
  sideways extent the width.
- **The ground under the object:** steps, a ramp or a kerb can put the
  object on another level than the camera's ground. A lower level makes it
  look small and near, a higher one large and far. `place_on_local_ground`
  takes the 12 scene points horizontally nearest to the median foot (within
  1.5 m) and, if their densest layer lies more than 0.1 m away, moves the
  ground there and places the object again, up to 5 times. Nearest points
  rather than all within a radius, so that over steps the ground walks down
  one step at a time.
- **Over a track:** `summarize_ground_track` gives the median height, width
  and distance, and the velocity of a straight line fitted to the foot's
  path.
  - **Weights:** each foot is weighted by how far it can be off. Across the
    ray that is the angular noise (0.004 rad) times the distance. Along the
    ray it is that times `(h² + d²) / h`, metres far away. Both get half the
    object's width (at least 0.2 m) on top, for not knowing which point of
    the object the bottom edge falls on.
  - **Outliers:** feet more than a chi-square of 9.21 off the line are
    dropped and the line fitted again, which handles a 2D track that jumps
    to a neighbour.
  - **Still or moving:** if the velocity does not stand out of the noise
    (chi-square 9.21, 1 %), the object is still and its speed is 0.
    `speed_sigma` is the speed's standard deviation.
- **Scale source:** `ScaleSource::GroundContact`. The scale is the camera's
  metric height above the ground, so it is as good as the VIO's.
- **Class priors:** `plausible_height` holds rough heights of the classes
  that stand on the ground (people 1.0–2.1 m, cars 1.2–2.1 m, bicycles
  0.8–1.4 m...). The height is what ground contact measures best; the width
  depends on the side the object is seen from. They only flag.
- **Error:** the distance is `h / tan(depression)` for a camera `h` above
  the ground. A pitch error `δ` changes it by about `δ · d / h`: 0.5° at
  10 m with the camera 1.1 m up is 8 %, at 5 m 4 %. A ground `δh` off
  changes every length by `δh / h`.
- **Tests:**
  - a person 1.75 m tall walking at 1.44 m/s about 5 m from a walking
    camera comes out at 1.75 ± 0.1 m and 1.44 ± 0.05 m/s, with the camera
    upright or rolled 90°;
  - the same walk with the trajectory doubled gives a person twice as tall
    and fast, which the prior flags;
  - a person at the bottom of a flight of steps 1 m down, or at the top of
    one 0.5 m up, comes out on the right level and at the right size;
  - the floor is found under a table top and walls;
  - a parked car whose box is 1.5 px loose, driven past from 40 m to 10 m,
    is still, although a plain line fit sees it approach; a car crossing at
    5 m/s moves at 5 ± 1 m/s;
  - a standing person whose 2D track jumps to a neighbour for 6 frames
    stays still.

`objects_metric_offline` also places the boxes of the movable COCO classes
(people, vehicles, animals) and writes `ground_objects.csv`: per 2D track,
the frames used, the camera's height above the ground, how far the ground
under the object lies from the camera's (`ground_shift`), the median foot
position, height, width, distance, speed, `speed_sigma` and `moving`, and
the height flag. `ground_tracks.csv` has every frame's foot and camera.
- Boxes scored below 0.5, or within 2 px of the image border (the foot may
  be cut off), are skipped. A track needs 1 s.
- The ground uses the scene points seen and the path driven within 10 s of
  the track. The object goes on the path ground under the camera first,
  then under its foot, then on the local ground.

### LaMAria R_08_hard

The same 746 m walk through Zurich as above (Phase 2), with its YOLOX boxes
(`--rotate 90`). Feature tracks are dumped from the VIO with the LaMAria
config (`euroc_config_big_window_multicam.json` for stereo,
`euroc_config_big_window.json` with `--mono`).

| VIO | Trajectory ATE | People: height, median (IQR) | Cars: height | Bicycles: height | Walking speed | Moving cars |
| --- | --- | --- | --- | --- | --- | --- |
| Stereo + IMU | 0.92 m (scale 0.9994) | 1.77 m (1.59–1.90), n = 24 | 1.49 m (1.35–1.63), n = 34 | 1.06 m (1.02–1.13), n = 15 | 1.17 m/s (17 people moving) | 6.1 m/s (15 cars) |
| Mono + IMU | 3.74 m (scale 0.982) | 1.79 m (1.60–1.89) | 1.48 m (1.34–1.63) | 1.06 m (1.03–1.21) | 1.16 m/s (15 people) | 6.2 m/s (14 cars) |

- **Plausible:** the people's median is a tall adult height, cars come out
  at a typical 1.4–1.6 m, and the parked bicycles at about 1.1 m. People
  walk at 1.2 m/s, and cars drive at about 22 km/h in town. There is no
  ground truth on LaMAria: KITTI (below) has it.
- **Mono + IMU against stereo + IMU** (the same 2D tracks): the heights
  agree to a median of 2.4 % for people, 2.7 % for cars and 3.6 % for
  bicycles, with no bias. They call the same objects moving or still in 75
  of 78 cases.
- **The flat ground of #385** gave people 5 % shorter (1.67 and 1.70 m) and
  mono and stereo 7 % apart. The path ground replaced it because it is
  better against KITTI's ground truth.
- **Without the IMU** there is nothing to compare. The visual-only VO
  (`--mono-vo`) stopped 526 s into the walk (a non-finite window solve at
  frame 10,519). It also has no gravity: its world is the first camera's
  frame, so its z is not up, and neither the ground nor the bottom of a box
  is defined. The IMU gives ground contact both its scale and its "down".
- **The camera is low:** about 1.1–1.2 m above the ground. Two checks that
  do not depend on the ground estimate: the camera-height line from the
  VIO's gravity runs through the road's vanishing point, and it crosses the
  people walking ahead at about two thirds of their height, as a camera
  1.1–1.2 m up would. In stretches of 10–40 s the camera drops another 0.55 m (the world
  height of the camera falls while the ground's stays), and the ground
  estimate follows.
- **The ground under the object** moved 7–9 people by 0.1–0.55 m, mostly
  down (steps and kerbs below the camera's ground).
- **Flagged:** 4 of the 24 people.
  - Two at the foot of stairs or a ramp (0.7 m). The points under them do
    lie lower, but spread over the steps or the slope, so no layer stands
    out. Taking their median instead fixes one such case and breaks five
    others.
  - Two 8–10 m away (2.2 m).
- **Still objects:** 14 of the 15 parked bicycles come out still (speed 0);
  the last one creeps at 0.2–0.3 m/s. Before the weighted fit, still
  objects showed up to 0.6 m/s, as the box's bottom edge slid along a car
  while the camera passed.
- **Tuning:** the local-ground settings were chosen on this sequence; KITTI
  (below) is the held-out check.

## Metric accuracy on KITTI raw (Phase 4)

KITTI raw has what LaMAria lacks: 3D boxes of the cars, vans, pedestrians
and cyclists (the tracklets), and a 100 Hz OXTS IMU. The evaluation scores
`ground_objects.csv` against them.

- **Data:** `scripts/fetch_kitti_raw_drive.py` takes only image_00/01 and
  the OXTS from a drive's zips (HTTP range requests), plus the calibration
  and tracklets. `scripts/kitti_raw_to_euroc.py` writes an EuRoC sequence:
  - the 100 Hz OXTS of the `_extract` zip as the IMU;
  - a Basalt calibration (pinhole from `P_rect`, `T_imu_cam` from the
    calibration chain, the stereo baseline);
  - the OXTS poses as `gt_imu.tum`.
- **Ground truth:** `scripts/kitti_tracklets_to_csv.py` puts every tracklet
  box in that world and in camera 0's image, each frame.
- **Scoring:** `scripts/evaluate_kitti_objects.py` matches each YOLOX track
  to the tracklet its box overlaps most (IoU ≥ 0.3, compatible class, half
  its frames). It then compares:
  - the height with the tracklet's `h`;
  - the distance with the distance to the tracklet footprint's near edge
    (what ground contact sees);
  - the speed with a line fitted to the tracklet's path;
  - with `--static`, the triangulated objects' box sizes.
- **Runs:** `scripts/run_kitti_objects_eval.sh WORK DRIVE START` runs the
  pipeline, for mono + IMU, stereo + IMU, and the OXTS poses (the method's
  own error, without VIO error). The pipeline is: fetch, convert, YOLOX,
  VIO (`configs/basalt/kitti/kitti_config.json`: four pyramid levels for
  10 Hz frames), objects, score.
- **Start at rest:** the mono-inertial VIO must start at rest. It sets the
  first velocity to zero, and a car moving at constant speed shows no
  acceleration to correct it. On 0091, which starts at 7 m/s, its scale
  stays four times too small. The drives used start at rest, or are cut to
  start at a stop.

| Drive | Start frame | Frames | Path | Mono + IMU ATE (Sim3 scale) | Stereo + IMU ATE (scale) | Objects scored |
| --- | --- | --- | --- | --- | --- | --- |
| 0084 | 0 | 383 | 239 m | 3.6 m (0.96) | 0.99 m (0.99) | 36 cars |
| 0022 | 0 | 800 | 515 m | 11.1 m (0.91) | 1.9 m (0.98) | 55 cars, vans and trucks |
| 0059 | 245 | 128 | 63 m | 0.9 m (1.02) | 0.42 m (0.98) | 17 cars |
| 0018 | 170 | 100 | 51 m | 0.5 m (0.97) | 0.16 m (0.99) | 3 |
| 0029 | 255 | 175 | 130 m | 3.0 m (0.91) | 0.65 m (0.98) | 1 |
| 0070 | 0 | 420 | 645 m | 6.8 m (0.97) | 1.8 m (0.99) | 2–3 |
| 0091 | 0 (moving) | 340 | 198 m | 39.8 m (3.93) | 0.69 m (0.99) | 18 pedestrians (stereo) |

Objects on the ground, all drives (cars and vans; pedestrians only in 0091,
so stereo):

| Poses | Cars: height error, median \|.\| (bias) | Cars: distance error | Pedestrians: height error | Pedestrians: distance error |
| --- | --- | --- | --- | --- |
| Mono + IMU (n = 110, 5 drives) | 0.16 m, 11 % (+1 %) | 19 % (+12 %) | – (moving start) | – |
| Stereo + IMU (n = 111) | 0.09 m, 6 % (−3 %) | 12 % (+6 %) | 0.07 m, 4 % (−4 %), n = 18 | 5 % (+3 %) |
| OXTS poses (n = 111) | 0.11 m, 7 % (−6 %) | 10 % (+3 %) | 0.10 m, 6 % (−6 %) | 4 % (−2 %) |

- **The height is metric to a few centimetres:** 7–11 cm for pedestrians
  and cars with stereo + IMU or the OXTS, 16 cm with mono + IMU, where the
  trajectory's scale is 2–9 % off.
- **The OXTS poses are no better than stereo + IMU.** What is left (6–7 %
  in height, a few percent low) is the method's own error. A 2D box is a
  coarse outline of a 3D box, and its top edge is the roof's far edge,
  seen at the near edge's distance.
- **The flat ground** of `estimate_ground_height` gave cars 31–33 % off in
  height and 34–41 % in distance on 0084, a road falling 4.5 m.
- **Speed is the weakest part.** Far away, a car's foot comes out a few
  percent near (−6 % at 25–40 m, −14 % beyond 40 m), and close by a little
  far. A still car the ego car drives towards therefore seems to approach,
  at 0.5–4 m/s with a plain line fit.
  - **Fix:** the weighted fit with the still test (above) brings still cars
    to a median 0.15–0.17 m/s, and calls 64–77 % of them still.
  - **All cars:** the median error falls from 0.53–0.85 m/s to 0.19–0.30
    m/s.
  - **Moving cars (20):** 2.1–3.1 m/s (22–35 %) off, and 2–3 are called
    still.
  - **Pedestrians:** on crowded 0091 they stay 0.6–1.1 m/s off (17 of them).
- **Triangulated (static) cars** come out 10–25 % off in length and width,
  and 16–23 % short in height, because only the faces the camera sees are
  tracked. For cars, ground contact is the better size.

### The demo GIF

`scripts/render_objects_gif.py` draws the README's GIF from one drive's run
(0084, mono + IMU):

- **Top:** cam0 with the YOLOX boxes. Each placed object is labelled with
  its track's height, its distance in that frame, and its speed (marked
  "avg": the track's line fit) or "parked". A height outside its class's
  range is grey with a question mark.
- **Bottom left:** a top view that follows the camera, heading up.
- **Bottom right:** the whole drive.
  - Mono + IMU is aligned to the OXTS by a rotation and a translation.
  - The visual-only VO (`basalt_euroc_vio_demo --mono-vo`, the same config)
    starts at 10.7 s, once the car moves. Its numbers are drawn as metres,
    rotated and moved onto the OXTS but not scaled. The 219 m from there
    come out 5.7 m: its Sim(3) scale is 38.6, and after that scale its ATE
    is 0.50 m.
  - Beside it, the camera's own speed from each.

```bash
python3 scripts/render_objects_gif.py --euroc WORK/0084/euroc \
    --boxes WORK/0084/yolox/boxes.csv \
    --objects WORK/0084/objects_mono/ground_objects.csv \
    --feet WORK/0084/objects_mono/ground_tracks.csv \
    --trajectory WORK/0084/run_mono/trajectory.tum \
    --visual-only WORK/0084/run_monovo/trajectory.tum \
    --matches WORK/0084/objects_mono/matches.csv \
    --out docs/assets/hero_metric_objects_kitti.gif
```
