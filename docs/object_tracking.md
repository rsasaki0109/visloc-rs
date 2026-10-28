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

- **The ground:** `estimate_ground_height` takes the triangulated scene
  points (static, so metric) and returns the mean height of the densest
  5 cm layer at least 0.3 m below the lowest camera position. The world is
  gravity-aligned, so the ground is the plane `z = ground`.
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
  and distance, and the speed of a straight line fitted to the foot's path.
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
  - the floor is found under a table top and walls.

`objects_metric_offline` also places the boxes of the movable COCO classes
(people, vehicles, animals) and writes `ground_objects.csv`: per 2D track,
the frames used, the camera's height above the ground, how far the ground
under the object lies from the camera's (`ground_shift`), the median foot
position, height, width, distance and speed, and the height flag.
- Boxes scored below 0.5, or within 2 px of the image border (the foot may
  be cut off), are skipped. A track needs 1 s.
- The ground uses the scene points seen within 10 s of the track.

### LaMAria R_08_hard

The same 746 m walk through Zurich as above (Phase 2), with its YOLOX boxes
(`--rotate 90`). Feature tracks are dumped from the VIO with the LaMAria
config (`euroc_config_big_window_multicam.json` for stereo,
`euroc_config_big_window.json` with `--mono`).

| VIO | Trajectory ATE | People: height, median (IQR) | Cars: height | Bicycles: height | Walking speed | Moving cars |
| --- | --- | --- | --- | --- | --- | --- |
| Stereo + IMU | 0.92 m (scale 0.9994) | 1.67 m (1.58–1.78), n = 24 | 1.44 m (1.31–1.57), n = 34 | 1.08 m (0.99–1.12), n = 15 | 1.19 m/s (18 people) | 6.6 m/s (20 cars) |
| Mono + IMU | 3.74 m (scale 0.982) | 1.70 m (1.59–1.82) | 1.46 m (1.35–1.60) | 1.09 m (1.03–1.20) | 1.23 m/s | 6.6 m/s |

- **Plausible:** the people's median is a typical adult height, cars come
  out at a typical 1.4–1.6 m, and the parked bicycles at 1.0–1.1 m. People
  walk at 1.2 m/s, and cars drive at about 24 km/h in town.
- **Mono + IMU against stereo + IMU** (the same 2D tracks): the heights
  agree to a median of 7 % for people (2.3 % bias), 3 % for cars and 2 %
  for bicycles; the speeds to 0.07 m/s.
- **The camera is low:** about 1.1–1.2 m above the ground. Two checks that
  do not depend on the ground estimate: the camera-height line from the
  VIO's gravity runs through the road's vanishing point, and it crosses the
  people walking ahead at about two thirds of their height, as a camera
  1.1–1.2 m up would. In stretches of 10–40 s the camera drops another 0.55 m (the world
  height of the camera falls while the ground's stays), and the ground
  estimate follows.
- **The ground under the object** moved 7 people by 0.1–0.6 m, all down
  (steps and kerbs below the camera's ground). Without it they would come
  out at 0.8–1.6 m; with it they are 1.5–1.9 m.
- **Flagged:** 5 of the 24 people with stereo + IMU (3 with mono + IMU,
  all at stairs or the ramp).
  - Three at the foot of stairs or a ramp (0.6–0.8 m). The points under
    them do lie lower, but spread over the steps or the slope, so no layer
    stands out. Taking their median instead fixes one of the three and
    breaks five others.
  - One 1.7 m from the camera (2.2 m), one 8.4 m away (2.3 m).
- **Still objects** show a residual speed of 0.1–0.5 m/s: as the camera
  passes, the box's bottom edge slides along a car. Below about 0.6 m/s,
  moving and still cannot be told apart.
- **Tuning:** the local-ground settings were chosen on this sequence. No
  other sequence with people has been run yet.
