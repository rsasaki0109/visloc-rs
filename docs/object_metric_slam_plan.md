# Metric-scale objects from monocular VI-SLAM (plan)

**Status:**
- Phase 0 is done (go).
- Phase 1 (2D detection and tracking) is in `pipelines/objects`
  (`visloc-objects`).
- Phase 2's moving-object filter is in `visloc-basalt`. It works on synthetic
  data and showed no gain on a LaMAria walk.
- Phase 3 has static objects (triangulation and upright boxes,
  `visloc_objects::metric`) and objects on the ground, moving or not
  (ground contact, `visloc_objects::ground`). The `objects_metric_offline`
  example runs both on a VIO run.
- Phase 4 has KITTI raw: object heights and distances against the
  tracklets, for mono + IMU, stereo + IMU and the OXTS poses
  (`scripts/run_kitti_objects_eval.sh`).

See [`object_tracking.md`](object_tracking.md). The world-frame EKF and the
per-frame output of Phase 3, and the later phases, are not implemented yet.

## Goal

Recover the **size, position and speed of objects in metres** from a single
camera and an IMU. The IMU is what makes this possible:

- **Monocular VO alone** cannot separate an object twice as large and twice
  as far away. Its trajectory, and everything triangulated from it, is up to
  scale (the visual-only mode of #298 is in this class).
- **With an IMU**, the accelerometer makes the VIO trajectory metric, so
  object points triangulated from it are metric too.
- **Gravity** from the IMU gives a vertical axis. Object boxes are fitted
  upright, and the ground plane can be found. The ground plane is what fixes
  the scale of *moving* objects (Phase 3).

The demo shows the same clip twice: with the IMU off the numbers are
meaningless, with it on each box is labelled like "1.72 m tall, 4.3 m away,
1.2 m/s".

Rejecting dynamic objects in the VIO (Phase 2) comes first because it
protects the metric trajectory that everything else stands on. It may also
help LaMAria's Moving track, the only track where we trail GroundedSLAM.

## Phases

### Phase 0: prerequisites (go / no-go)

The headline stands or falls with the **metric scale of monocular-inertial
VIO**. `docs/mono_inertial_vio.md` lists that mode as experimental, with no
EuRoC result yet. Object sizes cannot be more accurate than the trajectory's
scale, so this number is the ceiling.

- Measure, on all EuRoC sequences with `basalt_euroc_vio_demo --mono`:
  - the scale error (Sim3 vs SE3 alignment against ground truth);
  - the time to scale convergence after start-up;
  - ATE.
- Run stereo-inertial on the same sequences as the upper bound.
- How: the **EuRoC mono scale** workflow
  (`.github/workflows/euroc_mono_scale.yml`) runs both modes on every
  sequence and tabulates them; see `docs/mono_inertial_vio.md`.
- **Go** if the scale error is a few percent after convergence. Otherwise,
  fix mono-inertial first: that becomes the project.
- **Result: go.** The ETH Research Collection answers GitHub's runners with
  HTTP 429 only, so the workflow ran on TUM-VI room1–6 instead (motion-capture
  ground truth over each whole sequence). Mono-inertial scale error: median
  1.3 %, at most 3.4 %; every 10 s window within ±5 % from the start; ATE
  at most 0.014 m above stereo-inertial's. Table in `docs/mono_inertial_vio.md`.
  EuRoC stays a cross-check for when its server lets the runners in.
- Detector: an ONNX model through the existing `ort` support in
  `crates/vision`, the way SuperPoint and DPVO are loaded. The model must be
  Apache-2.0 or similar: RT-DETR or YOLOX. Avoid Ultralytics YOLOv8/11
  (AGPL-3.0).
- Put the detector behind a trait, so tests can feed ground-truth or dummy
  detections (the repository's `*_dummy` pattern).

### Phase 1: 2D multi-object tracking

- A ByteTrack-style tracker in pure Rust: Kalman filter on boxes, IoU cost,
  Hungarian assignment, low-score second pass.
- New crate `pipelines/objects`. The existing `pipelines/tracking` crate is
  camera tracking against a map, so a separate name avoids confusion.
- Tests on synthetic box sequences: ID switches, occlusion, re-entry.
- **Done: the tracker.** `visloc_objects::ByteTracker` follows the reference
  `byte_tracker.py`. It uses the same box Kalman filter, IoU costs fused
  with the score, and lapjv-style assignment with a cost limit (here
  Hungarian on the extended matrix). It also has the three associations, the
  one-second lost buffer and duplicate removal. Two additions: class-aware
  association (on by default), and IDs given at confirmation, so they have
  no gaps. `MotAccumulator` computes CLEAR MOT.
- The tests in `pipelines/objects/tests/byte_track_synthetic.rs` cover:
  - steady motion with no ID switches;
  - a crossing where the hidden object's score drops to 0.3: tracked
    throughout, and missed for exactly the occluded frames without the
    low-score pass;
  - a 15-frame occlusion that keeps the ID, and a 40-frame absence that gets
    a new one;
  - one-off false positives that are never reported;
  - class gating.
- On a 12-object crowd with 3 % box noise, 5 % missed detections, 10 %
  low-score detections and a false positive per frame, it scores MOTA 0.94
  with no ID switches.
- **Done: the detector.** `ObjectDetector` is the boundary.
  - `ReplayDetector` replays stored boxes.
  - `YoloxOnnxDetector` runs the official YOLOX ONNX models (Apache-2.0)
    through the `onnx-inference` feature. YOLOX won over RT-DETR because its
    models need no export step. Its pre- and post-processing are pure Rust and
    match a Python reference to within 5e-4 px.
  - Examples: `track_objects_dummy` and `yolox_track_demo`.

### Phase 2: dynamic-object rejection in the VIO

- Drop KLT tracks inside the boxes of objects that are **actually moving**
  from the Basalt estimator. Movable classes (person, car, ...) are only the
  candidates. A parked car is static structure and is kept.
- "Moving" is decided from the residuals of the points inside the box against
  the VIO pose: reprojection error for points with a landmark, epipolar error
  for the rest.
- Opt-in config key, default off. With it off, outputs must stay
  bit-identical (`trajectory.tum` sha256, as `scripts/bench/basalt_vio_parity.sh`
  checks).
- Evaluate on LaMAria Moving (Score up?) and EuRoC (no regression).
- **Done: the filter.** It is in `visloc_basalt::dynamic_objects` and runs
  inside the estimator, right after the IMU prediction, so it covers the
  serial, pipelined and stage-split paths alike.
  - It is opt-in by supplying boxes (`set_object_boxes`,
    `basalt_euroc_vio_demo --object-boxes`) rather than by a config key.
    Without boxes the code path is skipped, so the output is unchanged.
  - The test compares a box's median error with the background's, against
    the IMU prediction. Tracks without a landmark use their oldest
    observation in the window, because an object moving along with the
    camera shows almost no frame-to-frame epipolar error.
  - Moving objects stay out for 0.5 s after their last moving decision.
  - The synthetic test has a person-sized object walking with the camera.
    It raises the RMS position error from 7.8 mm to 0.58 m, and the filter
    brings it back to 12 mm. A box over static points ("a parked car")
    leaves the result bit-identical. See `docs/object_tracking.md`.
- **Real data, LaMAria R_08_hard (746 m through streets, about 400
  tracked people and cars): no gain.** SE3 ATE goes from 0.922 m to 0.938 m.
  Only 289 of 5843 boxes were judged moving, and 3101 observations were
  dropped. Moving objects are too small a share of the view on a walk for
  the filter to matter. Details in `docs/object_tracking.md`. The filter
  stays opt-in.
- LaMAria's "Moving" track is a moving *platform* (the wearer rides a
  vehicle), which `config.vio_platform_filter` handles, not moving objects.

### Phase 3: metric 3D objects (the core)

- **Static objects:** triangulate the feature tracks inside the box with the
  metric VIO poses, then fit a gravity-aligned box. The result is width,
  depth, height and position in metres.
- **Moving objects:** from one moving camera, triangulation alone does not
  fix the scale of an object that moves too. Use **ground contact**:
  - Get the ground plane in metres from VIO landmarks and the gravity
    direction.
  - Put the bottom of the box on the ground plane. The ray through the box's
    bottom edge then fixes the distance, and the size follows.
  - Track the object as a rigid body. A constant-velocity EKF in the world
    frame gives position and velocity.
- **Honesty rule:** class size priors ("a person is about 1.7 m") are used
  **only as a sanity check**, never as the source of scale. Otherwise the
  "metric thanks to the IMU" claim would be hollow.
  - Each estimate records where its scale came from: `triangulated` or
    `ground_contact`.
  - An estimate far from its class prior is flagged.
- Output: per-object tracks (ID, class, timestamp, centre, size, yaw,
  velocity, scale source) in a TUM-like text format.
- **Done: static objects.**
  - `visloc_objects::metric` triangulates the tracks and fits the box, and
    `objects_metric_offline` runs it on a VIO run's dumped tracks.
  - On TUM-VI room1, a whiteboard comes out at 2.30 × 1.77 m with mono +
    IMU, 2.21 × 1.76 m with stereo + IMU (an independent scale), and
    0.60 × 0.80 without the IMU (no scale).
- **Done: objects on the ground (moving or not).**
  - `visloc_objects::ground` finds the ground from the scene points, puts
    the bottom edge of each box on it (lowest in the world, so sideways
    cameras work), moves the ground to the level under the object (steps,
    kerbs), and fits the speed to the foot's path. The scale source is
    `ground_contact`.
  - On LaMAria R_08_hard with mono + IMU, 24 people come out at a median
    1.70 m (IQR 1.59–1.82) walking at 1.23 m/s, 34 cars at 1.46 m tall,
    with moving ones at 6.6 m/s, and 15 parked bicycles at 1.09 m. Stereo +
    IMU agrees to 7 % (people), 3 % (cars) and 2 % (bicycles) in height.
    3 people are flagged (5 with stereo), most at the foot of stairs or a
    ramp.
  - Next: the world-frame EKF (the speed is a line fit for now), and
    per-frame output.

### Phase 4: metric accuracy evaluation

- **KITTI raw:** monocular camera plus the OXTS IMU, with 3D box ground truth
  (sizes, positions) from the KITTI tracking labels. Report size, position and
  velocity errors in metres. This is the main evaluation.
- **Own capture:** phone recordings of objects measured with a tape measure
  (desk, boxes, a car). Smaller but easy to show, and it doubles as demo
  material.
- **Synthetic tests:** scenes with known boxes. The recovered sizes must be
  within tolerance. These run in CI.
- **Comparison table:** monocular only (scale unknown), mono + IMU and
  stereo + IMU, each with size, position and velocity error.

- **Done: KITTI raw** (`object_tracking.md`, "Metric accuracy on KITTI
  raw").
  - 7 drives with tracklets. The VIO starts at rest: 6 drives start at rest
    or are cut at a stop.
  - Heights of cars (110 on 5 drives): 11 % (0.16 m) median error with
    mono + IMU, 6 % with stereo + IMU, 7 % with the OXTS poses. Pedestrians
    (18): 4 % with stereo + IMU.
  - The path ground came out of this: a flat ground put cars 33 % off on a
    falling road.
  - Speed is not good enough yet (0.5–2.7 m/s), and mono + IMU fails from a
    moving start.
  - Next: own tape-measured captures, the comparison table with mono only.

### Phase 5: demo

1. **Offline GIF for the README hero.** Left: the image with boxes, IDs and
   metric labels, and features coloured static or dynamic. Right: a top-down
   map with a metre grid, the camera trajectory and the object tracks. A
   side-by-side with the IMU off. Made the same way as the existing hero GIFs.
2. **Live on a phone,** on top of the browser VIO of #311 (phone camera and
   IMU, mono + IMU exactly). Metric labels as an AR overlay. The detector in
   the browser (WebGPU) is the heavy part, so this comes second, and it
   depends on #311 landing.
3. **ROS 2 (optional):** a node publishing the objects with their sizes as a
   `MarkerArray`, in `visloc-ros2`.

### Phase 6 (stretch): joint optimization

Object poses in the sliding window with rigid-motion factors, in the spirit of
VDO-SLAM and DynaSLAM II. Only if Phase 3 shows a clear benefit.

## Decisions

- Order: Phase 0 first, as a go / no-go on the mono-inertial scale.
- Evaluation: KITTI raw as the main benchmark, plus own tape-measured
  captures and synthetic tests.
- Demo: the offline GIF first, then the live phone demo on #311.

## Risks

- **Mono-inertial scale.** Measured on TUM-VI room1–6 (Phase 0): at most
  3.4 % off, indoor and hand-held. In a car (KITTI raw, Phase 4): 2–9 % off
  from a start at rest, and four times too small from a start at 7 m/s.
- **Scale at start-up.** Mono-inertial needs motion to observe scale. Object
  estimates made before convergence are marked as such.
- **Ground contact** fails for objects not on the ground (things carried,
  stacked, or on tables) and on slopes. These fall back to `triangulated`
  when the object is static, or are reported without metric size.
- **Detector licences and size** (ONNX weights are not bundled, like the
  existing ONNX models).
