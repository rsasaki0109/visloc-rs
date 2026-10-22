# Metric-scale objects from monocular VI-SLAM (plan)

**Status: plan, not started.** Nothing below is implemented yet.

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
- **Go** if the scale error is a few percent after convergence. Otherwise,
  fix mono-inertial first: that becomes the project.
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

- **Mono-inertial scale.** Not yet measured on real data (Phase 0).
- **Scale at start-up.** Mono-inertial needs motion to observe scale. Object
  estimates made before convergence are marked as such.
- **Ground contact** fails for objects not on the ground (things carried,
  stacked, or on tables) and on slopes. These fall back to `triangulated`
  when the object is static, or are reported without metric size.
- **Detector licences and size** (ONNX weights are not bundled, like the
  existing ONNX models).
