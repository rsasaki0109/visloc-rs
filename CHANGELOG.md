# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Fixed

- **visloc-basalt: `vio_min_triangulation_dist` was ignored.** The VIO's
  landmark baseline gate was hard-coded at 5 cm; it now reads the
  configuration value, as upstream does. The pinned EuRoC file says 5 cm,
  so its results are unchanged.

- **visloc-basalt: `vio_obs_std_dev` and `vio_obs_huber_thresh` were
  ignored.** The visual residuals were weighted with a hard-coded 0.5 px
  standard deviation and 1 px Huber threshold; they now read the
  configuration, as upstream does. The pinned EuRoC file has the same
  values, so its results are unchanged.

- **ROS 2 nodes: relative topics were invisible to ROS 2.** `ros2-client`
  0.11 resolves a relative topic name in the root namespace to the DDS
  topic `rt//odom` instead of `rt/odom`. Only other `ros2-client` nodes
  saw those topics, so a ROS 2 subscriber to `/odom`, `/map_pose`,
  `/pose` or `/path`, or a camera driver publishing `/left/image_raw`,
  never connected; only the absolute `/tf` and `/diagnostics` did. The
  nodes now resolve names under their namespace themselves (`odom` is
  `/odom`, `/robot1/odom` with `-r __ns:=/robot1`).

### Added

- **README: map-based localization animation.** EuRoC MH_02 localized in
  the MH_01 map, with the matched keypoints and the rays to their map
  landmarks (`docs/assets/hero_map_localization.gif`). It is rendered by
  `scripts/render_map_localization_gif.py` from
  `basalt_euroc_localize_demo --dump-inliers FILE`, which writes each
  accepted localization's PnP inliers (keypoint pixel and map landmark).

- **Live monocular-inertial VIO in a phone browser** (`bindings/wasm`). The
  Basalt port's VIO runs as WebAssembly on the phone's camera and motion
  sensors, on iPhone Safari and Android Chrome.
  - **Calibration per phone:** Client Hints model or iPhone screen family,
    with intrinsics from the 35 mm-equivalent focal length. The camera-IMU
    time offset is estimated online and saved per phone (−25 ms until
    measured).
  - **Watchdog:** a diverged estimate is restarted and re-anchored to the
    last good pose.
  - **Page:** AR overlay, map, wasm threads when cross-origin isolated, and
    session recording and upload.
  - **Telemetry:** the page reports clocks, frame and IMU flow, worker
    state, resets, stalls and errors to the serving host
    (`telemetry/<session>.jsonl`). The sync estimate also measures the
    camera-IMU rotation.
  - **iOS:** WebKit's gyroscope axis order (alpha / beta / gamma about x /
    y / z), corrected automatically from the camera where a phone
    disagrees. The main lens is chosen over the virtual multi-camera.
  - **IMU only:** a button that freezes the map and moves it by the motion
    sensors alone, to check the camera-IMU calibration by eye.
  - **Phone defaults:** a 28 px feature grid, a 1 cm triangulation
    baseline, a keyframe allowed every 2 frames with 2 non-keyframe states
    in the window, bias random walks 30x stiffer than the white noise's
    scaling, and the first attitude from 0.5 s of motion samples instead of
    one. On ADVIO, against the EuRoC file's values: mean ATE 2.6 m instead
    of 4.7 m, 2 resets in four walks instead of 8. On three iPhone
    recordings: 93 landmarks in view against 85 before the keyframe change.
  - **Still start:** each start and restart waits (up to 2 s) for the
    phone to be held still, judged from the tracked points' motion after
    the gyroscope's rotation, and takes its attitude from the still frames.
    Basalt assumes a start at rest; on 60 starts over iPhone recordings
    with the phone moving, 27% diverged within 3 s. The page only corrects
    the camera-IMU time offset by 15 ms or more (estimates swing by ±8 ms,
    and each correction restarts tracking). `replay` reports why the
    frontend dropped tracks.
  - **Coasting:** frames without a landmark after some were seen report
    `"coast"`, and the page keeps drawing the IMU's pose ("Coasting on
    IMU") instead of showing "Initializing". ⚙ → "Film at 60 fps" captures
    at 60 fps (half the longest exposure, so less blur in fast moves).
  - **Frontend on the GPU:** the pyramid, the frame-to-frame KLT and the
    grid FAST run as WGSL compute shaders with wgpu (WebGPU in the
    browser, Vulkan or Metal natively; ⚙ → "Track points on the GPU", on
    where WebGPU exists), one submission and one readback a frame; the
    estimator stays on the CPU. From the same tracks it keeps and loses the
    same points as the CPU frontend (4 differ in 28,356; positions agree
    to 0.002 px), and end to end ADVIO stays at 2.65 m. A GPU error falls
    back to the CPU frontend. `replay` and
    `advio` take `--gpu`; `examples/gpu_check.rs` compares the two. The
    worker runs the GPU frontend one frame ahead of the estimator, so the
    GPU's round trip overlaps the estimate (same poses).
  - **60 Hz:** filming at 60 fps and tracking every frame are on by
    default (⚙ → "Film at 60 fps", "…and track every frame at 60 Hz").
    Tracking every frame of a 60 fps camera kept more landmarks in view on
    two iPhone recordings than the same frames at 30 Hz (172 against 94)
    without a reset, once the IMU noise fix below is in. Recordings keep
    every camera frame (`replay --every 2` replays one at 30 Hz), the rig
    carries `frameRateHz`, and `--estimate-every 2` estimates every other
    frame.
    The session's watchdog, still start and coasting are timed in seconds
    instead of frames.
  - **Android phones:** profiles for the Galaxy Note20 Ultra / Note20 and
    the Nothing Phone (4a) Pro (24 mm). The page takes the lowest-numbered
    back camera (Chrome's default on the Note20 Ultra was the ultra-wide),
    defaults the gyroscope's axis order by browser (Chrome on Android now
    reports WebKit's order, Firefox the spec's), waits up to 10 s for the
    first motion events, and reports each frame's brightness and contrast.
  - **Lighter tracking for slower phones:** ⚙ → "Lighter tracking…" (on
    by default in Firefox, which has no WebGPU and took 68–120 ms a frame
    on Android) sets a feature grid of a tenth of the frame's short side, at
    most 3 solver iterations, and the estimator on every other frame. Natively
    it costs 2.4x less a frame on iPhone recordings, with 10 resets against
    15 but ADVIO's mean ATE at 3.7 m against 2.6 m.
  - **Overlay matched to the image:** the AR view is drawn at the pose of
    the image that will be on screen when it appears (a display refresh
    ahead, less the camera's measured latency to the page; ⚙ → "Overlay
    timing" fine-tunes it), carried forward from the last estimate by the
    gyroscope and velocity, the camera's lever arm included, and the
    tracked points, which belong to an older frame, are hidden unless asked
    for, then turned onto the frame on screen.
  - **ADVIO results** (iPhone, browser-like 60 Hz IMU): 1.9–4.1 m ATE over
    142–244 m walks, 98–99% of frames tracking (ARKit: 1.2–2.6 m).
  - **Speed:** about 14–17 ms a frame natively on one thread at 360 x 640.
    The page reports whether the phone offers WebGPU and the main
    thread's time to read a frame back.

- **Duplicate-tolerant ratio test.**
  `BinaryLandmarkIndex::with_duplicate_radius` makes the map-wide search
  compare the best landmark with the best one farther than a radius from
  it. Landmarks of one physical point (broken SLAM tracks, sessions of a
  merged map) then no longer reject each other's matches. Off by default.
  Enable it with `basalt_euroc_localize_demo --duplicate-radius-m` or
  `visloc_map_localize_node -p duplicate_radius:=...`. At 10 cm, MH_02
  against MH_01 localizes 83.1% of its images (81.2%), with the worst pose
  7.6 cm off instead of 16.8 cm. V2_02 against V2_01 merged with V2_03 goes
  from 18.4% to 19.3%.

- **Merging maps of several flights.** `basalt_map_merge --base A --add B
  --out AB` registers saved SLAM maps of the same place to the first one
  and writes one map in its frame. `visloc_localization::register_map`
  localizes the added map's keyframes in the base map and keeps the yaw +
  translation that at least 3 of them agree on;
  `visloc_localization::merge_maps` renumbers the added map, moves it and
  fuses its landmarks that coincide with base landmarks. The merged map
  loads like any saved map (`basalt_euroc_localize_demo`,
  `visloc_map_localize_node`); `map.json` lists the sessions and
  `images.csv` gains a `session` column. On EuRoC, MH_04 against MH_01
  merged with MH_05 localizes 55.8% of its images (4.6% against MH_01) and
  the anchored VIO gives 99.7% of frames a pose at 9.5 cm RMSE (73.1% at
  23.0 cm). An added map keeps its own error: MH_05 against MH_01 + MH_04
  gets 99.8% of frames (77.1%) and half the median error, but a worse tail
  where MH_04's part of the map is 25 cm off. Results and limits in
  `docs/basalt_map_localization.md`.

- **Ground robots (ROS 2): `base_link` and wheel odometry.**
  - `visloc_vio_node -p base_frame:=base_link` publishes `odom -> base_link`
    (REP-105) instead of `odom -> imu_link`: odometry, pose, path and `/tf`
    for the base, placed by the `/tf_static` transform between the IMU's
    frame (`body_frame`) and the base. `visloc_ros2::vio::rebase` moves an
    estimate, velocity included, to a frame rigidly attached to the IMU.
  - `visloc_map_localize_node -p extrinsics:=tf` takes the camera's
    position on the robot from `/tf_static` (image `frame_id` to the
    odometry's `child_frame_id`) instead of the map's IMU-from-camera, so
    any gravity-aligned odometry can be anchored, e.g. wheel odometry on a
    level floor. New parameter `camera_frame`.
  - `visloc_ros2::tf_tree::StaticTransforms` composes static transform
    chains like tf2; the nodes subscribe to `/tf_static` transient-local,
    as `tf2_ros` does.
  - On ROS 2 Jazzy, with `static_transform_publisher` providing
    `base_link -> imu0 -> cam0`, EuRoC MH_02 against the MH_01 map: the VIO
    publishing `base_link` gives a map pose for 99.5% of frames at
    2.21 cm RMSE; simulated wheel odometry (98.7 cm ATE on its own) 98.3%
    at 3.45 cm.

- **The real-time ROS 2 map localization is validated on all eight EuRoC
  query sequences** (rosbag2 at 1.0x, Fast DDS): within 1.2 points of the
  offline `basalt_euroc_localize_demo --vio` on the share of frames with a
  map pose (V2_03 never anchors, as offline), with no VIO reset and at most
  27 dropped frames per sequence; table in `ros2/visloc-ros2/README.md`.

- **The ROS 2 nodes are tested against ROS 2 Jazzy** with both
  `rmw_fastrtps_cpp` and `rmw_cyclonedds_cpp`, on one host:
  - `ros2 topic` / `ros2 node` see the nodes and decode their messages.
  - `tf2_echo map imu_link` resolves the REP-105 chain.
  - EuRoC MH_02 played from a rosbag2 into the VIO and map nodes, with
    reliable or best-effort publishers, gets a map pose for 98.7-99.7% of
    frames at 2.06 cm RMSE.
  - New helper scripts, both run with a sourced ROS 2 environment:
    `ros2/visloc-ros2/tools/euroc_to_rosbag2.py` converts a EuRoC sequence
    to a bag, and `tools/bag_to_tum.py` writes a recorded pose topic as a
    TUM trajectory.

- **Real-time map localization (ROS 2).** `visloc_vio_node` and
  `visloc_map_localize_node` now keep up with EuRoC MH_02 replayed over DDS
  at real time against an MH_01 map, with all three processes on 4 cores.
  99.6% of frames get a map pose (was 90.0%), at 2.05 cm RMSE (5.2 cm max).
  The median latency from frame to `map_pose` is 49 ms (was 171 ms) and the
  90th percentile 90 ms (was 271 ms).
  - `visloc_vio_node` runs KLT tracking and the estimator on two threads, as
    upstream Basalt does: `BasaltVioEstimatorAdapter::into_stages` splits
    it into `BasaltVioFrontendStage` / `BasaltVioEstimatorStage`, and
    `VioCore::into_stages` into `VioFrontend` / `VioBackend`. The split
    gives the same estimates as the serial path (tested). New parameter
    `frontend_threads` (default 2).
  - Once anchored, `visloc_map_localize_node` predicts each image's pose
    from the anchor and the odometry. It then matches each map landmark only
    against the keypoints within `projection_radius` (20 px) of its
    predicted projection, and falls back to the map-wide search if that
    fails. Matching and PnP drop from about 50 ms to 4 ms per image, with
    twice the inliers. New parameters `projection_radius`, `prior_wait_ms`
    and `match_threads`; `/diagnostics` reports `matching`.
  - `visloc_localization::BinaryLandmarkIndex` packs a map's binary
    descriptors once. `match_all` and `match_projected` give the same
    correspondences as `CorrespondenceBuilder` /
    `ProjectionCorrespondenceBuilder` with a `HammingMatcher` (tested).
    `LocalizationPipeline::localize_correspondences` runs the pose
    estimation and quality gate on correspondences built elsewhere.

- **`visloc_map_localize_node` (ROS 2): robot-style localization in a saved
  SLAM map.** Run next to `visloc_vio_node`, it localizes every 5th camera
  image against a map from `basalt_euroc_online_slam_demo --save-map`,
  pairs each fix with the VIO odometry of the same stamp, and publishes
  `/tf` `map -> odom` (REP-105), `map_pose` (body pose in the map, position
  covariance 3 cm + 2.5 cm per metre since the last fix) and
  `/diagnostics`. EuRoC MH_02 replayed over DDS against an MH_01 map: a map
  pose for 99.8% of frames at 2.18 cm RMSE (5.6 cm max) at half speed;
  at real time on 4 cores the VIO node drops 9.5% of frames.
  `visloc_vision::matching::HammingMatcher` (binary descriptors as 0/1
  floats) is now shared by the node and the localize example.

- **Save a VI-SLAM map and localize against it.**
  `basalt_euroc_online_slam_demo --save-map DIR` saves the online
  mapper's final map: keyframe poses, ORB landmarks and their
  descriptors, as a COLMAP text model (`DIR/sparse`) with
  `landmark_descriptors.txt`, `images.csv` and `map.json` (IMU-from-camera
  extrinsics). The new `basalt_euroc_localize_demo` runs each image of a
  EuRoC sequence through the mapper's own ORB frontend, matches it to the
  map by Hamming distance, estimates the pose with P3P RANSAC, accepts it
  with at least 30 inliers (`--min-inliers`), and writes IMU poses in the
  map frame to `localization.tum`.
  - The Double Sphere cameras are saved as COLMAP `OPENCV_FISHEYE` with
    `k1..k4` fitted over the field of view (`fit_opencv_fisheye`; at most
    0.0003 px from the Double Sphere projection over the EuRoC cam0
    calibration's 57° half field of view), so the model also opens in
    COLMAP.
  - Library: `NfrMapper::localization_map` (`visloc-basalt`),
    `write_landmark_descriptors_txt` (`visloc-io`),
    `euroc::read_camera_manifest` / `read_camera_image`.
  - EuRoC, one map per environment (MH_01, V1_01, V2_01) queried with the
    other eight sequences: no accepted pose is more than 25 cm off; MH_02
    localizes 81.2% of its images at 1.8 cm RMSE against ground truth,
    and flights the map barely covers far fewer (MH_04: 4.6%). Below 30
    inliers wrong poses appear (at 20, up to 3.6 m off on V2_02).
    `docs/basalt_map_localization.md`, scored by the new
    `scripts/evaluate_map_localization.py` (`--per-frame --min-inliers`
    scores any threshold from one run).
  - `--vio --calibration FILE --config FILE` runs the Basalt
    stereo-inertial VIO on the query sequence and anchors it to the map
    with the fixes of every `--localize-every`-th image
    (`visloc_localization::MapAnchor`: a gated, gravity-aligned yaw +
    translation offset). Every frame gets a map pose in
    `trajectory_map.tum`; `trajectory_map.csv` adds the time and distance
    since the last fix. MH_02: 99.8% of frames at 2.0 cm RMSE localizing
    every 5th image, at 0.83× real time on 4 cores; on all seven anchored
    queries no pose within 2 m and 5 s of a fix is more than 12.6 cm off,
    while poses carried tens of metres between rare fixes reach decimetres.

### Changed

- **visloc-basalt builds and runs on wasm32-unknown-unknown.** Natively
  everything is identical, and the crate's tests pass unchanged.
  - The libstdc++ bucket arithmetic is width independent.
  - The diagnostic probes skip the environment and process id where there
    are none.
  - `mul_add` goes through a `fused_mul_add` that inlines on wasm. The
    software `fmaf` it replaced took 40% of a frame.
  - `BasaltVioEstimator::landmark_positions_world` exposes the window's
    landmarks for display.
  - `BasaltVioEstimatorAdapter::process_tracks_no_trace` runs the estimator
    on tracks from another frontend, and `DirectKltStream::cam0_tracks`
    exposes the tracks' patch transforms.
  - `config.vio_imu_noise_per_step` (opt-in, default off): the IMU noise is
    discretized by each preintegration step's length. Basalt injects the
    per-sample noise into every step, which undercounts the short steps at
    each frame interval's edges: a third of the covariance with one IMU
    sample per frame (60 Hz IMU at 60 fps), whose overconfident IMU took
    ADVIO at 60 fps from 2.65 m to 7.0 m. With it, 4.3 m and 2 resets
    instead of 13; at 30 Hz neutral. A test checks the covariance no
    longer depends on where an interval's samples fall.

- `basalt_euroc_localize_demo` matches map-wide through
  `BinaryLandmarkIndex`: the same matches as before (byte-identical
  output on MH_02), at a median 67 ms per image instead of 86 ms.

- **Basalt KLT frontend about 38% faster, bit-identical.** The five bilinear
  samples of an interpolated gradient now share one neighbourhood read. FAST
  replenishment converts each grid cell to u8 once and rejects pixels with a
  cardinal-pixel test before the full ring test (2.6 to 0.6 ms per frame).
  Each frame's backward searches already sample the source patches the
  next frame's forward searches need, so those patches are cached by camera
  and exact position and reused. MH_02 on 4 cores: 12.3 to 7.6 ms per
  frame, with a byte-identical trajectory.
- **Basalt VIO estimator about 26% faster, bit-identical.** The LM
  iteration's linearization and its model cost decrease (on the retained
  QR payload path) evaluate landmarks in parallel and combine them in the
  original order. MH_02 on 4 cores: 30.0 to 22.1 ms per frame (115 to 76 ms
  at worst), with a byte-identical trajectory.
- `visloc_vio_node` runs 3 LM iterations per frame by default (new
  parameter `max_iterations`, overriding `config.vio_max_iterations`, which
  the EuRoC config sets to 5). On eight EuRoC sequences the VIO's ATE moved
  by at most 0.13 cm, and the estimator got 15-30% faster. `0` keeps the
  config's value.
- `visloc_vio_node`: `frame_queue` defaults to 1 (was 4), and the frontend
  hands frames to the estimator without buffering. When the estimator falls
  behind, the node drops older frames before tracking instead of building
  a backlog. `visloc-ros2` builds `visloc-basalt` with
  `basalt-lm-workspace-reuse`.
- `MapLocalizer` (`visloc-ros2`): `localize(image, prior)`; the new
  `extract` / `features` produce `ImageFeatures` for
  `localize_features(&features, prior)`; `Fix::matching` names the search
  used.

## 0.3.1 - 2026-10-25

Patch release: `--help` on the release binaries, fewer cameras left far off
by `gsplat_photos`' intrinsics refinement, and prebuilt binaries attached
to every release.

### Fixed

- **`--help` on the release binaries' command-line tools.** `visloc_mvs`
  (the `mvs` example), `gsplat_photos` and `colmap_incremental_mapper`
  print their options with `-h` / `--help`. `visloc_mvs` panicked on it
  ("unknown argument --help"), `gsplat_photos` stopped with the same error
  and the mapper panicked on the missing `--manifest`. A missing or unknown
  argument now prints the message (with the usage for `visloc_mvs` and the
  mapper) and exits with status 2 instead of a panic and a backtrace.

### Changed

- **`gsplat_photos` damps its intrinsics refinement from the start.** The
  joint pose + intrinsics BA after the mapper starts Levenberg–Marquardt at
  λ = 1e4 instead of 1e-4. The damping is absolute and a photo set's
  Hessian diagonals are 1e4–1e8, so 1e-4 was no damping: its first steps
  swung cameras by up to a radian and were rejected until λ reached that
  scale.
  - **Three 500-image SmallCity pass-1 subsets**, camera centres against
    the release's COLMAP model (Sim(3) refit on errors under three times
    the median), median / p90 / max:

    | Subset | before | after |
    |---|---|---|
    | 0–499 | 13.6 / 22.6 / 282 cm (4 cameras over 1 m) | 12.1 / 17.3 / 23 cm (none) |
    | 500–999 | 23.8 / 29.2 / 66 cm | 24.9 / 30.8 / 69 cm |
    | 1000–1499 | 8.6 / 16.1 / 22 cm | 7.9 / 15.2 / 21 cm |

  - **ETH3D DSLR, all 13 training scenes** (one camera per scene, 1,600 px,
    the focal started 5% off, ground-truth poses): the same images register
    and the scores barely move. Means over the scenes before → after:
    relative rotation 0.084° → 0.085°, camera-centre median 0.76 → 0.79 cm,
    p90 1.38 → 1.42 cm, maximum 3.99 → 3.86 cm. meadow is the largest
    change either way (median 4.0 → 4.5 cm), then facade (1.71 → 1.61 cm),
    office (maximum 6.4 → 4.3 cm) and playground (0.26 → 0.30 cm); seven
    scenes are unchanged.
  - **Why the first subset needed it:** the mapper starts from images 66
    and 324, a revisit 3.5 m apart, because consecutive photos are too
    close for COLMAP's 16° initial-pair angle. Its model leaves image 66
    32 m off and images 319–352 bent by up to 29 m; the refinement pulls
    them back, and with the damping it no longer stops halfway. Starting
    from nearby photos instead (an 8° initial-pair angle, or a pair at most
    30 photos apart) avoids the bend but leaves a 18 cm median on that
    subset: the revisit pair also closes the loop, so neither is used.

### Added

- **Prebuilt binaries on every release** (`.github/workflows/release.yml`).
  A `v*` tag push creates the GitHub release from that version's CHANGELOG
  section and attaches `gsplat_photos`, `visloc_mvs` (the `mvs` example),
  `colmap_incremental_mapper` and `visloc_web`, built with their GPU
  features, for Linux x86_64 (built on Ubuntu 22.04, glibc 2.35), Windows
  x86_64 and macOS on Apple silicon, as `.tar.gz` / `.zip` with SHA-256
  sums. A manual run (`workflow_dispatch`, input `tag`) builds an existing
  tag, e.g. `v0.3.0`.

## 0.3.0 - 2026-10-24

### Highlights

- **Multi-view stereo: `visloc-mvs` (new crate).** Dense point clouds and
  meshes from a COLMAP model, on the CPU or on the GPU through wgpu. On
  the ETH3D high-res training set (13 scenes at 3,200 px, one NVIDIA L4)
  F1 at 2 cm is 83.7%, against the 83.42% ACMMP reports; against
  ACMMP's released code on the same L4, 83.4% against 83.2% over six
  scenes, in 27.5 minutes against 110. From the photos end to end,
  76.3% against 57.4% for COLMAP 4.2.1 CUDA over the 10 scenes both
  finished, in 58 minutes against 372.
- **City-scale photos to splats.** The README hero is Hierarchical 3DGS
  SmallCity: 5,822 photos, 5,583 registered, 4.7M gaussians and a
  5.9M-triangle mesh from one `gsplat_photos` command in 2.5 h on a Colab
  A100. Along the way: VLFeat-compatible SIFT on the GPU (SmallCity's
  photos in 43.8 s on an RTX 3090), optional GPU bundle adjustment, a
  converged final BA, completion of the largest model, and a COLMAP-port
  mapper more than twice as fast with about half the memory (SmallCity
  peak 9.0 GiB instead of 18.9 GiB).
- **SfM ahead of COLMAP 4.2.1 on ETH3D.** On COLMAP's matches of the 13
  DSLR training scenes the mapper scores a mean AUC@1°/3°/5° of
  53.69 / 79.94 / 86.90 (452 of 454 images), against 51.34 / 76.25 /
  82.88 for COLMAP's incremental mapper and 48.65 / 70.41 / 76.36 for
  its global mapper.
- **360° panoramas.** `gsplat_photos --equirect` (the six cube faces of a
  panorama as one rig) and `--equirect-native` (COLMAP 4.1's
  `EQUIRECTANGULAR` camera); `gsplat_train` trains on equirectangular
  COLMAP models.
- **New ways in.** `visloc_web` runs the photo pipeline from a browser;
  Python bindings (`bindings/python`, package `visloc`); ROS 2 nodes
  (`ros2/visloc-ros2`) for the VIO and localization.
- **Sensors and fusion.** Multi-camera Basalt VIO for divergent rigs
  (opt-in), monocular-inertial VIO (experimental), and a joint GNSS +
  visual-odometry pose graph.
- **Lens models in bundle adjustment.** Pose/structure BA and rig BA use
  each camera's full distortion model; tangential distortion can be
  self-calibrated (opt-in).

### Full change log

The itemized 0.3.0 change log (every Added / Changed / Fixed entry from
2026-10-08 to 2026-10-24, about 1,400 lines) is archived in
[`docs/archive/changelog_0.3.0_full.md`](docs/archive/changelog_0.3.0_full.md).

## 0.2.1 - 2026-10-08

Patch release: camera-distortion correctness fixes. Each fix below corrected
results that were silently wrong, not just errors that were reported.

### Fixed

- **COLMAP camera export with self-calibrated distortion.** A `Pinhole`
  camera carrying the refined radial `[k1, k2]` tail
  (`--refine-distortion`) was written as `PINHOLE` with 6 params, which is
  not valid COLMAP (`PINHOLE` has 4); COLMAP rejects it and other readers
  silently drop the distortion. Every COLMAP writer (text and binary) now
  exports it as `OPENCV` with `p1 = p2 = 0` (the identical projection), and
  fallible writers reject any camera whose parameter count does not match its
  COLMAP model. New public helper `visloc_io::colmap::colmap_camera_record`.
- **OPENCV tangential distortion.** `Camera::project` / `normalize_pixel` /
  `unit_ray_from_pixel` ignored `p1, p2` of an `OpenCv` camera, so a COLMAP
  `OPENCV` model with non-zero tangential terms was silently treated as
  radial-only. Such cameras now use the full Brown-Conrady model (shared with
  `FULL_OPENCV`, k3..k6 = 0); results with `p1 = p2 = 0` are bit-identical to
  before. New `Camera::tangential_distortion`.
- **Pose/structure BA ignored self-calibrated radial distortion.** With
  `refine_intrinsics = false` (e.g. the SfM polish / warm-start passes after
  `--refine-distortion`), `optimize_weighted` built its residuals and
  Jacobians with an undistorted pinhole while the cost it accepts steps
  against used the distorted `Camera::project`, so LM optimised a different
  model than it measured and could stall. The monocular residual and Jacobian
  (serial, parallel, and dense-QR assembly, plus the GNC residual vector) now
  apply `1 + k1·r² + k2·r⁴` with an analytic Jacobian; distortion-free cameras
  keep the historical closed form bit-for-bit.

### Added

- **`BaConfig::shared_focal` / `--shared-focal`** (unordered and sequential
  SfM demos): constrain intrinsics self-calibration to `fx == fy`. Previously
  `fx` and `fy` were always refined independently, so weakly observable
  motion (near-pure forward translation) could settle on a spurious aspect
  ratio. Default off; the joint solve is unchanged when it is not set.

## 0.2.0 - 2026-09-15

### Highlights

- **Real-data SfM at scale.** All ten ETH3D low-resolution many-view scenes:
  **9,996/10,008 cameras registered (99.88%)**, with no mapper run above
  3.32 GiB. Same-input ETH3D Electro 1,200 runs **3.46× faster than COLMAP**
  with **25.2% lower** camera-centre RMSE, and the 38-image courtyard control
  reaches **0.5379 cm vs COLMAP's 1.6166 cm**.
- **Visual-Inertial SLAM (faithful Basalt Rust port).** A tightly-coupled
  stereo-inertial VIO estimator plus offline mapper that matches native
  Basalt's ATE within **0.1%** on all 11 EuRoC sequences and, with EuRoC's
  official calibration, beats measured ORB-SLAM3 on **8 of 11** sequences.
- **COLMAP-compatible unordered SfM** with VLAD retrieval, essential-matrix
  verification, incremental mapping, connected-component models, and COLMAP
  text export for downstream 3DGS / MVS.
- **Online stereo SLAM** with loop closure, SE(3)/Sim(3) pose-graph
  optimization, GNC outlier rejection, and bundle adjustment, benchmarked on
  KITTI, EuRoC, and TUM RGB-D.
- **Opt-in deep frontend** (SuperPoint / LightGlue via in-Rust ONNX Runtime,
  CUDA-accelerated), plus public-data map-reuse localization and GNSS-prior
  tracking demos.

### Full change log

The itemized 0.2.0 change log (every Added / Changed / Fixed entry from
2026-05-07 to 2026-09-15, about 4,800 lines) is archived in
[`docs/archive/changelog_0.2.0_full.md`](docs/archive/changelog_0.2.0_full.md).

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
