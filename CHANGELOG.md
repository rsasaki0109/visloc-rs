# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- **CUDA 3DGS rasterizer and trainer (`visloc-gsplat-cuda`, experimental,
  opt-in).** This is a CUDA backend for 3DGS rendering and training, for
  machines where wgpu cannot reach the GPU (WSL2 exposes CUDA but no NVIDIA
  Vulkan driver). It also serves as a reference for CUDA-level
  optimizations.
  - The packaging follows the DPVO kernels: `native/gsplat_cuda/gsplat_cuda.cu`
    is built by `scripts/build_gsplat_cuda.sh` and loaded at runtime. Cargo
    never compiles CUDA, and the crate is `publish = false`.
  - The rasterizer's maths is the wgpu renderer's. Against the CPU f64
    oracle, every parameter gradient is within 4.2e-6 of its largest value;
    the tests use the wgpu tests' tolerances, 3e-3.
  - The training step is the wgpu trainer's: L1 + D-SSIM, Adam with brush's
    aux losses, refine statistics and mean noise. The scene, its moments and
    the images stay on the device.
  - `CudaTrainer` (example `gsplat_train_cuda`) runs the wgpu trainer's
    brush-strategy loop.
  - On south-building (1024x768, 112 train / 16 eval views) it trains 30k
    steps in 691 s on an 8 GB laptop GPU, ending at 1.41M gaussians and
    23.22 dB held-out PSNR. The README's wgpu run reached 22.77 dB with a
    different SfM and the depth-normal loss.
  - Rasterizer optimizations cut a training step from 30.9 to 21 GPU ms;
    before them the run took 978 s.
    - Sub-tile masks: 8x4 sub-tiles per warp, 8 bits in the sorted value.
    - `project_bwd` skips gaussians that blended into nothing.
    - Gradients go to global memory with native float reductions; on this
      GPU the shared-memory float atomic was a CAS loop.
    - Three dense runs spread over 22.71–23.22 dB.
  - `--sparse-sh-adam` (opt-in) updates SH only for blended gaussians,
    halving the optimizer (18 ms per step). In training it saved 5% at
    22.87 dB.
  - Tried and not adopted, each under 1% per step: the two-stage sort, a
    vectorized Adam and dropping the gradient memset.
    `gsplat_cuda_bench --data` now also reports each step's wall time.
  - Meshes from CUDA:
    - The median depth comes from the splat centre, or with
      `--planar-depth` from where the ray meets the splat's plane.
    - `gsplat_mesh_cuda` meshes a trained splat; `gsplat_train_cuda --mesh`
      meshes after training.
    - The TSDF fusion is shared with wgpu
      (`visloc_gsplat_train::mesh::fuse_views`, behind the new `mesh`
      feature).
  - **ETH3D geometry.** Three high-res training scenes, on their
    ground-truth poses at 1600x1067, score a mean F1 of 33.1% at 2 cm
    (terrains 34.1%, playground 43.9%, office 21.4%). visloc-mvs's clouds
    score 80.0% on the same scenes. The mesh options needed are 1 cm
    voxels, no support filter, depths up to 20 m, min weight 1 and planar
    depth; `gsplat_photos`' defaults score 20.8%. See
    `scripts/eth3d_3dgs_mesh.sh`.
- **Depth priors for 3DGS (opt-in).**
  - `visloc-mvs --save-fused-depth` writes each view's depths that fusion's
    source views agree with.
  - `gsplat_train_cuda --depth-prior <dir> [--depth-weight 0.1]` trains the
    splat's expected depth against them, with a relative L1. A map of another
    size is resampled by nearest neighbour.
  - The expected-depth channel's gradients match an exact colour-channel
    identity to 1.8e-7.
  - On the three ETH3D scenes, weight 0.1 raises the 3DGS mesh's mean F1
    at 2 cm from 33.1% to 35.0% (office 21.4% → 25.5%); weights 0.5 and 1.0
    give 34.0% and 35.3%. Accuracy rises by up to 32 points, but
    completeness does not, and the mesh stays far below the MVS cloud that
    supervised it (67.6%).
  - CUDA ABI 5.
- **Depth-normal loss for the CUDA trainer (opt-in).**
  - `gsplat_train_cuda --normal-weight w [--normal-start 7000]` runs the wgpu
    trainer's 2DGS / PGSR-style loss, which `CudaTrainer` refused before.
  - The step blends a normal channel next to the expected depth and
    back-propagates it to the rotations.
  - `CudaGsplat::render_geometry`, `normal_loss` and
    `backward_with_geometry` expose the channels and their gradients.
  - The normal channel's gradients match an exact colour-channel identity
    to 9.0e-7. The loss's gradients match central differences of an f64 CPU
    copy to 2.5e-6.
  - It does not improve the ETH3D meshes. At weight 0.005 the mean F1 at
    2 cm is 32.0% against 33.1%; with the depth prior 0.1 it is 34.0%
    against 35.0% for the prior alone. 0.05 is far too strong (terrains
    22.9%).
  - CUDA ABI 6.
- **TSDF on the GPU for CUDA meshes (`gsplat_mesh_cuda --gpu-tsdf`).**
  - `mesh::extract_mesh_gpu` (`CudaGsplat::tsdf_surface`) fuses the median
    depth and extracts surface nets on the device, with `Tsdf`'s maths,
    options and filters.
  - Every block is allocated before any view is fused. The blocks are then
    fused in slabs along x, so memory stays bounded at any voxel size.
  - On ETH3D at 1 cm it matches the CPU's F1 to 0.2 points in 4–5x less time
    and 1/6 of the memory (terrains: 5.4 s and 1.6 GB against 24.0 s and
    10.3 GB). playground, which ran out of memory on the CPU, meshes in
    12.0 s and 4.2 GB.
  - One view gives the CPU's mesh, and any number of slabs gives the same
    surface bit for bit.
  - `Mesh::remove_small_components` counts triangles per component in a
    vector instead of a hash map.
  - CUDA ABI 7.
- **Descriptor matching on the GPU for photos-to-dataset
  (`gsplat_photos_cuda`, `visloc-gsplat-cuda` feature `photos`).**
  - `PhotoSfmConfig::matcher` lets a `PairMatcher` take
    `build_photo_dataset`'s candidate pairs. The CPU matches if it fails.
  - `matching::CudaPairMatcher` computes each pair's top-2 scores in both
    directions with exact 8-bit dot products (`CudaGsplat::match_top2`). It
    then applies the CPU matcher's filter, so the matches are
    `BruteForceMatcher::match_descriptors_cross_checked`'s, bit for bit.
  - On south-building (128 images, 10.5k SIFT each), matching and
    verification take 18.4 s against 780.7 s for the CPU matcher, with the
    same 2461 verified pairs. The whole SfM takes 148.4 s against 945.0 s,
    with the same model: 128/128 registered, 101,450 points, focal 850.2 px.
  - Verification alone takes 5.3 s once the CPU matcher no longer holds
    every core.
  - `CudaGsplat` is `Send`. CUDA ABI 8.
- **Meshes from the expected depth (`gsplat_mesh_cuda --expected-depth`,
  `visloc_gsplat_cuda::mesh::MeshDepth`).**
  - The depth prior and the depth-normal loss train the expected depth, but
    meshes were fused from the median depth.
  - Fusing the expected depth (where the opacity exceeds 0.5) from splats
    trained with a strong prior lifts the ETH3D mesh F1 at 2 cm from 33.3%
    to 63.6% at prior weight 30: terrains 78.1%, playground 60.4%, office
    52.3%. The prior's own MVS clouds score 67.6%.
  - Completeness rises with it, which no loss achieved in the median mesh
    (terrains 26.9% → 70.2%).
  - The cost is the image fit: the final training L1 about doubles at 30.
  - `extract_mesh` / `extract_mesh_gpu` take a `MeshDepth` instead of
    `planar: bool`. CUDA ABI 9.
- **`colmap_resize`** (example of `visloc-gsplat-train`) scales a COLMAP text
  model of pinhole images and the images themselves to a maximum or an
  exact size (`--size` makes cameras of slightly different sizes, as in some
  ETH3D scenes, share one).
- **`scripts/build_eth3d_evaluator.sh`** builds the official ETH3D
  multi-view evaluator on current toolchains (CMake 4, Boost 1.90, PCL 1.15).
  - See [docs/gsplat_cuda.md](docs/gsplat_cuda.md).
- **`visloc_gsplat_train::config`.** `TrainConfig` moved here from the
  wgpu-only `trainer` module, together with the image packing, brush's
  bounds and `population_to_scene`, so that a trainer without wgpu can use
  them. `trainer::TrainConfig` is still exported.
- **`--detector-quality Q`: a lower ORB detector threshold (opt-in).** The
  mapping detector keeps corners stronger than 1% of the image's strongest
  (upstream's quality level); in OpenLORIS-Scene's corridor a few bright
  corners push the walls' texture below that.
  `basalt_euroc_online_slam_demo --detector-quality Q` builds a map with
  another level (`map.json`: `feature_quality_level`), and the localize demo
  and `visloc_map_localize_node` detect the same way
  (`visloc_basalt::mapper::features::detect_keypoints_mapping_with`).
  With the blur at 0.003: corridor1-1 localizes 360 times in corridor1-5's
  map instead of 75, its VIO + map 13 cm off at the median instead of 62 cm;
  office1-7 186 times instead of 52, at 2.66 cm instead of 6.61 cm. About
  20 ms more per image. On EuRoC's bright, textured images, alone, it adds
  fixes in the Vicon rooms but not accuracy (six of the eight queries
  0.04–11 cm worse), so it stays off by default. `map_files::Frontend`
  gathers the examples' equalization, blur and quality level.

- **Blur before ORB (opt-in).** ORB's descriptor compares single pixels,
  and on OpenLORIS-Scene's grainy T265 images noise flips its bits: two
  near-identical views from two corridor sessions shared 22 consistent
  matches. `basalt_euroc_online_slam_demo --smooth-features SIGMA` blurs
  each keyframe image before ORB extraction
  (`visloc_basalt::smoothing::GaussianSmoothing`), `map.json` records it
  (`feature_smoothing_sigma`), and `basalt_euroc_localize_demo` and
  `visloc_map_localize_node` blur their images the same way;
  `basalt_map_merge` refuses maps built differently. At sigma 1.5 the pairs
  share 38 matches, corridor1-5's own images localize 52% of the time
  instead of 33.5%, and corridor1-1 localizes 75 times in its map instead
  of 57 (VIO + map 1.55 → 1.35 m against the map around it). Elsewhere more
  images localize at about the same error: office1-7 52 instead of 5 (6.86
  → 6.61 cm), V2_03 36 instead of 5 (29.3 → 28.8 cm), MH_02 1.97 → 2.03 cm.
  About 10 ms more per image. Off by default.

- **Wheel odometry with its heading from a gyro.** Wheels slip in turns in
  place: on OpenLORIS-Scene's corridor (corridor1-1, 272 s) the wheel
  odometry's heading drifts 44.5° from the ground truth.
  `visloc_localization::GyroWheelOdometry` keeps the wheels' travel and
  turns it by the heading a gyro integrates about the vertical, which the
  accelerometer gives (any IMU, mounted any way); at a standstill the
  heading holds and the gyro's rate is its bias. The heading then stays
  within 5.6°, and 60 s after a known pose the odometry is at most 1.4 m
  off instead of 15.9 m. Wheel odometry + map on corridor1-1 localizes 58
  times instead of 12 and its RMSE falls from 3.5 to 0.95 m; office1-6
  and office1-7 go from 5.11 to 4.79 cm and from 11.69 to 10.60 cm.
  `visloc_wheel_calibration_node` uses it with an `imu` topic
  (`gyro_heading`, on by default), `launch/map_localization.launch.py`
  passes `imu_topic` to it, and `wheel_calibration_replay` takes `--imu`.

- **Extra descriptors per landmark (opt-in).**
  `basalt_euroc_online_slam_demo --extra-descriptors` keeps, besides each
  landmark's medoid descriptor, up to three of its views' descriptors,
  farthest first and at least 24 bits apart
  (`landmark_descriptors_extra.txt`).
  `visloc_localization::BinaryLandmarkIndex::with_extra_descriptors`
  matches a keypoint against the nearest of a landmark's descriptors, with
  the ratio test against the nearest other landmark;
  `basalt_euroc_localize_demo --extra-descriptors N` and
  `visloc_map_localize_node`'s `extra_descriptors` use them. On the eight
  EuRoC queries they find more fixes (MH_04 56 instead of 48) but less
  accurate ones: six queries get 0.02–1.2 cm worse, V2_03 better (29.3 to
  23.6 cm), and the map-wide search up to three times slower. Off by
  default.
- **`evaluate_map_localization.py --mode local`.** Like `fixed`, but the
  map->ground truth transform is fitted for each scored pose on the stretch
  of the map's own trajectory within `--align-radius-m` (5) of it and, by
  the ground truth, facing within 90° of it, turning about the vertical
  only. On a map whose SLAM drifted over a long loop (corridor1-5: 1.33 m
  RMSE, no loop closed) it scores a localization against the map around
  it, on the pass whose landmarks it saw.
- **`wheel_calibration_replay`.** `ros2/visloc-ros2/examples` runs
  `visloc_wheel_calibration_node`'s calibration on recorded wheel odometry,
  VIO and map localizations (TUM files), without ROS, and writes the
  calibrated wheel odometry for `basalt_euroc_localize_demo --odometry`.
- **`basalt_euroc_localize_demo --stale-projection-min-inliers N`.** A
  prediction past the fresh limits is tried too, and its projection search
  counts only with N inliers. With 60, its 14 fixes over the eight EuRoC
  queries are at most 10.4 cm off; MH_04 goes from 22.0 to 21.2 cm and
  V1_03 from 4.80 to 4.24 cm, but V2_03, with 5 fixes in all, from 29.3 to
  39.1 cm. Off by default.

- **`gsplat_photos --imu`: metric models from the camera's IMU.** For
  photos that are frames of a video (file names ending in the frame
  number), the model is scaled to metres right after SfM, so sparse/0, the
  splats and the mesh come out in metres. `visloc_slam::estimate_imu_scale`
  is the Rust port of the per-run visual-inertial alignment validated on the
  50 OverMaps360 walks (docs/insta360_imu_scale.md): it searches the frame
  timing from the gyro and fits scale, gravity, accelerometer bias and a
  rolling-shutter term per run. It matches the Python reference to four
  decimals in seconds. The IMU comes from an Insta360 `.insv`
  (`visloc_io::insv` reads its trailer: IMU and video frame timestamps) or a
  CSV file (`visloc_io::imu_csv`: SI units, EuRoC, OverMaps360). On 136
  frames of an OverMaps360 walk the metric model agrees with the dataset's
  IMU scale within 0.01%. The `imu_metric_scale` example runs the estimator
  on any COLMAP model.
- **`gsplat_photos --insv`: Insta360 video straight to a metric model.**
  The input is the video's raw dual-fisheye frames, both lenses side by
  side (one ffmpeg command), with no Insta360 Studio export.

  - `visloc_io::insv` reads each lens's factory calibration (unified model)
    from the `.insv` metadata. The video's lens image is the square spanning
    the 200° image circle around the calibrated centre
    (docs/insta360_lens.md).
  - `visloc_gsplat_train::dual_fisheye` cuts each lens into five 90° faces
    within 90° of its axis. A frame's ten faces are solved as one rig, with
    the back lens's pose refined by the mapper. The video's IMU scales the
    result to metres.
  - The IMU frame search starts from the `.insv`'s own frame period.
  - Without `--images`, `gsplat_photos --insv VID.insv --out run` cuts the
    frames itself with ffmpeg (`--frame-step`, default 10;
    `dual_fisheye::extract_dual_fisheye_frames`).
  - On 200 frames each of five OverMaps360 walks (CPU, 13–30 min), SfM
    registered 99.4–100% of the faces.
    - Outdoors, the metric scale is within 4% of the dataset's IMU scale
      over the same frames.
    - Indoors (a mall), it is off by 12%, with a weak fit (|g| 9.57).
    - These runs used 640 px faces. With 1024 px faces the mall's fit is
      sound (|g| 9.80) and lands +1.2% over the same frames, while the
      outdoor walks move by at most 0.8 points. Keep `--max-size` at 1024
      or more.
    - The `.insv` image-circle string is read after `p2_` or `n2_`.

### Fixed

- **Wheel calibration: the offset wandered on long runs.** A robot's
  wheels slip in turns in place, and on OpenLORIS-Scene's corridor
  (corridor1-1: 284 s, 232 m) their heading drifted 38° off the VIO's in
  two minutes. Fitted with one heading over the 120 s window, the offset
  absorbed that drift and wandered over −117…29 cm forward and −64…71 cm
  left, and the map's scale from localizations 140 s apart came out 0.485.
  `visloc_localization::WheelCalibration` now gives each 5 s of VIO pairs
  (`segment_s`) and each 30 s of map localizations (`fix_segment_s`) a
  heading and a position of their own, sharing the offset and the scale:
  the offset stays at −4.0…2.8 / 6.7…12.1 cm (the office sessions':
  1–3 / 6–8 cm), the fit residual falls from 62 to 2.3 cm, and the scale
  is the map's 1.0315. `visloc_wheel_calibration_node` takes both as
  parameters. The short office sessions' wheel odometry + map stays within
  0.01 cm.
- **`openloris_to_euroc.py` on large .7z files.** py7zr fed its LZMA decoder
  more input on every call while the decoder still held some, so on the
  T265's incompressible PNGs its memory grew with the archive (to 9 GB, and
  killed, on corridor1-1's 13.9 GB). The converter now feeds it only when
  it needs input: 54 MB instead of 417 MB on a 547 MB test archive.
- **ROS 2 nodes: a missed `/tf_static` stalled them for good.** A RustDDS
  reader sometimes never receives a transient-local writer's samples: a
  probe with the nodes' `/tf_static` reader
  (`ros2/visloc-ros2/examples/tf_static_probe.rs`), started 200 times next
  to freshly started `static_transform_publisher`s, had nothing after 30 s
  in 4 cases, and in one OpenLORIS-Scene replay `visloc_vio_node`
  published no odometry. A second reader in the same node receives them.
  `visloc_vio_node` (`base_frame`) and `visloc_map_localize_node`
  (`extrinsics tf`) now subscribe again while a transform they need is
  missing (`dds::StaticTfListener`): 5 s after the first subscription,
  then with doubling waits, at most six times. With that, 300 of 300
  probes had both transforms.
- **`gsplat_photos --equirect-native` crashed writing its model.** Since
  0.3.0 the sparse model lists each image's observations. With
  `--equirect-native` the tracks observe the panoramas while the model
  holds their cube faces, so the panorama observations were indexed as
  faces, and every run panicked at the end. Each observation now goes to
  the face nearest its ray, at the pixel where the ray crosses it. On the
  synthetic room the faces' observations reproject at a median 0.09 px, as
  with `--equirect`. The two end-to-end tests that cover this (run with
  `--ignored`) misread the observation lines and failed too; they now also
  check the observations' reprojection.
- **Saving a map from a lens wider than the Double Sphere model failed.**
  The map export fits COLMAP's `OPENCV_FISHEYE` to each Double Sphere
  camera. It gave up when an image corner did not unproject, which happens
  with a RealSense T265's fisheyes. The corner is now moved toward the
  principal point until it unprojects, so the fit covers the model's
  domain: 0.03 px over 118° for the T265. EuRoC maps are unchanged.

- **visloc-basalt: `vio_min_triangulation_dist` was ignored.** The VIO's
  landmark baseline gate was hard-coded at 5 cm; it now reads the
  configuration value, as upstream does. The pinned EuRoC file says 5 cm,
  so its results are unchanged.

- **visloc-basalt: `vio_obs_std_dev` and `vio_obs_huber_thresh` were
  ignored.** The visual residuals were weighted with a hard-coded 0.5 px
  standard deviation and 1 px Huber threshold; they now read the
  configuration, as upstream does. The pinned EuRoC file has the same
  values, so its results are unchanged.

- **`basalt_map_merge` wrote merged maps without point tracks.** The
  COLMAP text reader fills only the keyframes' observations, so the
  merged `sparse/points3D.txt` had an empty `TRACK[]` for every point.
  Localization never reads the tracks, but COLMAP and other readers do.
  The saved-map loader of the Basalt examples now rebuilds each
  landmark's track from the keyframes, and the save/load round trip test
  checks the tracks. Maps merged before this fix should be merged again.

- **ROS 2 nodes: relative topics were invisible to ROS 2.** `ros2-client`
  0.11 resolves a relative topic name in the root namespace to the DDS
  topic `rt//odom` instead of `rt/odom`. Only other `ros2-client` nodes
  saw those topics, so a ROS 2 subscriber to `/odom`, `/map_pose`,
  `/pose` or `/path`, or a camera driver publishing `/left/image_raw`,
  never connected; only the absolute `/tf` and `/diagnostics` did. The
  nodes now resolve names under their namespace themselves (`odom` is
  `/odom`, `/robot1/odom` with `-r __ns:=/robot1`).

### Added

- **`visloc_wheel_calibration_node`: wheel odometry calibrated while
  driving.** It fits where a robot's wheel odometry is measured relative to
  `base_link`, `base_link`'s lean and the map's scale
  (`visloc_localization::WheelCalibration`, the Rust port of
  `scripts/calibrate_wheel_offset.py`), and publishes the wheel odometry as
  `base_link`'s pose and `odom -> base_link`. The offset comes from
  `visloc_vio_node`'s odometry once the robot has turned 180° and travelled
  1 m; the lean and the scale from `visloc_map_localize_node`'s
  localizations (the new `map_fix` topic), which carry the map's own gravity
  and scale, and without a VIO the offset too, held near the saved one
  where their turns do not show it. A VIO session's scale is reported but
  not used: on OpenLORIS-Scene it was up to 3.6% off the map's. The node
  refits every 5 s over the last 120 s, continuous wherever a new estimate
  takes over, and a saved `calibration_file` applies from the start. On
  OpenLORIS-Scene through ROS 2, the map's localizations alone take
  office1-6's wheel odometry + map from 8.01 to 5.49 cm; office1-3's
  offline calibration, saved, gives 4.92 cm and office1-7 4.66 cm (12.10 as
  is), where its 24 localizations show only the lean. In the demo, the
  map's lean and scale with office1-3's offset take office1-7 from 5.07 to
  3.64 cm. `launch/map_localization.launch.py` starts it with
  `wheel_odom_topic` (and `wheel_calibration_file` to keep the calibration
  between launches).
- **The ROS 2 nodes on a real robot's wheel odometry.**
  `tools/euroc_to_rosbag2.py --odometry FILE.tum` adds a robot's wheel
  odometry to the bag (`nav_msgs/Odometry` on `/wheel/odom`, `odom ->
  base_link`). On OpenLORIS-Scene, played at real time into
  `visloc_map_localize_node` (`extrinsics:=tf`, `weak_gate_m:=0.05`)
  against the scale-corrected maps of other sessions, it gives 2.47 and 3.25
  cm RMSE with the wheel odometry and 1.51 and 2.22 cm with
  `visloc_vio_node`, within 0.4 cm of the offline demo.
- **`calibrate_wheel_offset.py` levels the body frame.** Planar wheel
  odometry keeps its body level, so a tilt in the extrinsic moves the IMU
  sideways, by 1.6 cm per degree with OpenLORIS-Scene's camera 0.9 m above
  `base_link`. The script now turns the body frame by the smallest rotation
  that makes the VIO's mean up its z axis (`--keep-tilt` skips it): 1.9° on
  OpenLORIS-Scene. Wheel odometry + map falls from 4.4–6.2 to 3.5–5.1 cm
  RMSE on the maps as shipped (five of six runs better), and the wheel
  odometry alone to 0.5–4.3 cm. It also normalizes the quaternion it reads.
- **`scripts/scale_stereo_baseline.py`: the stereo VIO's scale from the
  wheel odometry.** A stereo VIO takes its scale mostly from its camera
  baseline. On OpenLORIS-Scene the T265 calibration shipped with the
  dataset makes the SLAM 1–3.5% long, which is most of the maps' error.
  `calibrate_wheel_offset.py` prints the SLAM's scale against the wheel
  odometry (1.013 and 1.015 on the two mapping sessions; 1.017 and 1.010
  against the ground truth), and the script scales the calibration's
  baseline by the inverse of their mean (0.986). Maps rebuilt with it are
  0.76 and 0.91 cm from the ground truth instead of 2.75 and 1.58 cm, and
  VIO + map falls from 2.8–6.9 cm to 1.2–5.4 cm RMSE on three queries.
  With `--weak-gate-cm 5` too, it is 1.1–2.4 cm.
- **Close weak fixes after the first map pose, as an option.**
  `OdomCorrectionConfig::weak_gate_m` (`basalt_euroc_localize_demo
  --weak-gate-cm`, `visloc_map_localize_node` `weak_gate_m`; off by
  default): once the correction is set, a weak fix within that distance
  (and 2°) of the corrected odometry is used as a fix. Localizations
  matched around the predicted projections can then be weak too. On
  OpenLORIS-Scene office1-7, whose images rarely reach 30 inliers against
  another session's map, VIO + map falls from 6.86 to 3.35 cm (map B) and
  from 7.62 to 3.80 cm (merged map). On the eight EuRoC queries, where
  full fixes are frequent, it changes the error by -0.6 to +0.5 cm, so it
  stays off.
- **`scripts/calibrate_wheel_offset.py`: where the wheel odometry is
  measured.** `--odometry` and the node's `extrinsics tf` place the camera
  relative to the odometry's body. If that body is not the point the wheel
  odometry reports (the turning centre), every turn moves the computed
  camera pose, which looks like drift. The script fits that planar offset
  from a run with both the VIO and the wheel odometry, with no ground truth,
  and prints the corrected `--imu-from-body`. On OpenLORIS-Scene the
  dataset's `base_link` sits 5–8 cm from it. Calibrated from one
  session's VIO, the wheel odometry + map error falls from 7.2–11.9 cm to
  4.4–6.2 cm RMSE over five queries. On office1-7 at 15 inliers it reaches
  2.27 cm, ahead of the VIO's 4.06 cm. Details in
  `docs/openloris_map_localization.md`.

- **Local contrast equalization for maps that must localize under other
  lighting.** `visloc_basalt::contrast::LocalContrast` equalizes an
  image's local contrast (contrast-limited adaptive histogram equalization
  over 8 × 8 tiles; black pixels stay black).
  `basalt_euroc_online_slam_demo --equalize-local-contrast` applies it to
  each keyframe image before ORB extraction (the VIO tracks the raw
  images), `map.json` records it as `feature_local_contrast`, and
  `basalt_euroc_localize_demo` and `visloc_map_localize_node` apply it to
  every query image of such a map; `basalt_map_merge` refuses to merge an
  equalized map with a raw one. On OpenLORIS-Scene the daylight session
  office1-4, which no night map localized before, gets a map pose for
  33.6% of its frames at 2.46 cm RMSE (at `--min-inliers 15`; its best
  image has 23 inliers); the dawn session still gets none. On EuRoC it
  changes little (MH_02 and V2_02 within 0.1 cm) except V2_03 (29.6 to
  39.8 cm), so it stays off by default. Results in
  `docs/openloris_map_localization.md`.

- **Keyframe retrieval for the map-wide search.**
  `visloc_localization::KeyframeRetrieval` ranks a map's keyframes by
  hashed visual words of their landmarks' binary descriptors (Basalt's
  `HashBow` scheme, histogram intersection through an inverted index).
  `BinaryLandmarkIndex::match_among` matches among the landmarks of the best
  ones. `basalt_euroc_localize_demo --retrieval-keyframes N` tries that
  before the whole map. On the EuRoC queries it localizes 0.1–0.9 points
  more images. On a merged OpenLORIS map it localizes 4.6% of office1-6's
  images instead of 2.9%. On a Machine Hall map grown session by session
  to 41,000 landmarks (`docs/basalt_map_localization.md`, "Lifelong use"),
  it cuts MH_05's median time per image from 125 to 74 ms. Off by
  default.

- **Map-based localization on a real wheeled robot (OpenLORIS-Scene).**
  `scripts/openloris_to_euroc.py` converts an OpenLORIS-Scene sequence's
  T265 data to the EuRoC layout with a Basalt calibration:
  - the fisheyes are resampled to a Double Sphere camera;
  - the accelerometer is interpolated to the gyro;
  - the ground truth (`base_link`) is moved to the IMU.

  `basalt_euroc_localize_demo --odometry FILE [--imu-from-body ...]`
  corrects an external odometry, such as the robot's wheel odometry, into
  the map frame instead of the VIO.

  `docs/openloris_map_localization.md` reports the office scene, seven
  sessions over four days:
  - Sessions with the same lighting, up to 2.8 days later with furniture
    moved, get a map pose for 95–99.7% of frames at 2.8–4.1 cm with the VIO.
    With the wheel odometry they get 7–12 cm, because it drifts in in-place
    turns.
  - No session localizes across the day/night lighting change.
  - Maps of opposite driving directions merge through a session that turns
    in place.

- **README: map-based localization animation.** EuRoC MH_02 with the VIO
  corrected into the MH_01 map's frame: the map-frame trajectory (a pose for every
  frame), the keypoints that matched the map and the rays to their
  landmarks (`docs/assets/hero_map_localization.gif`). It is rendered by
  `scripts/render_map_localization_gif.py` (`--trajectory` for a `--vio`
  run) from `basalt_euroc_localize_demo --dump-inliers FILE`, which writes
  each accepted localization's PnP inliers (keypoint pixel and map
  landmark).

- **`basalt_euroc_localize_demo`: matching around the predicted projections
  with `--vio` (and `--odometry`), by default.** Once the VIO has a map
  pose, each image is first matched around the landmarks' projections
  from the pose the corrected VIO predicts, as `visloc_map_localize_node`
  does, with the map-wide search as the fallback (`--projection-radius-px`,
  20; `0` turns it off).
  - Over the eight EuRoC queries, localizing every 5th image, more images
    localize (MH_02 98.4% instead of 81.6%, V2_02 56.4% instead of 21.1%)
    in about half the time.
  - The VIO + map error falls on 6 of the 7 queries with map poses (V1_03 4.60 cm
    instead of 6.66). On MH_05 it rises from 14.4 to 15.9 cm.
  - The prediction is used only within 5 s and 2 m of odometry travel after
    the correction's newest fix (`--projection-fresh-s`, `--projection-fresh-m`).
    After a 27 s gap on MH_03, a drifted prediction had produced fixes
    16–42 cm off.

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

- **Launch file, URDF and RViz config for the ROS 2 nodes.**
  `ros2/visloc-ros2/launch/map_localization.launch.py` starts
  `robot_state_publisher`, the VIO (`odom -> base_link`), the map
  localization (extrinsics from `/tf_static`) and optionally RViz with
  `launch/rviz/map_localization.rviz`. `tools/calib_to_urdf.py` writes a
  URDF from a Basalt calibration (`launch/euroc_rig.urdf` for EuRoC).
  `visloc_map_localize_node` publishes the map's landmarks once on
  `map_cloud` (`sensor_msgs/PointCloud2`, transient-local;
  `publish_map_cloud`). Tested with ROS 2 Jazzy: MH_02 through the launch
  file gives 99.6% of frames a map pose at 2.06 cm RMSE, and RViz shows
  the map, path, pose and TF tree.

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
  the corrected VIO gives 99.7% of frames a pose at 9.5 cm RMSE (73.1% at
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
    any gravity-aligned odometry can be corrected into the map frame, e.g.
    wheel odometry on a level floor. New parameter `camera_frame`.
  - `visloc_ros2::tf_tree::StaticTransforms` composes static transform
    chains like tf2; the nodes subscribe to `/tf_static` transient-local,
    as `tf2_ros` does.
  - On ROS 2 Jazzy, with `static_transform_publisher` providing
    `base_link -> imu0 -> cam0`, EuRoC MH_02 against the MH_01 map: the VIO
    publishing `base_link` gives a map pose for 99.5% of frames at
    2.21 cm RMSE; simulated wheel odometry (98.7 cm ATE on its own) 98.3%
    at 3.45 cm.

- **The real-time ROS 2 map localization is validated on all eight EuRoC
  query sequences** (rosbag2 at 1.0x, Fast DDS): within 0.5 points of the
  offline `basalt_euroc_localize_demo --vio` on the share of frames with a
  map pose (V2_03: 77.5%, from 21 s on), with no VIO reset and at most 15
  dropped frames per sequence; table in `ros2/visloc-ros2/README.md`.

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
  - Once the odometry has a map pose, `visloc_map_localize_node` predicts
    each image's pose from the corrected odometry. It then matches each map landmark only
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
    stereo-inertial VIO on the query sequence and corrects it into the map
    frame with the fixes of every `--localize-every`-th image
    (`visloc_localization::OdomCorrection`: a gated, gravity-aligned yaw +
    translation offset). Every frame gets a map pose in
    `trajectory_map.tum`; `trajectory_map.csv` adds the time and distance
    since the last fix. MH_02: 99.8% of frames at 2.0 cm RMSE localizing
    every 5th image, at 0.83× real time on 4 cores; on all seven queries
    with map poses no pose within 2 m and 5 s of a fix is more than 12.6 cm off,
    while poses carried tens of metres between rare fixes reach decimetres.

### Changed

- **The mapping ORB frontend is about four times faster, with the same
  output.** The corner detector (`detect_keypoints_mapping_with`) runs each
  filter along whole rows, reflecting only at their ends, and checks the
  minimum distance only against the chosen corners in the 3x3 cells around
  a candidate; the blur (`GaussianSmoothing`) sums its taps across a row at
  a time. Every sum keeps its order, so corners, descriptors and maps are
  the same to the bit (a test compares the detector with the direct
  computation). On a T265 image (848x800) with the blur and quality 0.003,
  detection, blur and description take 14 ms instead of 50 ms; on an EuRoC
  image detection takes 4 ms instead of 25 ms. The localize demo's median
  time per image falls from 28.9 to 14.5 ms on MH_02 against MH_01 and
  from 69.9 to 34.6 ms on office1-7 against office1-2's blurred map, with
  byte-identical trajectories. Times quoted elsewhere were measured before.

- **The IMU scale fits the gyro's bias.** A constant gyro bias bends every
  frame-to-frame rotation, and with it the camera-from-IMU fit and the
  frame-timing search.
  - `visloc_slam::estimate_imu_scale` now fits the bias after the first
    timing search, takes it off the gyro, and searches the timing again.
    This is `ImuScaleConfig::gyro_bias`, on by default; turn it off with
    `imu_metric_scale --no-gyro-bias`. The fitted bias is reported in
    `ImuScaleEstimate::gyro_bias`, the `gsplat_photos` log and
    `imu_scale.txt`.
  - On OverMaps360's 9d2fe9b1 the camera's gyro reads 1.8°/s off, which
    left a 1.18° rotation residual. The fit takes it to 0.05° and puts the
    frame timing back (drift 18.5 → 2.7 ms).
  - Over all 50 walks the bias is 0.07–1.69°/s, and the rotation residual
    falls from 0.03–0.60° to 0.01–0.05° on all walks but one. The scale
    moves by 0.06% in the median and 1.1% at most.
  - Against ARKit and GPS the agreement is unchanged (MAE 1.85% and 1.38%,
    from 1.86% and 1.34%; docs/insta360_imu_scale.md, "Gyro bias").

- **`gsplat_photos` frees the working images after SIFT, and `--insv`
  writes its lens faces to disk as it cuts them.**
  - Each SfM point takes its colour from its first keypoint, as before. The
    colours under the keypoints are now kept at SIFT time, so the decoded
    photos, cube faces or lens faces are freed after feature extraction
    instead of being held through SfM.
  - With `--insv` each frame is cut into its ten faces and written as it
    is read, and SIFT reads the faces back (PNG, so the same pixels).
  - The output is byte-identical for photos, `--mask-black`, `--equirect`,
    `--equirect-native` and `--insv` (100 frames).
  - On 4 cores and 15 GB, 100 frames with 1024 px lens faces peak at
    4.5 GB instead of 7.9 GB. 200 frames, killed at 14 GB before, finish
    at 8.6 GB.

- **`gsplat_photos` keeps VLFeat-compatible SIFT descriptors as bytes.**
  - They come out of SIFT as COLMAP's bytes, but were held as f32 vectors
    until matching ended, about 550 bytes a keypoint. They are now kept as
    bytes, 128 a keypoint.
  - They are turned into the same f32 values (`byte / 512` is exact) only
    where they are read: per chunk of pairs for the CPU matcher, per block
    for the GPU bank, per image for the retrieval, and before the
    intrinsics refinement.
  - The output is byte-identical for photos (with and without retrieval
    pairs), `--equirect` and `--insv`.
  - The mall walk's 200 frames with 1024 px lens faces peak at 4.5 GB
    instead of 7.8 GB.
  - At the defaults (1600 px faces, 8,192 keypoints; `--window 2`) the same
    frames peak at 9.5 GB, in 2 h 23 min on 4 CPU cores, most of it
    matching. On a CPU, `--max-size 1024 --max-keypoints 4096` is the
    practical setting (docs/photos_to_splat.md).

- **Weak fixes: localizations with 15–29 inliers can set the first map
  pose.** With `--vio` or `--odometry`, `basalt_euroc_localize_demo` passes
  a map-wide localization turned down only for too few inliers, but with at
  least `--weak-min-inliers` (15), to `OdomCorrection::add_weak_fix`.
  `visloc_map_localize_node` does the same (`weak_min_inliers`). Weak fixes
  count only toward setting the first offset, which then needs three
  agreeing fixes instead of two. Once it is set they are
  `FixOutcome::Unused`. Replaying all eight EuRoC queries through the rule,
  it never uses a fix more than 25 cm off. On OpenLORIS-Scene, office1-7,
  whose images never reach 30 inliers against another session's map, gets
  a map pose for 99.4% of its frames by default (6.86 cm RMSE). The
  daylight office1-4 gets one against an equalized night map (33.4%,
  2.62 cm), where before it needed `--min-inliers 15`. Rerun on all eight
  EuRoC queries, MH_04 gets its first map pose 2.6 s sooner (76.1% of the
  frames instead of 73.6%), V2_03 improves from 29.6 to 23.7 cm, and the
  other six change by at most 0.1 cm.

- **`OdomCorrection` measures travel along the odometry's path.** The gate
  widens with the odometry's travel since the last accepted fix, which was
  measured only between the odometry positions of consecutive fixes. After
  a long gap that is one straight line, and a correct fix on EuRoC V2_03, 65
  s after the previous one, was rejected. `OdomCorrection::observe_odometry`
  now takes every odometry pose and keeps the path length by stamp, so it
  also works when the odometry runs ahead of the localizations. The demo
  and `visloc_map_localize_node` feed it every pose. Replayed on EuRoC
  without weak fixes, it takes V2_03 from 29.1 to 24.6 cm and leaves the
  other seven queries unchanged.

- **Map localization tries every image until the odometry has a map
  pose.** No frame has one before two fixes agree, and with every 5th
  image tried, EuRoC V2_03 never got two. `basalt_euroc_localize_demo`
  (`--vio`, `--odometry`) and `visloc_map_localize_node` now try every image
  until the first map pose, then every `--localize-every`-th /
  `localize_every`-th. V2_03 gets a map pose for 77.8% of its frames, from
  21 s on (29.6 cm RMSE; 5.8 cm within 5 s and 2 m of a fix). On the other
  seven queries the first map pose comes up to 1.2 s sooner and the
  accuracy is unchanged; MH_04 and MH_05, whose starts the map does not
  cover, try about twice as many images. After 30 s without any
  localization (`--initial-window-s` / `initial_window_s`), where the map
  likely does not cover the place, they back off to every
  `localize_every`-th image until one localizes. `--initial-localize-every`
  / `initial_localize_every` (default 1) sets the start-up cadence; 5 is
  the previous behaviour.

- **`visloc_map_localize_node`: the projection prior expires.** The node
  matches around the predicted projections only within `projection_fresh_s`
  (5) seconds and `projection_fresh_m` (2) metres of odometry travel after
  the correction's newest fix. After a longer gap it matches map-wide, since a
  drifted prediction can steer the projection search to a wrong pose that
  agrees with it.

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
