# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- **Faster search of dense maps** (`basalt_euroc_localize_demo`):
  `--fast-radius` (the radius search through the packed index;
  `BinaryLandmarkIndex::landmarks_within`, `match_among_threaded`),
  `--prior-hold N`, `--threads N` and `--no-global-with-prior`. On a
  681k-landmark map, 300 images of a second walk took 150 s instead of
  564 s, with 169 of the 175 fixes kept. A failed image no longer reports
  the previous image's pose.
- `scripts/openloris_to_tum_rgbd.py`: OpenLORIS-Scene's D435i colour and
  aligned depth as a TUM RGB-D sequence with motion-capture camera poses.
  A map of office1-1 localizes office1-6 (2.8 days later, furniture moved)
  63% of the time at 2.4 cm median.

- **Selectable PnP solver for wall-heavy scenes** (`LocalizationConfig::pnp_solver`,
  `visloc_vision::pnp::PnpMinimalSolver`). The default localization
  pipeline's RANSAC can now use P3P instead of the 6-point DLT, which is
  degenerate when the matched landmarks are coplanar (a view of one wall).
  `PnpMinimalSolver::Dlt` stays the default, so existing results do not
  change. A built pipeline switches with `set_pnp_solver` /
  `with_pnp_solver`. Also `--pnp-solver dlt|p3p` in
  `localize_openloris_map` and `localize_rne_map_sequence`, and
  `pnp_solver="p3p"` in the Python `localize` / `estimate_pose_pnp_ransac`.
  On a synthetic 25-point wall, DLT fails and P3P recovers the camera centre
  to under 0.1 mm. API change: the default pipeline type is now
  `LocalizationPipeline<_, _, LocalizationRansac>`
  (`PnPRansac<PnpMinimalSolver>`). Code that names
  `LocalizationPipeline<_, _, PnPRansac>` and calls `default()`, `new` or
  `with_candidate_selector` must switch to `LocalizationRansac`.
- **`visloc_vio_node` drives on through lost tracking on wheel odometry**
  (`wheel_odom_topic`, `visloc_ros2::odom_bridge`). A lost-tracking reset
  no longer silences the outputs: the pose follows the wheel odometry until
  tracking holds again, then the new estimator continues from there. The
  `odom` frame never jumps, so no `odom_reset` is published. In Gazebo, a
  5 s camera blackout no longer stops the robot: it reached 5/5 goals at
  the first try (1.62 m driven blind) in 3 of 4 runs. The fourth run got
  no goal through, because Nav2's controller never received `map -> odom`.
  `nav.launch.py wheel_odom:=`.
- **People in the 3DGS simulator** (`gs_sim.py --pedestrians N`,
  `gs/people.py`): they walk the route in the camera, drawn over the splat
  by depth, and in the scan. With eight people on the 360-degree street,
  Nav2 reached 4/4 goals with no contacts, at 4.2 cm `/map_pose` RMSE.
  `chase_video.py` draws them. They are gaussians rendered with the scene
  by default (`gsplat_render_server --dynamic`, `GpuScene::write_range`).
  `--cyclists` adds bicycles with pedalling riders. Everyone keeps clear of
  the robot and of each other (no overlap over 2 cm). With 6 walkers and 2
  cyclists, Nav2 reached 4/4 goals at the first try at 3.5 cm RMSE.
  `--dog-walkers`, `--strollers` and `--cars` (on a road estimated from
  the splat's grey ground, or given with `--road FILE`) add more traffic.
- **`gs/real2sim.py`**: an Insta360 `.insv` walk to a simulator, its map
  and, with `--nav`, Nav2 in one resumable command. It took about 1 h 45
  min on 4 CPU cores for a 121 m walk.
- **Appearance robustness for maps of rendered images**:
  `tum_rgbd_save_map --smoothing-sigma` and `--local-contrast` (recorded in
  `map.json`, so the localizers prepare queries alike).
  `gs/perturb_replay.py` sweeps darkening, gamma, noise, blur, synthetic
  night and patches over a replay. Smoothing keeps 838 of 858 images
  localized under σ 10 noise, where the plain map gets 218 fixes, metres
  wrong. `gs/real_map_test.py` localizes real held-out photos, splat
  renders and a second walk against a map of real photos. With
  `tum_rgbd_save_map --min-observations 1`, a real-photo map localizes
  58% of a second walk's images 12 minutes later (from 12%), all
  consistent with that walk's SfM, at 6x the search time.
- **Real to sim: a robot simulator made of a 3D Gaussian splat** (docs/navigation.md,
  `ros2/visloc-ros2/gs/`). From public photos (TUM RGB-D's Pioneer hall),
  visloc-rs's SfM (put in metres by a 7-number fit to the motion capture,
  3.7 cm RMSE) and 3DGS (held-out PSNR 25.0 dB) make the world;
  `gs_sim.py` drives a robot through it with the camera rendered by the new
  `gsplat_render_server` (pose in, image and optionally depth out), wheel
  odometry, a scan of a collision grid built from the splat, and `/clock`.
  Nav2 on `visloc_map_localize_node` reaches 5/5 goals with no contacts at
  8.7 cm `/map_pose` RMSE, on a map built from the splat's renders
  (`render_tum_seq.py`); a map of the real images localizes 4 of 450
  rendered images. Also `gsplat_train --alpha-mask`, `level_sfm.py`;
  and from a public 360-degree street walk (OverMaps360, its own IMU for
  scale, no ground truth), Nav2 reaches 4/4 goals along 121 m at 3.6 cm
  `/map_pose` RMSE with a longer projection window (`equirect.py` renders
  the robot's 360-degree view for videos).
- **Robot navigation with Nav2** (docs/navigation.md). New crate
  `visloc-navigation`: `OccupancyMapper` builds the 2D occupancy grid of a
  visual map from its landmarks and camera-to-landmark rays (a 2.5D log-odds
  grid, so low furniture survives rays passing over it), written in ROS
  `map_server` format; `examples/visual_map_to_occupancy.rs` converts a saved
  map for Nav2. `visloc_navigation::sim` simulates a camera robot (textured
  apartment, differential drive, drifting wheel odometry, synthetic
  detections, depth scan, mapping run) and `sim_robot` localizes it with
  visloc's pipeline (P3P-RANSAC) and `OdomCorrection`. New ROS 2 node
  `visloc_nav_sim_node` (with `sensor_msgs/LaserScan` and
  `geometry_msgs/Twist`), `launch/nav_sim.launch.py`, Nav2 parameters without
  AMCL, and `tools/nav_sim_goals.py`. Nav2 Jazzy in simulation: 5/5 goals at
  3.8 cm pose RMSE and 0 collisions on visloc localization, 0/5 on wheel
  odometry alone, 5/5 with an unmapped obstacle (replanned). ROS-free:
  `examples/nav_sim_build_map.rs`, `examples/nav_sim_drive.rs`. In Gazebo
  (`ros2/visloc-ros2/gz/`: world with rendered stereo cameras, IMU and lidar,
  mapping recorder, goal runner): Nav2 on the real nodes
  (`visloc_vio_node` + `visloc_map_localize_node`) and a map from visloc's own
  SLAM reaches 5/5 goals at 1.6 cm `/map_pose` RMSE, and 5/5 with an
  unmapped box (lidar obstacle layer, detour). The Gazebo Nav2 parameters
  lower both obstacle-height limits, since the SLAM map's frame starts at
  the IMU, 0.5 m above the floor. `visloc_vio_node` detects lost tracking
  (`lost_min_observations`, `lost_max_speed`; off by default): it resets the
  estimator and publishes nothing until tracking holds again, and announces
  each reset on `odom_reset`, on which `visloc_map_localize_node`
  relocalizes (it used to keep its old `map -> odom` across a VIO reset). A
  5 s camera blackout in Gazebo: the robot stops within 7 cm and goes on
  1 s after the cameras come back (5/5 goals, 4.4 cm max error), instead of
  driving 1.5 m blind with up to 43 cm of error. A threshold of 20
  observations raised a false alarm under extra CPU load; 10 did not. The
  Gazebo goal runner can turn a robot that stopped in front of a bare wall
  until it relocalizes (`--spin-after`). A 3D video of the runs:
  `docs/media/gz_nav2_blackout_3d.mp4`.
  In the public Depot warehouse (OpenRobotics, Gazebo Fuel, CC BY 4.0;
  `gz/make_depot_world.py`, which projects its meshes into a collision grid,
  and `--scene` for the route and goal tools): SLAM within 4.0 cm of the
  ground truth over 115 m, 5/5 goals at 6.1 cm `/map_pose` RMSE through the
  blackout with the monitor (20 cm max error without). Map localization now
  carries the last map pose over an odometry reset as a projection prior
  (`reset_prior_s`): without it the robot did not relocalize there.
  `scripts/plot_nav_sim.py` draws the runs.
- **GPS priors in incremental SfM (COLMAP `pose_prior_mapper` /
  `use_prior_position`).** `IncrementalSfmConfig::position_priors` takes a
  prior camera centre and per-axis σ per image.
  - **What it does.** Once three registered images have one, every global
    and periodic BA first aligns the model to the priors with a robust,
    deterministic Sim(3). It then adds a covariance-weighted camera-centre
    prior in place of the anchor/scale-anchor gauge. The model comes out
    metric in local ENU.
    - Every prior stays in the BA, under a Cauchy loss by default (COLMAP
      uses least squares; `--gps-prior-least-squares` restores that).
    - Without priors, or with fewer than three, nothing changes.
  - **Inputs.** `visloc_io::gps_prior` reads EXIF GPS and a
    `NAME LAT LON ALT [STD_H [STD_V]]` (or Cartesian) prior file and
    converts fixes to ENU (exact WGS84).
  - **CLIs.**
    - `unordered_sfm_demo --gps-priors FILE | --gps-priors-from-exif`
      (incremental mapper).
    - In the `colmap` shim, `feature_extractor` stores EXIF GPS, and
      `pose_prior_mapper` / `mapper --Mapper.use_prior_position 1` use it.
    - Both write `enu_origin.txt`. σ defaults to 1 m per axis, as COLMAP.
  - **Real onboard GNSS** (gravel-mine UAV set: 158 nadir images, low-cost
    GNSS, a reference adjusted against surveyed control points; default
    SIFT).
    - The block's shape is right to 2.5 cm with priors, 0.47 m without.
    - Camera positions land at 0.18 m once the GNSS's constant bias is
      removed; the GNSS itself scatters 1.72 m.
    - Least squares, Cauchy and σ 5/10 m are all equal there.
  - **ETH3D courtyard, simulated GPS** (σ 0.5/0.5/1 m, two 30 m jumps):
    38/38 registered, RMSE 0.30 m with the Cauchy default, 1.21 m with
    least squares.
  - **GPS pair selection** (COLMAP `spatial_matcher`):
    `unordered_sfm_demo --pair-source spatial` and `colmap spatial_matcher`
    pair each image with its nearest fixes (50 neighbours within 100 m,
    horizontal; `visloc_io::gps_prior::spatial_pairs`).
    - On the gravel set they recover 2,748 of the 2,781 pairs exhaustive
      matching verifies, from 37 % of the candidates.
    - VLAD retrieval was as good and cheaper there with the default SIFT.
  - Details in docs/gps_prior_sfm.md.

- **Localize a photo whose focal length is unknown**
  (`LocalizationPipeline::localize_with_unknown_focal`, `UnknownFocalLocalization`):
  P4Pf RANSAC (`visloc_vision::pnp::estimate_pose_with_focal`, minimal solver
  vendored from OpenMVG-rs) estimates the focal length from the 2D-3D
  correspondences, then the usual estimator and quality gate run with it.
  Leave-one-out on the browser SfM's courtyard sample: 36/38 localized at a
  1.03% median centre error, like the true-focal oracle (1.13%), against
  351% with the no-EXIF prior (docs/unknown_focal_localization.md,
  `bindings/sfm-web/examples/unknown_focal_loo.rs`).

- **`merge_maps` merges maps of another calibration of the rig**
  (another day's): a camera that differs from the base map's under the same
  id is added under a new id (`MergedMap::camera_ids`), instead of failing
  with `CameraMismatch`. `basalt_map_merge` records such a session's
  `t_imu_cam` in `map.json`. `scripts/euroc_stereo_to_rgbd.py` turns a
  rectified stereo sequence and its SLAM trajectory into the RGB-D layout of
  `tum_rgbd_save_map` (`--keep-world` keeps a visual-inertial world), and
  `fetch_kitti_raw_drive.py --colour-frames` fetches the colour cameras of
  chosen frames. Three KITTI raw drives on two days merge into one district map at
  1.12 m RMSE against the OXTS (docs/basalt_map_localization.md, GIF under
  KITTI's CC BY-NC-SA 3.0). `render_map_merge_gif.py --zoom` adds a
  close-up of where a session joined, `--drive` a street-level drive along a
  session's trajectory.

- **Maps from TUM RGB-D sequences, merged across sessions**
  (`examples/tum_rgbd_save_map.rs`). It saves a map in the layout
  `basalt_euroc_online_slam_demo --save-map` writes, from the RGB images,
  their depth and a trajectory (the virtual-stereo VO), levelled by the
  Kinect's accelerometer or, without one, the map's largest horizontal
  plane. `basalt_map_merge` merges four parts of fr3_long_office_household,
  each seeing one side of the desks, into the whole office at 13.1 cm RMSE
  against ground truth (docs/tum_rgbd_benchmark.md).
- **Map-merge GIFs** (`scripts/render_map_merge_gif.py`): separate SLAM maps
  growing in their own frames, then moved by the merge's transforms; with
  `--cloud`, RGB-D colour clouds (`scripts/rgbd_map_cloud.py`). EuRoC's five
  Machine Hall flights in one map at 6.2 cm RMSE (README), OpenLORIS-Scene's
  office over three days at 4.6 cm.

- **CUDA trainer: Mip-Splatting's 3D smoothing filter (opt-in,
  `gsplat_train_cuda --filter3d 0.2`, `CudaTrainer::set_filter3d`).**
  - **How.** Each gaussian's covariance gains `0.2 (z / fx)^2`, the finest
    footprint a training camera gives it. Its opacity is scaled so its
    integral stays the same. The filter is recomputed every 100 steps and
    after each refine.
  - **The step.** It renders and differentiates the filtered parameters,
    then chains the gradients to the raw ones (ABI 18).
  - **Output.** Renders, meshes and the written splat carry the filter.
  - **Evaluation at other resolutions.** `--eval-zoom <dir>` scores the
    held-out views on images at another resolution.
  - **Mip-NeRF 360, nine scenes, at the training resolution.** Level:
    27.89 dB on average against 27.86, with 0.4% more training time.
  - **Trained on `images_4`, rendered at 2x** (`images_2`): bonsai
    +1.38 dB and room +0.76 dB. At 0.5x (`images_8`) both gain
    +0.36-0.38 dB (docs/gsplat_cuda.md, "Mip-Splatting's 3D filter").

- **`visloc_vio_node` with the online mapper, through `ros2 bag play` on ROS 2
  Jazzy** (EuRoC V1_01, 1.0x, Fast DDS):
  - stereo: shutdown trajectory 0.0355 m (live `map_pose` 0.0385 m), the
    same as the DDS-free replay;
  - mono: 0.034 m;
  - all 2,912 frames processed in both runs, no mapper packet dropped, and
    `map -> odom` on `/tf` for every frame (ros2/visloc-ros2/README.md).

- **`visloc_vio_node` runs monocular-inertial VIO (opt-in, `mono:=true`).**
  - The calibration is cut to one camera (`mono_camera`, default 0), and only
    `left/image_raw` is subscribed. Every image is a frame once IMU data
    covers it (`VioInputs::new_mono`).
  - With `mapper:=true` the online mapper runs on the one-camera rig.
  - `SensorPacket::right` and `StereoPair::right` are now `Option`.
  - `euroc_mapper_replay --mono` measures it on V1_01 (0.5 s averaged tilt):
    VIO 0.088 m, live `map_pose` 0.086 m, shutdown trajectory 0.040 m, the
    same as the demo's `--mono` run. No packet was dropped.

- **GNSS fusion: an opt-in robust kernel on the VO factors**
  (`GnssPoseGraphConfig::vo_robust`, `kitti_raw_gnss_vo_fusion --vo-robust
  cauchy:<c>|huber:<delta>`). A corrupted VO step is down-weighted instead
  of the correct fixes around it being rejected.
  - KITTI raw 0027 (a truck crossing) with native fixes: batch 0.61 →
    0.18 m (Cauchy 3), fixes rejected 17 → 0, window 0.87 → 0.19 m.
  - With degraded GNSS it changes nothing to two decimals.
  - On 0020 the max error grows (1.21 → 1.79 m), so it stays off by default
    (docs/gnss_fusion.md, "Robust VO factors").

- **Monocular-inertial online SLAM: `basalt_euroc_online_slam_demo --mono`.**
  The online mapper (loop closure, global BA) now runs on a one-camera rig.
  - **What changed.** The mapper's stereo stage, both batch
    `NfrMapper::match_stereo` and the online per-timestamp match, returns
    without matching when the calibration has one camera. Before, it failed
    on the missing camera 1. Landmarks then come from temporal keyframe
    matches alone. A stereo rig never takes the branch: the stereo VIO
    trajectory is byte-identical.
  - **EuRoC, all 11** (with `config.vio_init_accel_average_s: 0.5`):
    - mean ATE 0.139 m for the VIO alone, **0.062 m** with the mapper;
    - every sequence improves;
    - lower than ORB-SLAM3 mono-inertial's published values on 5 of 11
      (its mean 0.043 m);
    - scale error −1.3 % to +1.8 %.
  - **TUM-VI room1:** 0.096 m → 0.014 m (stereo online 0.011 m).
  - Details in docs/mono_inertial_vio.md, "Mapping and loop closure".

- **`config.vio_mono_camera`: which camera a monocular EuRoC replay reads
  (opt-in, 0 or 1).**
  - **Why.** V2_03's mono failure at 73 s comes from cam0 dropping every
    other frame (414 gaps, mostly 55–75 s; cam1 has none). Turning at
    about 1 rad/s, the KLT tracks then live one frame, and no landmark is
    triangulated for about 15 s.
  - **Result on V2_03.** Reading cam1 takes the mono VIO from 0.828 m to
    0.104 m (scale +19 % → −0.2 %), and mono online SLAM from 0.100 m to
    0.045 m.
  - Urgent keyframes, IMU-seeded KLT and a fourth pyramid level only got
    cam0 to 0.21–0.29 m. On V1_01, which has no gaps, cam1 is not better
    (0.052 against 0.047 m).
  - `BasaltCalibration::select_camera(index)` builds the one-camera rig.
    Absent or 0, and on the stereo path, nothing changes.

- **`visloc_vio_node` can run the online mapper (opt-in, `mapper:=true`).**
  Loop closure and global BA as in `basalt_euroc_online_slam_demo`, with
  its mapper settings, so a ROS 2 user gets loop-closed poses.
  - The estimator runs the MargData-retaining path (same trajectory, checked
    by a test) and sends mapper packets through a bounded queue to a mapper
    thread. The VIO never waits: a full queue (`mapper_queue`, 64) drops
    the packet and counts it in the status line.
  - Live output, REP-105: `map -> odom` on `/tf`, from
    `T_mapper(k) * T_vio(k)^-1` at the newest keyframe the global BA has
    optimized, plus `map_pose` and `map_path`.
  - On SIGINT/SIGTERM the node runs the mapper's final optimization and,
    with `mapper_trajectory_out`, writes every frame's loop-closed pose
    (TUM, corrections interpolated between keyframes like the demo's
    `trajectory_online.tum`). A second signal exits at once.
  - Without `mapper` the node's outputs and signal handling are unchanged.
    The mapper's output is not deterministic run to run.
  - On EuRoC V1_01, replayed through the node's own stages and mapper
    session without DDS (`ros2/visloc-ros2/examples/euroc_mapper_replay.rs`,
    paced at the recording's rate):
    - the shutdown trajectory has ATE 0.0354 m, the same as the demo;
    - the live `map_pose` has 0.0385 m, against 0.0390 m for the VIO;
    - no packet was dropped.

    Not yet run through `ros2 bag play` (ros2/visloc-ros2/README.md,
    "Online mapper").
  - `BasaltVioFrontendStage::set_retain_marg_data` runs the two-thread
    stages on the MargData path (default off, lean path unchanged). The
    demo's keyframe-correction propagation moved, unchanged, to
    `visloc_basalt::mapper_correction`.

- **VO + GNSS fusion on real data: KITTI raw with OXTS ground truth.**
  - **What.** Three `2011_09_30` drives (odometry 07/06/05; 695 m, 1.2 km,
    2.2 km). The in-repo deep-frontend stereo VO is fused with the OXTS
    positions, degraded to a simulated 1 Hz receiver: 2 m / 4 m sigma, 80 %
    Gauss–Markov with τ = 45 s, multipath jumps of 10–30 m, and a 20 s
    dropout, from a fixed seed.
  - **Result.** Batch fusion beats both inputs on every drive and seed. The
    3-seed means are 3.5–4.8 m, against 9.2–10.1 m for the SE(3)-aligned VO
    and 6.0–6.6 m for the raw fixes. The gain is much smaller than the synthetic
    improvement (5.25 → 0.90 m). The correlated receiver error is not
    averaged out, and the sliding window is worse than batch
    (docs/gnss_fusion.md, "Results on KITTI raw").
  - **Negative results.** Correct fixes are rejected when the VO fails. Near
    a crossing truck on 0027 the error peaks at 4.9 m, even with native RTK
    fixes. The deep frontend's motion-scale rescue locks at 1.57 m per frame
    while the car decelerates on 0020 (VO 30 m). Both are recorded.
  - **Defect found, opt-in fix: `GnssVoFusionConfig::gate_at_fix_accuracy`.**
    Splitting a 1 Hz fix's information over ten 10 Hz frames also loosened
    its χ²/GNC gate by √10, so multipath jumps were accepted (0 of 151
    adjacent frames flagged on 0027). With the option, consistency is judged
    at the fix's own accuracy: 74–79 % of those frames are flagged, none
    elsewhere in batch, and the 3-seed mean batch ATE drops by 0.1–0.5 m
    (one of nine runs is 0.04 m worse). The weights are
    unchanged (`GnssPoseGraph::gnss_gate_scales`). It is off by default, and
    default outputs are unchanged (synthetic demo still 0.898 m, 19 frames).
  - **New code.** `visloc_io::kitti_raw_gnss` converts WGS84 to local ENU,
    gives OXTS IMU poses and GNSS fixes, and reads the raw
    `calib_{cam_to_cam,imu_to_velo,velo_to_cam}` chain.
    `examples/kitti_raw_gnss_vo_fusion.rs` and
    `scripts/run_kitti_raw_gnss_fusion_eval.sh` run the evaluation.
    `fetch_kitti_raw_drive.py` gains `--skip-tracklets` /
    `--skip-extract-oxts`.

- **A container image: `ghcr.io/rsasaki0109/visloc-rs`.** `gsplat_photos`,
  `visloc_mvs`, `colmap_incremental_mapper` and `visloc_web` on Debian 12,
  built by the new `Dockerfile` and pushed by the release workflow's new
  `image` job for each version tag (`:<version>`, `:<major>.<minor>`, and
  `:latest` for the newest tag). The GPU paths use wgpu through Vulkan: with
  `--gpus all` (NVIDIA Container Toolkit) on the GPU, otherwise on Mesa's
  software Vulkan device (docs/docker.md).

### Fixed

- 3DGS sim cars (`gs/people.py`): a car held up in the queue at the light
  no longer turns back mid-road and drives on the wrong side while it
  crosses over. Cars wait, keep right, and U-turn only at the road's
  ends, in a half circle.
- 3DGS sim cars look and move like cars: sedans, hatchbacks and SUVs
  lofted from side profiles (wheel arches, glass and pillars, lights,
  plates, mirrors) in street colours, with a shadow. They no longer start
  and stop at once: 2 m/s² up, at most 6 m/s² down, braking for the car
  ahead and for the line, and slowing for the U-turn.
- 3DGS sim cars drive along the street, not along the robot's route
  offset sideways (`people.fit_road_lines`, `gs_sim --no-road-fit`):
  where the route cut across a side street they had driven 26° off it.
- 3DGS sim: people no longer walk into cars. A car's footprint is its body
  (seven circles as wide as it, and its corners), not three circles with a
  waist. Walkers wait for cars on or close to the crossing and go back if
  blocked there; head-on deadlocks on the pavement resolve by brushing
  past (people and bikes only). With 4 seeds over 600 s: nobody 5 cm into
  a car, no standstill over 20 s (there had been 400 s ones).
- 3DGS sim people look and walk like people: a lofted torso (shoulders,
  waist, hips), tapered limbs, shoes, a face left clear by the hair, and
  everyday clothes (trousers or a skirt, a T-shirt, a top or a jacket,
  long hair, a bun, a cap, a backpack). Knees bend in the swing, the
  pelvis bobs, and the arms swing with a bent elbow.
- `gs/lane_map.py`: Nav2's map free only within 1 m of the walk the 360
  video was taken on, so the robot keeps to that pavement
  (`gs_nav.launch.py occupancy:=`; `real2sim.py` by default,
  `--lane-half-width`). On the street, the robot's distance from the walk
  went from 0.37 m median / 2.5 m max to 0.20 / 0.55 m; 4/4 goals, 4.1 cm
  RMSE.
- `lane_map.py` cuts the walk's detours (a walker crossing a side street
  halfway down it), so the robot crosses on the crossing at the street's
  mouth. Off the lane is very costly rather than lethal, small clumps of
  stray map points on it are cleared, `make_gs_scene.py --open-path` opens
  the sim's fence along the lane, and `real2sim.py` puts the goals on it.
- 3DGS sim: posts, sign poles and tree trunks are solid
  (`make_gs_scene.py --column-bins`). The robot had driven through them,
  since too few of their thin gaussians passed the obstacle filters.
- 3DGS sim: kerbs. Where the robot crosses a street, the kerbs are found
  in the splat's colours (`kerbs.find_kerbs`, warm pavement against grey
  road; `make_gs_scene.py --crossings`). The road between them is 10 cm
  lower, the robot pitches on its wheels over them, and it crawls at
  0.12 m/s while a wheel is on one (`gs_sim.py --kerb-speed`).

- **Stereo VO no longer hangs on KITTI odometry 01.** On raw drive
  2011_10_03_0042 the stereo VO stopped for good at pair ~326.
  - **Cause.** `kabsch_n` converted its rotation with nalgebra's
    `UnitQuaternion::from_matrix`, an iterative projection with no iteration
    limit. On a ~180° rotation from a bad RANSAC sample it cycles forever.
  - **Fix.** The same projection, capped at 1,000 iterations, falls back to
    the matrix itself, which is already a rotation. Every converging case
    gives the same bits as before.
  - **Result.** The whole drive now runs (1,170 frames in 423 s), and its
    first 260 poses are byte-identical. Its accuracy over the full highway
    drive is poor (263 m ATE against OXTS); only the first 260 frames were
    ever tuned.
  - Every other unbounded `from_matrix` call in the workspace now goes
    through `visloc_core::geometry::rotation_from_matrix` (next entry).

- **No unbounded rotation projection left in the workspace**
  (`visloc_core::geometry::rotation_from_matrix`).
  - **What.** It replaces nalgebra's `UnitQuaternion::from_matrix` in P3P,
    PnP, generalized PnP, two-view, `SO3::from_matrix`, gsplat-train's photo
    alignment, the SfM debug alignment, the Basalt mapper's loop geometry and
    six examples.
  - **How.** It runs the same iteration with a cap of 1,000 and accepts the
    result only if one more iteration leaves it unchanged, which is how the
    unbounded loop stops. A converged result has the same bits as before:
    a test checks 200 rotations and general matrices.
  - **Otherwise.** It takes the nearest rotation from the SVD, which covers
    the half turn and other cycles. A non-finite input, which also looped
    forever, now returns a rotation.

### Changed

- **Built-in SIFT defaults to the COLMAP-compatible extractor**
  (`unordered_sfm_demo --feature-extractor sift` and the `colmap` shim's
  `feature_extractor`).
  - The VLFeat/COLMAP-compatible detector (subpixel localization) and
    descriptor, with `gsplat_photos`' settings (L1-root, magnification 3,
    two orientations, larger scales first).
  - The historical extractor reports integer-pixel keypoints. It stays
    behind `--sift-legacy`, and is also used, with a note, when a flag
    only it supports is given (`--sift-affine`, `--gpu-sift`, ...).
  - **ETH3D courtyard** (38 photos): 38 registered instead of 23, and the
    prior-free shape error is 0.088 m instead of 2.15 m.
  - **Gravel-mine drone block:** 158 registered instead of 142–147, and
    the shape error with GPS priors is 2.5 cm instead of 0.6 m.
  - **The `colmap` shim** registers 37 of the 38 courtyard photos instead
    of 12.
  - The shim keeps the legacy extractor with `--SiftExtraction.use_gpu 1`.
  - Feature extraction is slower: CPU only, with more keypoints matched.

- **Stereo VO: the motion-scale rescue no longer locks in on a slowdown**
  (`StereoVoFrontendConfig::motion_scale_rescue_release_full_window`, on by
  default; `--no-motion-scale-rescue-release` restores the old behaviour).
  - **The bug.** The rescue judged each step against its own rescued steps.
    During a real, sustained slowdown, every slower step counted as
    "collapsed" and was scaled back up to the old speed, for good.
  - **The fix.** Once a whole history window (20 pairs) has been rescued in a
    row, the streak is handed back to the raw step lengths and the current
    step is kept.
  - **KITTI raw, pure-Rust deep frontend:**
    - 0020: VO 30.2 → 10.2 m, GNSS-fused 24.4 → 4.8 m;
    - odometry 09: VO 122.3 → 18.5 m;
    - 0027 and odometry 01's 260-frame highway slice, which the rescue was
      tuned on, are byte-identical.
  - `online_slam_stereo_vo_kitti_demo` also gains
    `--motion-scale-rescue-max-inlier-ratio` (docs/gnss_fusion.md).

- **A manual run of the release workflow takes its release notes from its
  branch's `CHANGELOG.md`** (normally `main`) instead of the tag's, so a
  rerun keeps later edits such as 0.4.0's highlights; `notes_ref` still
  picks another ref. A new `image` input chooses whether the image is
  rebuilt.

### Fixed

- **The default SIFT was not rotation invariant** (CPU `extract_sift` and
  the GPU shaders, which mirrored it). A 180° turn of an image left 3
  ratio-test matches, all wrong.
  - **Orientation.** The isotropic orientation histogram read the gradient
    at the keypoint's row for every window row, so orientations stayed near
    0 / π whatever the image rotation.
  - **Descriptor.** The default and scale-adaptive descriptors binned
    samples by their image-frame offset instead of the keypoint-frame one,
    so the 4×4 grid never turned with the keypoint.
  - **Test.** The new `sift_descriptors_are_rotation_invariant` turns a
    texture by 90° and 180°: 4 / 3 consistent matches before, 163 / 149
    after.
  - **Effect on the gravel-mine drone block**, whose strips alternate
    heading by ~165°: exhaustive SfM registers 142 of 158 images (was 54)
    and verifies 1,280 pairs (was 448).
  - **Effect on ETH3D courtyard** (upright photos): 262 verified pairs and
    32k inlier correspondences (were 233 and 21k).
  - **The default SIFT output changes**, so results measured with the
    built-in extractor shift. That includes
    `--sift-vlfeat-compatible-descriptor`: its descriptor was already
    rotation invariant, but it takes the default detector's orientations.
    `--sift-vlfeat-compatible-detector` has its own orientation code and
    is unchanged.

## 0.4.0 - 2026-11-17

### Highlights

- **Localization in a saved map, for robots (ROS 2).**
  `basalt_euroc_online_slam_demo --save-map` saves a VI-SLAM map, and
  `visloc_map_localize_node`, next to `visloc_vio_node`, localizes against
  it and publishes `map -> odom`.
  - EuRoC MH_02 against an MH_01 map, replayed over DDS at real time with
    all three processes on 4 cores: 99.6% of frames get a map pose, at
    2.05 cm RMSE and a median latency of 49 ms.
  - Ground robots get `base_link`, wheel odometry, a wheel-odometry
    calibration node (`visloc_wheel_calibration_node`), and a launch file
    with a URDF and an RViz config.
  - Tested on ROS 2 Jazzy and Lyrical, with Fast DDS and Cyclone DDS.
- **On a real wheeled robot (OpenLORIS-Scene).** The office scene has seven
  sessions over four days. Sessions with the same lighting, up to 2.8 days
  later and with furniture moved, get a map pose for 95–99.7% of frames at
  2.8–4.1 cm. No session localizes across the change from day to night.
- **CUDA backends (experimental, opt-in).** Both load at runtime, and Cargo
  never compiles CUDA.
  - A 3DGS rasterizer and trainer (`visloc-gsplat-cuda`), for machines
    where wgpu cannot reach the GPU, such as WSL2. south-building trains
    30k steps in 691 s on an 8 GB laptop GPU, at 23.22 dB held-out PSNR.
  - PatchMatch MVS (`visloc-mvs --features cuda`). It scores what the CPU
    path does, 16 to 21 times faster.
- **Meshes from splats and MVS together.** Two new mesh paths: the splat's
  expected depth, and MVS depth maps with the splat filling their holes.
  On three ETH3D scenes, mesh F1 at 2 cm rises from 33.3% (a splat's own
  mesh) to as much as 71.0%.
- **Semantic splats.**
  - SAM 3 labels give each gaussian a class and an object.
  - `gsplat_extract --select` / `--drop` cut out or remove classes or
    single objects, and the web viewer names the object under a click.
  - On FIORD MeetingRoom_In: 82.6% of held-out pixels and mIoU 71.3% over
    11 classes. 3,728 SAM 3 instances become 65 objects.
- **Fisheye, distorted and 360° cameras end to end.**
  - SfM, 3DGS and meshes now run through every COLMAP lens model,
    including fisheyes wider than 180°.
  - Bundle adjustment can refine the distortion
    (`gsplat_photos --refine-camera`).
  - A dual fisheye's two lenses are solved as one rig.
  - Per-image and per-camera masks.
  - `gsplat_photos --insv` turns an Insta360 video straight into a metric
    model. On five OverMaps360 walks it registered 99.4–100% of the faces.
- **Visual-inertial odometry (VIO).**
  - The Basalt port's monocular-inertial VIO runs live in a phone browser
    (`bindings/wasm`: iPhone Safari, Android Chrome).
  - On EuRoC, an averaged initial tilt (`vio_init_accel_average_s`, opt-in)
    lets the monocular VIO track all 11 sequences; without it, 3 diverge.
- **Faster, with the same output.**
  - Basalt's KLT frontend is about 38% faster, and its VIO estimator about
    26% faster, both bit-identical.
  - The mapping ORB frontend is about four times faster.
- **Breaking:** `GpuMatcher::match_pairs` (and `match_pairs_async`) returns
  a `Result` (#469).
- **Release tooling.**
  - The release workflow can publish the crates to crates.io and the
    `visloc` Python package to PyPI. This is opt-in and was not done for
    0.4.0 (docs/publishing.md).
  - This release ships as a GitHub release with binaries for Linux,
    Windows and macOS.

### Full change log

The itemized 0.4.0 change log (every Added / Changed / Fixed entry from
2026-10-25 to 2026-11-17, about 1,670 lines) is archived in
[`docs/archive/changelog_0.4.0_full.md`](https://github.com/rsasaki0109/visloc-rs/blob/main/docs/archive/changelog_0.4.0_full.md).

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
