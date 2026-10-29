# visloc-ros2

ROS 2 nodes for visloc-rs, written against the pure-Rust ROS 2 client
[`ros2-client`](https://crates.io/crates/ros2-client) (built on
[RustDDS](https://crates.io/crates/rustdds)). They talk DDS/RTPS directly:
**no ROS 2 installation, `colcon` workspace or `rclcpp`/`rclrs` toolchain is
needed to build or run them**, and they still join an ordinary ROS 2 graph.

| binary | what it does |
|---|---|
| `visloc_vio_node` | Stereo-inertial VIO: the repository's faithful Basalt port (`pipelines/basalt`) fed from two `sensor_msgs/Image` streams and `sensor_msgs/Imu`. Publishes odometry, pose, path and TF. |
| `visloc_localize_node` | Map-based localization: loads a COLMAP model plus a landmark descriptor store at start, extracts SIFT on each incoming image, matches against the map and solves PnP-RANSAC. Publishes the camera pose, inlier count and diagnostics. |
| `visloc_map_localize_node` | Robot-style map localization: loads a map saved by the stereo-inertial SLAM (`basalt_euroc_online_slam_demo --save-map`), localizes camera images against it and corrects the VIO odometry into the map frame. Publishes `/tf` `map -> odom` (REP-105), the body pose in the map and diagnostics. |
| `visloc_wheel_calibration_node` | A robot's wheel odometry calibrated while driving, against `visloc_vio_node` and `visloc_map_localize_node`'s localizations: where the wheel odometry is measured relative to `base_link`, the lean of `base_link`, and the map's scale. Publishes the wheel odometry as `base_link`'s pose and `odom -> base_link`, for `visloc_map_localize_node`. |

This crate is **not** a member of the root Cargo workspace (it is listed in
the root `[workspace] exclude` and declares its own `[workspace]`), has its
own `Cargo.lock`, and depends on the workspace crates by path. The root
workspace's builds, `cargo deny`, MSRV and packaging checks never see its
dependency tree.

## Build

```sh
cd ros2/visloc-ros2
cargo build --release            # both node binaries
cargo test                       # unit + core + over-the-wire DDS tests
```

Rust 1.88 or newer (the `ros2-client` floor; the repository pins 1.94 via
`rust-toolchain.toml`). The `dev` profile optimizes dependencies
(`opt-level = 2`) because the Basalt estimator and SIFT are impractically
slow unoptimized; use `--release` for real data.

## `visloc_vio_node`

```sh
visloc_vio_node \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --remap left/image_raw=/cam0/image_raw \
  --remap right/image_raw=/cam1/image_raw \
  --remap imu=/imu0
```

The calibration and config are the same Basalt JSON files
`examples/basalt_euroc_online_slam_demo.rs` uses, parsed by the same
`BasaltCalibration` / `BasaltConfig` code, and the estimator is built the
same way (`BasaltVioEstimatorAdapter::from_config`, with the demo's
urgent-keyframe defaults unless the config sets them). The node runs the
lean VIO path (`process_without_marg_data_no_trace`); the online NFR mapper
of the demo is **not** attached, so loop closure / global BA are not part
of the node's output.

| topic | type | dir | notes |
|---|---|---|---|
| `left/image_raw` | `sensor_msgs/Image` | in | cam0 of the calibration (reference camera) |
| `right/image_raw` | `sensor_msgs/Image` | in | cam1 |
| `imu` | `sensor_msgs/Imu` | in | `angular_velocity` (rad/s) and `linear_acceleration` (m/s²) in the IMU frame of the calibration; orientation ignored |
| `/tf_static` | `tf2_msgs/TFMessage` | in | only with `base_frame`: where the base sits relative to `body_frame` |
| `odom` | `nav_msgs/Odometry` | out | pose of the IMU (body), or of `base_frame`, in a gravity-aligned, z-up `odom` frame; twist in that frame |
| `pose` | `geometry_msgs/PoseStamped` | out | same pose |
| `path` | `nav_msgs/Path` | out | last `path_max_length` poses |
| `/tf` | `tf2_msgs/TFMessage` | out | `odom_frame -> body_frame`, or `odom_frame -> base_frame` |

Output stamps are the left image's header stamps.

Parameters (`--name value`, `--name=value` or `--ros-args -p name:=value`):

| parameter | default | |
|---|---|---|
| `calibration` | required | Basalt calibration JSON |
| `config` | required | Basalt VIO config JSON |
| `odom_frame` / `body_frame` | `odom` / `imu_link` | output frame ids; `body_frame` is the IMU's frame |
| `base_frame` | none | publish this frame's pose instead of the IMU's, e.g. `base_link` (REP-105 `odom -> base_link`), placed by the `/tf_static` transform between `body_frame` and it; nothing is published until that transform arrives |
| `publish_tf`, `publish_path` | `true` | |
| `path_max_length` | `2000` | |
| `stereo_sync_tolerance_ms` | `2.0` | max left/right stamp difference |
| `frame_queue` | `1` | paired frames waiting for the frontend (drop-oldest) |
| `frontend_threads` | `2` | worker threads of the KLT frontend (`0`: share the estimator's pool) |
| `imu_buffer` | `4000` | buffered IMU samples (drop-oldest) |
| `image_qos_depth`, `imu_qos_depth` | `5`, `400` | subscription history depth |
| `urgent_keyframes` | `true` | the demo's urgent-keyframe defaults |
| `imu_seed_klt` | `false` | `config.optical_flow_imu_seed_rotation` |
| `max_iterations` | `3` | LM iterations per frame, overriding `config.vio_max_iterations` (`0` keeps the config's); see the replay results below |
| `domain_id` | `$ROS_DOMAIN_ID` or 0 | |
| `node_name` | `visloc_vio` | also `-r __node:=...`, `-r __ns:=...` |

Data flow and threading: one receive thread per subscription decodes images
(mono8 / mono16 / rgb8 / bgr8 / rgba8 / bgra8; colour is converted with
BT.601 luma; 8-bit is widened `u16 = u8 << 8` exactly like the EuRoC
reader) and hands them to a stereo synchronizer (closest stamp within the
tolerance). Paired frames go into a bounded drop-oldest queue. A frame is
released only once IMU data reaches its stamp, with the half-open IMU
interval `(previous frame, this frame]`, which is the `EurocSensorFrame`
contract. As in upstream Basalt, KLT tracking and the estimator run on two
threads: the frontend thread tracks frame N+1 (on its own pool of
`frontend_threads` workers) while the estimator thread solves frame N and
publishes. The frontend hands each tracked frame over directly, so it is at
most one frame ahead. If the estimator falls behind, the queued frames are
dropped, oldest first, before tracking. The newest frame is processed next
and latency stays bounded. The IMU interval of the next processed frame
still starts at the last processed frame, so preintegration stays
contiguous. The two-thread split gives the same estimates as running both
stages in turn (`tests/core_pipeline.rs` checks this). The `imu_seed_klt`
seed is the exception: it uses the latest gyro bias the estimator has
published, not the previous frame's. `cam_time_offset_ns` from the
calibration is added to camera stamps. On an estimator error the node logs
it, rebuilds both stages and re-initializes from IMU. Every 10 s it logs
the dropped-frame counts and the frontend/estimator time per frame.

## `visloc_localize_node`

```sh
visloc_localize_node --map <colmap_model_dir> \
  --descriptors <landmark_descriptors.txt> \
  --remap image=/camera/image_raw --remap camera_info=/camera/camera_info
```

The map is a COLMAP model directory (`cameras/images/points3D` in `.bin`,
or `.txt` if no `cameras.bin` exists) read with `visloc-io`. The descriptor
store has one `<landmark_id> <float> <float> ...` line per landmark, as
written by `scripts/export_openloris_localization_map.py`. Localization
follows `examples/localize_openloris_map.rs`
(`LocalizationPipeline<BruteForceMatcher, AllLandmarksSelector, PnPRansac>`
with a minimum-inlier floor), except that the query features are extracted
in the node with the repository's pure-Rust SIFT.

**The map descriptors must come from a compatible extractor** (128-D SIFT;
choose L2 or RootSIFT normalization with `sift_root`). Descriptors from a
different extractor or normalization still "work" mechanically but match
poorly.

| topic | type | dir | notes |
|---|---|---|---|
| `image` | `sensor_msgs/Image` | in | raw (unrectified) image, same encodings as above |
| `camera_info` | `sensor_msgs/CameraInfo` | in, optional | `K`/`D` override the map camera; `plumb_bob`, `rational_polynomial`, `equidistant` |
| `pose` | `geometry_msgs/PoseWithCovarianceStamped` | out | camera (optical frame) pose in `map_frame`, only on success |
| `inlier_count` | `std_msgs/Int32` | out | PnP inliers, every frame (0 on failure) |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | out | success, inliers, matches, features, reprojection error, time |

| parameter | default | |
|---|---|---|
| `map` | required | COLMAP model directory |
| `descriptors` | none | descriptor store (otherwise descriptors embedded in the map) |
| `camera_id` | smallest id | map camera used without `camera_info` |
| `use_camera_info` | `true` | |
| `map_frame` | `map` | |
| `min_inliers` | `12` | |
| `ransac_iterations`, `reprojection_threshold`, `ratio` | `128`, `4.0`, `0.8` | `ratio <= 0` disables the ratio test |
| `sift_max_keypoints`, `sift_root` | `4000`, `false` | |
| `position_stddev`, `orientation_stddev` | `0.1` m, `0.05` rad | written to the covariance diagonal (PnP gives no calibrated covariance) |
| `image_queue` | `1` | images waiting while one is localized (drop-oldest: always the latest) |
| `domain_id`, `node_name` | | as above |

Images whose size differs from the camera are rejected (reported as a
failure), not rescaled.

## `visloc_map_localize_node`

Localization against a map built and saved by the repository's
stereo-inertial SLAM, the way a robot localizes in a prebuilt map: run it
next to `visloc_vio_node`, and it publishes the `map -> odom` transform that
places the VIO's drifting `odom` frame in the map (REP-105). Together with
the VIO's `odom -> imu_link`, every frame has a map pose.

```sh
# Once: map the place and save the map (repository root)
cargo run --release --example basalt_euroc_online_slam_demo -- \
  --euroc-dir /data/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir runs/mh01_slam --save-map runs/mh01_map

# On the robot
visloc_vio_node --calibration .../euroc_ds_calib.json --config .../euroc_config.json \
  --remap left/image_raw=/cam0/image_raw --remap right/image_raw=/cam1/image_raw --remap imu=/imu0
visloc_map_localize_node --map runs/mh01_map --remap image=/cam0/image_raw
```

Every `localize_every`-th image goes through the SLAM mapper's own ORB
frontend (every `initial_localize_every`-th, by default every image, until
the odometry has a map pose, which needs two agreeing fixes), is matched to
the map's landmarks by Hamming distance and solved with P3P RANSAC; a pose
with at least `min_inliers` inliers is a *fix*. Until the odometry has a
map pose, every keypoint is compared with every landmark. After that, the
correction and the odometry at the image's stamp predict the pose (the node
waits up to `prior_wait_ms` for that odometry). Each landmark is projected
with the predicted pose and compared only with the keypoints within
`projection_radius` pixels. If that does not localize, the node falls back
to the map-wide search. On EuRoC this cuts matching and PnP from about 50 ms
to 4 ms per image, and doubles the inliers (169 vs 81 on MH_02 against the
MH_01 map). Both searches pack the map's descriptors once
(`visloc_localization::BinaryLandmarkIndex`).
Each fix is paired with the odometry pose of the same stamp (the VIO node
stamps its odometry with the left image's stamp; a fix that arrives before
its odometry waits for it, and a stamp between two odometry poses at most
0.2 s apart is interpolated) and fed to `visloc_localization::OdomCorrection`,
which keeps the odometry-to-map transform:

- the transform is a yaw and a translation (both frames are gravity-aligned),
  averaged over the fixes of the last 5 s;
- it starts once two consecutive fixes agree, or three when one of them is
  a weak fix (`weak_min_inliers`); weak fixes are not used after that,
  unless `weak_gate_m` is set;
- it accepts a fix within 15 cm / 3 deg of where it puts the odometry pose,
  widened by 5 cm / 1 deg per metre travelled since the last accepted fix,
  along the path of every odometry message;
  three agreeing rejected fixes in a row replace it (a VIO jump).

Between fixes the VIO carries the pose, so its error grows with the distance
since the last fix. The published covariance follows that
(`position_stddev + position_stddev_per_m * metres_since_fix`, fitted to
EuRoC), and `/diagnostics` reports `seconds_since_fix` and
`metres_since_fix`. See `docs/basalt_map_localization.md` for the
measurements behind the defaults.

| topic | type | dir | notes |
|---|---|---|---|
| `image` | `sensor_msgs/Image` | in | raw images of the map's `camera` (cam0 by default), the same stream the VIO uses |
| `odom` | `nav_msgs/Odometry` | in | gravity-aligned odometry of a body frame, e.g. `visloc_vio_node`'s or a ground robot's wheel odometry; its `header.frame_id` names the odometry frame and its `child_frame_id` the body |
| `/tf_static` | `tf2_msgs/TFMessage` | in | only with `extrinsics tf`: where the camera sits relative to the body |
| `/tf` | `tf2_msgs/TFMessage` | out | `map_frame -> <odometry frame>`, stamped with each odometry message once the correction is set |
| `map_pose` | `geometry_msgs/PoseWithCovarianceStamped` | out | body pose in `map_frame` for each odometry message once the correction is set (not `pose`, which `visloc_vio_node` publishes) |
| `map_fix` | `geometry_msgs/PoseWithCovarianceStamped` | out | body pose of each image localized with `min_inliers`, stamped with the image (for `visloc_wheel_calibration_node`) |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | out | per localization attempt: localized, inliers, matches, matching (`projection` / `global`), time, `map_pose` (whether the odometry has one yet), seconds/metres since the last fix, pending fixes |
| `map_cloud` | `sensor_msgs/PointCloud2` | out | the map's landmarks in `map_frame`, published once at start, transient-local (a late subscriber such as RViz still gets it) |

| parameter | default | |
|---|---|---|
| `map` | required | directory written by `--save-map` (`map.json`, `sparse/`, `landmark_descriptors.txt`), or a map merged from several flights by `basalt_map_merge` (see `docs/basalt_map_localization.md`); a map built with `--equalize-local-contrast` or `--smooth-features` makes the node equalize or blur every image the same way before extraction |
| `camera` | `0` | Basalt camera index of `image` |
| `map_frame` | `map` | |
| `extrinsics` | `map` | where the camera sits relative to the odometry's body: `map` takes the map's IMU-from-camera (`map.json`; the body is the map's IMU, as with `visloc_vio_node`'s default output); `tf` looks up the `/tf_static` transform from the camera's frame to the odometry's `child_frame_id`, e.g. `base_link` (see [on a ground robot](#on-a-ground-robot)) |
| `camera_frame` | the image's `frame_id` | with `extrinsics tf`, the camera's frame |
| `publish_tf` | `true` | |
| `publish_map_cloud` | `true` | publish `map_cloud` |
| `localize_every` | `5` | localize every N-th received image |
| `initial_localize_every` | `1` | ... and every N-th until the first map pose; while localizing is slower than the camera, the node takes the newest image whenever it is free |
| `initial_window_s` | `30` | ... only within this many seconds of the first image or of the latest localization; after that every `localize_every`-th until an image localizes |
| `min_inliers` | `30` | below 30, wrong poses appear on EuRoC |
| `weak_min_inliers` | `15` | a map-wide localization with at least this many inliers, but fewer than `min_inliers`, is a *weak fix*: it counts only toward the first map pose, which then needs three agreeing fixes instead of two (`>= min_inliers`: off) |
| `weak_gate_m` | `0` (off) | once the odometry has a map pose, a weak localization (also one matched by projection) within this many metres and 2 deg of the corrected odometry is used as a fix too; helps where images rarely reach `min_inliers` (OpenLORIS-Scene office1-7: 6.9 to 3.4 cm), costs a little where they often do (EuRoC: up to +0.5 cm) |
| `max_hamming`, `ratio` | `70`, `0.8` | ORB match gates (`ratio <= 0` disables) |
| `ransac_iterations`, `reprojection_threshold` | `1000`, `3.0` | P3P RANSAC |
| `projection_radius` | `20` px | once the odometry has a map pose, search radius around each landmark's predicted projection (`<= 0`: always map-wide) |
| `projection_fresh_s`, `projection_fresh_m` | `5` s, `2` m | the prediction is used only this long and this far (odometry travel) after the correction's newest fix; after a longer gap the image is matched map-wide, since a drifted prediction can steer the projection search to a wrong pose that agrees with it |
| `prior_wait_ms` | `300` | longest wait for the odometry of an image's stamp before searching map-wide |
| `match_threads` | `1` | threads of the map-wide search |
| `duplicate_radius` | `0` (off) | map-wide search: landmarks within this many metres of the best one count as the same point in the ratio test; helps merged maps (see `docs/basalt_map_localization.md`) |
| `extra_descriptors` | `0` | with a map saved by `basalt_euroc_online_slam_demo --extra-descriptors`: match each landmark at the nearest of its main descriptor and up to this many of its other views' (finds more fixes, but less accurate ones on EuRoC; see `docs/basalt_map_localization.md`) |
| `max_points` | the map's | ORB keypoints per image |
| `position_stddev`, `position_stddev_per_m` | `0.03` m, `0.025` m/m | published position 1-sigma |
| `orientation_stddev` | `0.02` rad | |
| `image_queue` | `1` | images waiting while one is localized (drop-oldest) |
| `domain_id`, `node_name` | | as above |

**EuRoC, replayed over DDS.** A RustDDS player published MH_02's stereo
images and IMU to `visloc_vio_node` and `visloc_map_localize_node` (map
built from MH_01), all three processes on one 4-core Xeon. Scored against
ground truth as in `docs/basalt_map_localization.md`. Latency runs from the
player publishing a frame to its `map_pose` arriving back:

| replay speed | VIO frames processed | `map_pose` published | RMSE / max | latency p50 / p90 / max |
|---|---|---|---|---|
| 0.5x | 3,040 / 3,040 | 99.8% of frames | 2.06 / 5.2 cm | 54 / 72 / 137 ms |
| 1.0x | 3,035 / 3,040 (5 dropped behind) | 99.6% of frames | 2.05 / 5.2 cm | 49 / 90 / 195 ms |
| 1.0x, before the changes below | 2,743 / 3,040 (297 dropped behind) | 90.0% of frames | 2.02 / 5.9 cm | 171 / 271 / 339 ms |

Three runs at 1.0x dropped 1, 5 and 14 frames. The offline
`basalt_euroc_localize_demo --vio --localize-every 5 --projection-radius-px 0` gives 99.8% at 2.05 /
5.7 cm on the same data. These changes made the real-time run possible on
4 cores:

- The VIO frontend and estimator run on separate threads (see above).
- The estimator is about 26% faster with a bit-identical trajectory: two
  serial loops of its LM iteration (the linearization and the model cost
  decrease) now run per landmark in parallel.
- The KLT frontend is about 38% faster, also bit-identical (12.3 to 7.6 ms
  per frame).
- The node runs 3 LM iterations per frame instead of the config's 5
  (`max_iterations`). On the eight EuRoC sequences in
  `docs/basalt_map_localization.md`, the VIO's ATE changed by at most
  0.13 cm. The estimator got 15-30% faster.
- Localization uses the projection search once the odometry has a map pose.

The remaining drops are in MH_02's busiest stretch, where the estimator
still takes up to about 100 ms on a frame while sharing the cores with the
player and the map node. There the node drops frames instead of falling
behind. A dropped frame gets no pose, and drops did not change the VIO's
accuracy. With the VIO node alone at 0.3x, the latency is about 47 ms: the
frontend, the estimator and the DDS transport.

## Launch file, URDF and RViz

`launch/map_localization.launch.py` starts the whole stack:

- `robot_state_publisher` with a URDF;
- `visloc_vio_node`, publishing `odom -> base_link`;
- with `wheel_odom_topic`, `visloc_wheel_calibration_node`: the robot's
  wheel odometry, calibrated against the VIO and the map's localizations,
  becomes `odom_topic` and `odom -> base_link` (the VIO then only feeds
  it, on `vio_odom_topic`);
- `visloc_map_localize_node`, with the camera extrinsics from `/tf_static`;
- RViz (`rviz:=true`) with `launch/rviz/map_localization.rviz`. It shows
  the map's landmarks, the VIO path, the map pose with its covariance, the
  TF tree and the camera image, and follows `base_link`.

The nodes are plain binaries, not a colcon package, so `bin_dir` defaults to
this crate's `target/release`.

```sh
source /opt/ros/jazzy/setup.bash
ros2 launch ros2/visloc-ros2/launch/map_localization.launch.py map:=runs/mh01_map rviz:=true
ros2 bag play mh02_bag --delay 3      # or the robot's drivers
```

On a wheeled robot, with the wheel driver's own `odom -> base_link` turned
off (e.g. `enable_odom_tf: false` for `diff_drive_controller`):

```sh
ros2 launch ros2/visloc-ros2/launch/map_localization.launch.py map:=office_map \
  urdf:=robot.urdf wheel_odom_topic:=/wheel/odom \
  wheel_calibration_file:=$HOME/.ros/robot_wheel_calibration.json
```

The first launch calibrates while the robot drives; turns in place and
straight runs show the offset and the scale soonest. Each estimate is
saved to `wheel_calibration_file`, and later launches start from it.

| argument | default | |
|---|---|---|
| `map` | required | saved or merged map |
| `calibration`, `config` | the EuRoC files | Basalt calibration and VIO config |
| `urdf` | `launch/euroc_rig.urdf` | robot description for `robot_state_publisher` |
| `left_topic`, `right_topic`, `imu_topic` | `/cam0/image_raw`, `/cam1/image_raw`, `/imu0` | sensor topics; the left camera is the map's |
| `imu_frame`, `base_frame` | `imu0`, `base_link` | the IMU's and the base's links in the URDF |
| `use_vio` | `true` | `false`: another node (e.g. wheel odometry) publishes `odom_topic` |
| `odom_topic` | `/odom` | the odometry the map localization corrects into the map frame |
| `wheel_odom_topic` | empty | the robot's wheel odometry, to calibrate and use as `odom_topic` |
| `vio_odom_topic` | `/vio/odom` | with `wheel_odom_topic`, where the VIO publishes |
| `wheel_calibration_file` | empty | with `wheel_odom_topic`, where the calibration is kept between launches |
| `bin_dir` | `ros2/visloc-ros2/target/release` | where the node binaries are |
| `rviz`, `rviz_config` | `false`, `launch/rviz/map_localization.rviz` | |

`tools/calib_to_urdf.py` writes a URDF of a Basalt rig: `base_link`, the
IMU, and each camera's optical frame from the calibration's `T_imu_cam`.
`launch/euroc_rig.urdf` is the EuRoC rig made with it, with `base_link` at
the IMU. On a robot, use the robot's own URDF, with the IMU and camera
links named as the IMU and image messages' `frame_id`s.

Tested with ROS 2 Jazzy on EuRoC MH_02 against the MH_01 map, played at
1.0x. Through the launch file, 99.6% of frames get a map pose at 2.06 /
5.2 cm RMSE / max, the same as the nodes started by hand. With RViz
rendering in software on the same 4 cores (under Xvfb), the VIO dropped 11
of 3,040 frames:

![RViz: the MH_01 map's landmarks, the VIO path and the map pose during MH_02](../../docs/assets/ros2_map_localization_rviz.png)

With `wheel_odom_topic`, on OpenLORIS-Scene office1-6 against map A (the
bag at real time; a URDF from the dataset's `base_link -> imu0` and
calibration, written by `tools/calib_to_urdf.py --base-from-imu`), the
first launch gave 5.49 / 9.5 cm and a second one, starting from the saved
calibration, 5.11 / 9.4 cm. `/tf` carried `map -> odom` from the map
localization and `odom -> base_link` from the wheel calibration only.

## Running inside a ROS 2 graph

* **Discovery / domain**: standard SPDP multicast discovery on
  `ROS_DOMAIN_ID` (or `--domain-id`). Domain ids above ~230 overflow the
  RTPS port mapping, as in any DDS. RustDDS does not implement Fast DDS's
  *discovery server* or Cyclone's unicast-only peer lists. If your network
  blocks multicast, these nodes will not be discovered.
* **RMW interoperability**: RustDDS speaks standard RTPS 2.x with CDR
  (`CDR_LE`) payloads and ROS 2's topic/type mangling (`rt/<topic>`,
  `sensor_msgs::msg::dds_::Image_`). That is what `rmw_fastrtps_cpp` (the
  default in Humble/Jazzy) and `rmw_cyclonedds_cpp` use. Both are tested
  against ROS 2 Jazzy, see [below](#tested-against-ros-2-jazzy). `rmw_zenoh`
  is a different protocol and is not supported by these binaries.
* **Topic names** are resolved like `rclcpp` does: a relative name goes
  under the node namespace (`odom` is `/odom`, or `/robot1/odom` with
  `-r __ns:=/robot1`), and an absolute one (`/tf`) is kept. The nodes
  resolve names before handing them to `ros2-client` 0.11, whose own
  resolution would publish `rt//odom` in the root namespace, a topic only
  other `ros2-client` nodes see.
* **Startup**: discovery between these nodes and a newly started ROS 2
  publisher takes about 1.5 s, and samples published before it completes
  are lost (volatile QoS). Start the nodes first and give a bag player a
  head start, e.g. `ros2 bag play --delay 3`.
* **Message layouts** are hand-written serde mirrors of the ROS IDL
  (`src/msgs.rs`), and they match the Humble through Kilted definitions. Unit
  tests pin the CDR bytes against hand-assembled references (alignment,
  string NUL terminators, `uint8[]` vs fixed arrays).
* **QoS**: subscriptions are best-effort / volatile, which matches both
  `SensorDataQoS` drivers and reliable publishers. Publishers are reliable
  / volatile keep-last (the rclcpp default `QoS(10)`), which RViz and
  `ros2 topic echo` match with either reliability. Note that `/tf` is
  published volatile, the same as `tf2_ros::TransformBroadcaster`.
* **Large images** are fragmented over UDP (RTPS `DATA_FRAG`). Fast DDS
  uses shared memory only between Fast DDS peers and falls back to UDP for
  RustDDS, so expect UDP-loopback costs for full-resolution streams.
* **Type hashes (Jazzy+)**: `ros2-client` is built with its default
  `jazzy` feature. Type-description hashes are not computed for these
  hand-written messages, so `ros2 topic info -v` may show an empty hash.
  Matching is by type name, so this does not affect data flow.
* Parameters are command-line only. There are no runtime parameter
  services, no `--params-file`, and no lifecycle node. Security (SROS2)
  is off.

## Tested against ROS 2 Jazzy

On Ubuntu 24.04 with ROS 2 Jazzy from `packages.ros.org` (`rmw_fastrtps_cpp`
8.4 / Fast DDS 2.14, and `rmw_cyclonedds_cpp` 2.2 / Cyclone DDS 0.10), on
one host, with both RMWs:

* `ros2 topic list -t` / `ros2 node list` show the nodes and their topics
  with the right names and types, also in a namespace.
* `ros2 topic echo` decodes `odom`, `map_pose` and `/diagnostics`, and
  `ros2 topic hz` sees full-rate `odom` and 752x480 images.
* `ros2 run tf2_ros tf2_echo map imu_link` resolves the chain `map -> odom`
  (from `visloc_map_localize_node`) and `odom -> imu_link` (from
  `visloc_vio_node`).
* The nodes run on a rosbag2: EuRoC MH_02 converted with
  `tools/euroc_to_rosbag2.py` and played with `ros2 bag play`, with the
  outputs recorded by `ros2 bag record` and scored with
  `tools/bag_to_tum.py`. The bag publishers can be reliable (the
  default) or best-effort (`--qos-profile-overrides-path`).

| `ros2 bag play` MH_02 at 1.0x | `map_pose` published | RMSE / max |
|---|---|---|
| Fast DDS | 99.7% of frames | 2.06 / 5.2 cm |
| Cyclone DDS | 98.7% of frames | 2.06 / 6.0 cm |

These are the same accuracy as the RustDDS replay above. The Fast DDS run
used `--delay 3`; the Cyclone run had none and lost the first 1.6 s of
images to discovery. To reproduce:

```sh
source /opt/ros/jazzy/setup.bash
python3 tools/euroc_to_rosbag2.py /data/MH_02_easy mh02_bag     # needs Pillow
# a robot's wheel odometry too (TUM, seconds) on /wheel/odom, odom -> base_link:
#   python3 tools/euroc_to_rosbag2.py data/office1-7 o17_bag --odometry data/office1-7/odom.tum
visloc_vio_node --calibration .../euroc_ds_calib.json --config .../euroc_config.json \
  --ros-args -r left/image_raw:=/cam0/image_raw -r right/image_raw:=/cam1/image_raw -r imu:=/imu0 &
visloc_map_localize_node --map runs/mh01_map --ros-args -r image:=/cam0/image_raw &
ros2 bag record -o out /odom /map_pose /tf &
ros2 bag play mh02_bag --delay 3
python3 tools/bag_to_tum.py out /map_pose map_pose.tum
```

**All eight EuRoC queries.** The same setup (Fast DDS, 1.0x, `--delay 3`)
on every query sequence of `docs/basalt_map_localization.md`, each against
its room's map, next to the offline `basalt_euroc_localize_demo --vio
--localize-every 5` with its defaults (projection matching, every image
until the first map pose):

| query → map | VIO frames dropped | `map_pose` published (offline) | RMSE / max (offline) | first map pose |
|---|---|---|---|---|
| MH_02 → MH_01 | 0 | 99.8% (99.97%) | 2.1 / 5.2 cm (2.0 / 5.6) | 0.3 s |
| MH_03 → MH_01 | 10 | 99.5% (99.96%) | 4.3 / 13.2 cm (4.5 / 10.6) | 0.2 s |
| MH_04 → MH_01 | 1 | 73.3% (73.6%) | 21.5 / 40.0 cm (22.7 / 44.6) | 27.0 s |
| MH_05 → MH_01 | 0 | 78.1% (78.2%) | 17.9 / 30.2 cm (15.7 / 27.9) | 24.9 s |
| V1_02 → V1_01 | 15 | 94.2% (94.3%) | 3.3 / 5.9 cm (3.4 / 5.9) | 5.0 s |
| V1_03 → V1_01 | 7 | 99.6% (99.95%) | 4.7 / 11.0 cm (4.6 / 9.6) | 0.1 s |
| V2_02 → V2_01 | 0 | 99.9% (99.96%) | 3.6 / 14.1 cm (3.5 / 12.0) | 0.1 s |
| V2_03 → V2_01 | 4 | 77.5% (77.8%) | 27.1 / 43.1 cm (29.6 / 70.4) | 21.4 s |

No run reset the VIO, and none dropped more than 0.9% of its frames, also
while the node tried every image before the first map pose (MH_04 and
MH_05 for 25-27 s). The node's results follow the offline ones, also where
those are poor:

- MH_04 and MH_05 fly through parts of the hall the MH_01 map never saw.
  The first map pose comes only after 25-27 s, and the VIO carries the pose
  through long stretches without a fix. On MH_05 two fixes after such a
  stretch were rejected and the third reset the correction.
- V2_03 localized 8 of the 700 images the node tried. That is enough for a
  map pose from 21 s on (with every 5th image from the start, it never got
  one), but the VIO carries the pose most of the time: 27 cm RMSE.
- MH_03 has no 23 cm pose any more. In an earlier replay, before the node
  stopped trusting its prediction 5 s or 2 m after the newest fix
  (`projection_fresh_s`, `projection_fresh_m`), the first fix after a
  27 s gap was matched around the 27 s old prediction, with 30 inliers,
  and accepted. The next two, with 85 and 110 inliers, were rejected and
  the third reset the correction; the worst pose was 23 cm off. Now that
  image is matched map-wide, no fix is rejected and the worst pose is
  13.2 cm off.

Read the bag once before playing it (`cat bag/*.mcap > /dev/null`), as
every run above did. Played straight after a reboot, MH_02 lost 1,106 of
its 3,040 frames: reads from the cold disk made the playback bursty, and
the VIO's one-frame queue dropped the bursts. An MH_05 run with the test
suite compiling on the same cores dropped 194 frames; the run above dropped
none.

Not covered: Humble, several hosts on a real network, and live sensors. RViz
was run under Xvfb only ([launch file](#launch-file-urdf-and-rviz)).

## On a ground robot

A ground robot's frames follow REP-105: `map -> odom -> base_link`, with
each sensor fixed to `base_link` on `/tf_static` (from
`robot_state_publisher` and the URDF, or `static_transform_publisher`).
Both nodes fit into that tree:

- `visloc_vio_node -p body_frame:=<IMU frame> -p base_frame:=base_link`
  publishes `odom -> base_link`: the odometry's `child_frame_id` is
  `base_link` and its twist is `base_link`'s, from the `/tf_static`
  transform between the IMU's frame and `base_link`. Without
  `base_frame`, the node publishes `odom -> imu_link`. On a robot that
  gives the IMU frame a second parent next to `base_link -> imu_link`,
  which tf2 does not allow.
- `visloc_map_localize_node -p extrinsics:=tf` takes the camera's position
  from `/tf_static` (from the image's `frame_id` to the odometry's
  `child_frame_id`) instead of the map's IMU-from-camera calibration. The
  odometry's body can then be any frame. Wheel odometry works as well as
  the VIO: on a level floor its `odom` frame is gravity-aligned (z up),
  and the correction only estimates a yaw and a translation.

The map is still built by the stereo-inertial SLAM, and the localized camera
must be the map's camera.

Tested with ROS 2 Jazzy on EuRoC MH_02 against the MH_01 map, played at
1.0x. Two `static_transform_publisher`s provided `base_link -> imu0` (the
IMU 10 cm ahead of, 5 cm to the right of and 25 cm above the base, yawed
90°) and `imu0 -> cam0` (the calibration). `map_pose` is then `base_link`'s
pose; it is scored after moving it back to the IMU:

| odometry | `map_pose` published | RMSE / max | the odometry alone |
|---|---|---|---|
| `visloc_vio_node`, `base_frame:=base_link` | 99.5% of frames | 2.21 / 6.6 cm | 4.4 cm |
| simulated wheel odometry, 20 Hz | 98.3% of frames | 3.45 / 7.8 cm | 98.7 cm |

"The odometry alone" is its ATE after a rigid fit to ground truth. Both
nodes received the transforms from rclcpp's transient-local `/tf_static`
publishers. In the VIO run, `tf2_echo map imu0` resolved
`map -> odom -> base_link -> imu0`. The wheel odometry is simulated
(`base_link`'s ground-truth path with a 2% scale error and a yaw error
growing by 0.5° per metre), because no wheeled-robot data with a
stereo-inertial map was at hand. Real wheel odometry on an uneven floor
also tilts, which this does not model.

**A real robot: OpenLORIS-Scene.** Its sessions were played from rosbag2s
with the robot's own wheel odometry (`tools/euroc_to_rosbag2.py --odometry`)
into the nodes at real time, against maps of other sessions
([details](../../docs/openloris_map_localization.md#on-ros-2)):

| odometry | office1-6 → map A | office1-7 → map B |
|---|---|---|
| `visloc_vio_node`, `base_frame:=base_link` | 1.51 / 2.7 cm | 2.22 / 5.0 cm |
| wheel odometry on `/wheel/odom`, 20 Hz | 2.47 / 4.4 cm | 3.25 / 8.8 cm |

(RMSE / max of `map_pose`, `extrinsics:=tf`, `weak_gate_m:=0.05`.) Both
reproduce the offline demo within 0.4 cm. The wheel odometry needs
`base_link` where it reports it: `scripts/calibrate_wheel_offset.py` fits
that point and the level of the frame from a VIO run. As shipped, the
dataset's `base_link` is 5–8 cm from that point and leans 1.9°, with the
camera 0.9 m above it; office1-7's wheel odometry was 4.67 cm with the
point corrected but not the lean.

## `visloc_wheel_calibration_node`

Wheel odometry reports the point the robot turns about, in a level frame, at
the wheels' scale. A robot's description may put `base_link` centimetres from
that point and a degree or two off level; with the camera high above the
wheels, either moves the camera pose the wheel odometry implies by
centimetres in every turn. The map, built by a VIO, may also be a few
percent off the wheels' scale.

The node fits, in the plane, `base_link`'s path in a reference against the
wheel odometry's: the offset in the body frame, a rotation, a translation
and a scale (`visloc_localization::WheelCalibration`). Two references feed
it, and each gives what it measures best:

- **`visloc_vio_node`'s odometry** (paired within 20 ms, through the
  `/tf_static` transform between the VIO's frame and `base_link`) follows
  every turn, so it gives the offset, once the wheel odometry has turned
  `min_turn_deg` (180) and travelled `min_travel_m` (1 m): turns in place
  show the offset, and travel tells it from the scale.
- **`visloc_map_localize_node`'s localizations** (`map_fix`, each paired
  with the wheel odometry interpolated to its image) come only where the
  camera sees the map, rarely through a turn, but they carry the map's
  scale and gravity, which is what the map localization needs. From
  `min_fixes` (20) of them it takes `base_link`'s lean; once they span
  `min_travel_m`, the scale, and without the VIO's offset its own estimate
  of the offset, held within about 3 cm of the saved one (or of zero) where
  their turns do not show it. Localizations more than 10 cm from the fit
  through the others are left out.

A VIO session's scale is not the map's (it varied by up to 3.6% between
OpenLORIS-Scene sessions), so the node reports it but does not use it.
Until a reference shows a value, it keeps the saved calibration's, or none.

It refits every `refine_s` (5) seconds over the last `window_s` (120)
seconds of pairs, giving each `segment_s` (5) seconds of VIO pairs, and
each `fix_segment_s` (30) seconds of map localizations, their own heading
against the reference: the wheels slip in turns, and their heading drifts
by tens of degrees over minutes. It publishes each wheel odometry
message as `base_link`'s pose, levelled and at the map's scale, with a yaw
and a translation that keep the output continuous wherever a new estimate
takes over. Before the
first estimate it passes the wheel odometry through. With
`calibration_file`, a saved calibration applies from the first message, and
each new estimate is written there. It also publishes the same pose on
`/tf`, so that `map -> odom -> base_link` agrees with `map_pose`: turn the
wheel driver's own `odom -> base_link` off (as for `robot_localization`).
`scripts/calibrate_wheel_offset.py` fits the VIO's part offline, and
`examples/wheel_calibration_replay.rs` runs the node's calibration on
recorded wheel odometry, VIO and map localizations (TUM files) without ROS.

```sh
visloc_vio_node ... -p body_frame:=imu0 -p publish_tf:=false -r odom:=/vio/odom
visloc_wheel_calibration_node -p calibration_file:=wheel.json -r imu:=/imu0 \
  -r wheel_odom:=/wheel/odom -r vio_odom:=/vio/odom -r odom:=/wheel/odom_calibrated
visloc_map_localize_node --map ... -p extrinsics:=tf -r odom:=/wheel/odom_calibrated
```

Without a VIO, leave `visloc_vio_node` out: the map's localizations alone
give the lean, the scale and the offset. The [launch file](#launch-file-urdf-and-rviz)
starts all three with `wheel_odom_topic`.

Wheels slip in turns in place, so the wheel odometry's heading drifts with
every turn. Given an IMU on `imu` (any IMU on the robot, mounted any way;
here the VIO's), the node keeps the wheels' travel but turns it by the
heading the gyro integrates about the vertical, which the accelerometer
gives, and takes the gyro's bias wherever the wheels stand still. On
OpenLORIS-Scene's corridor (284 s, 232 m) the heading then stays within
5.6° of the ground truth instead of 44°, and 60 s after a known pose the
odometry is at most 1.4 m off instead of 16 m
([details](../../docs/openloris_map_localization.md#the-wheels-heading-from-a-gyro)).

| topic | type | dir | notes |
|---|---|---|---|
| `wheel_odom` | `nav_msgs/Odometry` | in | the wheel odometry; its `child_frame_id` is the body (`base_link`) |
| `imu` | `sensor_msgs/Imu` | in | optional: any IMU on the robot, in any orientation; the wheel odometry's heading comes from its gyro (`gyro_heading`) |
| `vio_odom` | `nav_msgs/Odometry` | in | optional: `visloc_vio_node`'s odometry (or `base_link`'s, with `base_frame`) |
| `map_fix` | `geometry_msgs/PoseWithCovarianceStamped` | in | optional: `visloc_map_localize_node`'s localizations of `base_link` (`extrinsics:=tf`) |
| `/tf_static` | `tf2_msgs/TFMessage` | in | the VIO's frame relative to the body, unless both are the same |
| `odom` | `nav_msgs/Odometry` | out | `base_link`'s pose, in the wheel odometry's frame and the map's scale; twist and covariances as received |
| `/tf` | `tf2_msgs/TFMessage` | out | the same pose, wheel odometry frame `-> base_link` (`publish_tf`) |

| parameter | default | |
|---|---|---|
| `calibration_file` | none | JSON calibration: applied from the start if it exists, written at each new estimate |
| `min_turn_deg`, `min_travel_m`, `min_pairs` | `180`, `1.0`, `200` | the VIO's offset once the wheel odometry has turned and travelled this much over this many pairs; the map's scale once its localizations span `min_travel_m` |
| `min_fixes` | `20` | map localizations for the lean (and the scale) |
| `refine_s` | `5` | refit this often (s); `0` keeps the first estimate, or the saved one |
| `window_s` | `120` | fit over this many seconds of pairs; `0` keeps them all |
| `segment_s`, `fix_segment_s` | `5`, `30` | each stretch of this many seconds of VIO pairs, or of map localizations, gets its own heading against them, so the wheels' heading drift stays out of the fit; `0`: one for all |
| `level` | `true` | level `base_link` against gravity (assumes a level floor) |
| `gyro_heading` | `true` | with `imu`, turn the wheels' travel by the gyro's heading instead of the wheels' own |
| `apply_scale` | `true` | publish at the map's scale |
| `publish_uncalibrated` | `true` | pass the wheel odometry through until calibrated |
| `publish_tf` | `true` | publish the output on `/tf` (turn the wheel driver's off) |

On OpenLORIS-Scene (both rosbag2s at real time, maps as shipped, the
dataset's own `base_link -> imu0`;
[details](../../docs/openloris_map_localization.md#calibrating-the-wheel-odometry-online)),
RMSE / max of `map_pose`:

| wheel odometry into `visloc_map_localize_node` | office1-6 → map A | office1-7 → map B |
|---|---|---|
| as is | 8.01 / 14.0 cm | 12.10 / 22.8 cm |
| calibrated in the run, map localizations only | **5.49 / 9.5 cm** | 10.89 / 21.6 cm |
| ... with the VIO too | 5.46 / 9.4 cm | 10.88 / 21.5 cm |
| office1-3's offline calibration, saved | 4.92 / 8.3 cm | 4.66 / 10.4 cm |
| ... and refined in the run (map; VIO and map) | 5.01 / 9.3; 5.16 / 9.7 cm | 4.66 / 10.5; 4.89 / 10.9 cm |

Without a VIO or a saved calibration, office1-6's own localizations take
the error from 8.01 to 5.49 cm. office1-7's 24 localizations span 0.5 m,
enough for the lean only, and the VIO turns 180° only at the end of the
36 s session. A saved calibration from a drive with turns and straight
runs stays the best start; these sessions are too short for refining to
improve on it.

## Tests

```sh
cargo test                 # everything below
cargo test --lib           # message CDR layouts, image encodings, stereo/IMU sync, params, queue
cargo test --test core_pipeline   # cores end to end, no DDS
cargo test --test dds_nodes       # node binaries over real RTPS/DDS
```

* `core_pipeline` drives the transport-agnostic cores synchronously. It
  feeds 30 synthetic 752x480 stereo frames plus 200 Hz static IMU through
  `VioInputs` and `VioCore` with the EuRoC calibration/config, and checks
  every frame yields a pose, the drift stays under 0.5 m and gravity
  alignment is correct. It also localizes a rendered query against a
  synthetic three-depth-slab COLMAP map (written to disk and read back) and
  recovers the 0.16 m camera translation within 2 cm.
* `dds_nodes` spawns the real `visloc_vio_node` / `visloc_localize_node` /
  `visloc_map_localize_node` binaries on a private domain id, and drives
  them from a `ros2-client` test node with synthetic images, IMU and
  odometry. It asserts that `odom`/`pose`, `pose`/`inlier_count`/
  `/diagnostics`, and `/tf` (`map -> odom`) / `map_pose` arrive with sane
  content; the last against a saved map written from the ORB corners of a
  synthetic image, with a stationary odometry pose to recover, once with
  the map's extrinsics and once with `extrinsics tf` and a `base_link`
  camera offset published once on a transient-local `/tf_static`. These
  tests need UDP multicast on loopback. They pass in an Ubuntu 24.04
  development container with no ROS installed. GitHub's `ubuntu-latest`
  runners are expected to work too, since `ros2-client`'s own CI runs
  equivalent loopback tests, but the CI job's first run is the
  confirmation.

## Limits (read before relying on this)

* **Tested with ROS 2 Jazzy on one host only**
  ([above](#tested-against-ros-2-jazzy)), RViz included (under Xvfb). It
  has not been tested with Humble, several hosts on a real network, or live
  sensors. Please
  report results.
* VIO accuracy is the Basalt port's (see the root README's EuRoC results).
  The node adds no tuning. It has been run on synthetic data and on all
  eight EuRoC query sequences, replayed over DDS by a RustDDS player (MH_02)
  and played from a rosbag2, not on a live sensor.
* `nav_msgs/Odometry` covariances are zero (unknown): the marginalized
  Basalt window does not expose a calibrated per-frame covariance.
* The localization covariance is a fixed, configurable diagonal, not an
  estimate.
* Localization extracts SIFT on the CPU per frame, which takes about a
  second per 320x240 image in the test profile. Expect a few Hz at best in
  `--release` at VGA, so the node keeps only the newest image. There is no
  tracking prior between frames.
* The default PnP minimal solver (DLT) needs non-coplanar landmarks.
  Real SfM maps are fine; a map of a single wall is not.
* The camera/IMU time offset is taken from the calibration
  (`cam_time_offset_ns`) and not estimated online.
* `visloc_map_localize_node` needs gravity-aligned odometry (z up), as a
  VIO provides. Planar wheel odometry qualifies on level floors only; on a
  ramp its missing pitch and height change are errors the correction can
  only remove at fixes. It has been tested with simulated wheel odometry and
  with a real robot's on a level office floor ([above](#on-a-ground-robot)).
  A RustDDS reader sometimes never receives a transient-local
  `/tf_static` writer's transforms (about 2% of readers started alongside
  freshly started publishers, in a test); both nodes then subscribe again,
  5 s after the first subscription and with doubling waits, up to six
  times, while a transform they need is missing
  (`examples/tf_static_probe.rs` reproduces it). Camera extrinsics come from the
  map or from `/tf_static`, which is read as static: a camera that moves
  on the robot (pan-tilt) is not supported. Its map is the stereo-inertial SLAM's (ORB),
  not a COLMAP/SIFT model. The correction is a filtered yaw + translation
  offset, not a joint optimization with the VIO, so the pose error grows
  between fixes where the map does not cover the route.
