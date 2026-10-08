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
| `odom` | `nav_msgs/Odometry` | out | pose of the IMU (body) in a gravity-aligned, z-up `odom` frame; twist in the body frame |
| `pose` | `geometry_msgs/PoseStamped` | out | same pose |
| `path` | `nav_msgs/Path` | out | last `path_max_length` poses |
| `/tf` | `tf2_msgs/TFMessage` | out | `odom_frame -> body_frame` |

Output stamps are the left image's header stamps.

Parameters (`--name value`, `--name=value` or `--ros-args -p name:=value`):

| parameter | default | |
|---|---|---|
| `calibration` | required | Basalt calibration JSON |
| `config` | required | Basalt VIO config JSON |
| `odom_frame` / `body_frame` | `odom` / `imu_link` | output frame ids |
| `publish_tf`, `publish_path` | `true` | |
| `path_max_length` | `2000` | |
| `stereo_sync_tolerance_ms` | `2.0` | max left/right stamp difference |
| `frame_queue` | `4` | paired frames waiting for the estimator (drop-oldest) |
| `imu_buffer` | `4000` | buffered IMU samples (drop-oldest) |
| `image_qos_depth`, `imu_qos_depth` | `5`, `400` | subscription history depth |
| `urgent_keyframes` | `true` | the demo's urgent-keyframe defaults |
| `imu_seed_klt` | `false` | `config.optical_flow_imu_seed_rotation` |
| `domain_id` | `$ROS_DOMAIN_ID` or 0 | |
| `node_name` | `visloc_vio` | also `-r __node:=...`, `-r __ns:=...` |

Data flow and threading: one receive thread per subscription decodes images
(mono8 / mono16 / rgb8 / bgr8 / rgba8 / bgra8; colour is converted with
BT.601 luma; 8-bit is widened `u16 = u8 << 8` exactly like the EuRoC
reader) and hands them to a stereo synchronizer (closest stamp within the
tolerance). Paired frames go into a bounded drop-oldest queue. The estimator
thread releases a frame only once IMU data reaches its stamp, and passes it
the half-open IMU interval `(previous frame, this frame]`, which is the
`EurocSensorFrame` contract. If the estimator falls behind, the oldest
frames are dropped. The IMU interval of the next processed frame still
starts at the last processed frame, so preintegration stays contiguous.
`cam_time_offset_ns` from the calibration is added to camera stamps. On an
estimator error the node logs it, rebuilds the estimator and
re-initializes from IMU.

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

## Running inside a ROS 2 graph

* **Discovery / domain**: standard SPDP multicast discovery on
  `ROS_DOMAIN_ID` (or `--domain-id`). Domain ids above ~230 overflow the
  RTPS port mapping, as in any DDS. RustDDS does not implement Fast DDS's
  *discovery server* or Cyclone's unicast-only peer lists. If your network
  blocks multicast, these nodes will not be discovered.
* **RMW interoperability**: RustDDS speaks standard RTPS 2.x with CDR
  (`CDR_LE`) payloads and ROS 2's topic/type mangling (`rt/<topic>`,
  `sensor_msgs::msg::dds_::Image_`). That is what `rmw_fastrtps_cpp` (the
  default in Humble/Jazzy) and `rmw_cyclonedds_cpp` use, and `ros2-client`
  is maintained against both. `rmw_zenoh` is a different protocol and is
  not supported by these binaries.
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
* `dds_nodes` spawns the real `visloc_vio_node` / `visloc_localize_node`
  binaries on a private domain id, and drives them from a `ros2-client`
  test node with synthetic images and IMU. It asserts that `odom`/`pose`,
  and `pose`/`inlier_count`/`/diagnostics`, arrive with sane content. These
  tests need UDP multicast on loopback. They pass in an Ubuntu 24.04
  development container with no ROS installed. GitHub's `ubuntu-latest`
  runners are expected to work too, since `ros2-client`'s own CI runs
  equivalent loopback tests, but the CI job's first run is the
  confirmation.

## Limits (read before relying on this)

* **Not tested against a real ROS 2 installation.** Interop with
  `rmw_fastrtps_cpp` / `rmw_cyclonedds_cpp` follows from RTPS/CDR
  conformance and the pinned byte layouts, but it has only been exercised
  RustDDS-to-RustDDS. Please report results with `ros2 topic echo`, RViz
  and a rosbag.
* VIO accuracy is the Basalt port's (see the root README's EuRoC results).
  The node adds no tuning. It has only been run on synthetic data here, not
  on a live sensor or a replayed EuRoC bag.
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
