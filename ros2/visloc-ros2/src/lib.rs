//! ROS 2 nodes for visloc-rs over pure-Rust DDS (`ros2-client` / RustDDS).
//!
//! The crate is split into a transport-agnostic core and a thin DDS shell:
//!
//! * [`msgs`] -- serde mirrors of the ROS 2 message types the nodes use;
//! * [`image`] -- `sensor_msgs/Image` decoding (mono8/mono16/rgb8/bgr8/...);
//! * [`sync`] -- stereo pairing, IMU interval bookkeeping, drop-oldest
//!   frame queue for the VIO node;
//! * [`vio`] -- Basalt VIO estimator wrapper and output message builders;
//! * [`vio_mapper`] -- the VIO node's opt-in online mapper (loop closure,
//!   global BA) and its `map -> odom` correction;
//! * [`localize`] -- COLMAP-map localization core and output builders;
//! * [`map_localize`] -- localization against a saved Basalt map, correcting
//!   VIO odometry into it (`map -> odom`);
//! * [`params`] -- ROS-style command-line parameters and remapping;
//! * [`queue`] -- bounded drop-oldest hand-off queue;
//! * [`wheel_odom`] -- wheel odometry calibrated against the VIO while
//!   driving;
//! * [`nav_sim`] -- messages of the simulated, visloc-localized robot
//!   that `visloc_nav_sim_node` drives for Nav2;
//! * [`dds`] -- the `ros2-client` shell (node, QoS, pub/sub threads).
//!
//! The binaries `visloc_vio_node` and `visloc_localize_node` wire these
//! together; see `README.md` for topics and parameters.
#![forbid(unsafe_code)]

pub mod dds;
pub mod image;
pub mod localize;
pub mod map_localize;
pub mod msgs;
pub mod nav_sim;
pub mod odom_bridge;
pub mod params;
pub mod queue;
pub mod sync;
pub mod tf_tree;
pub mod tracking;
pub mod vio;
pub mod vio_external;
pub mod vio_mapper;
pub mod wheel_odom;
