//! ROS 2 nodes for visloc-rs over pure-Rust DDS (`ros2-client` / RustDDS).
//!
//! The crate is split into a transport-agnostic core and a thin DDS shell:
//!
//! * [`msgs`] -- serde mirrors of the ROS 2 message types the nodes use;
//! * [`image`] -- `sensor_msgs/Image` decoding (mono8/mono16/rgb8/bgr8/...);
//! * [`sync`] -- stereo pairing, IMU interval bookkeeping, drop-oldest
//!   frame queue for the VIO node;
//! * [`vio`] -- Basalt VIO estimator wrapper and output message builders;
//! * [`localize`] -- COLMAP-map localization core and output builders;
//! * [`params`] -- ROS-style command-line parameters and remapping;
//! * [`queue`] -- bounded drop-oldest hand-off queue;
//! * [`dds`] -- the `ros2-client` shell (node, QoS, pub/sub threads).
//!
//! The binaries `visloc_vio_node` and `visloc_localize_node` wire these
//! together; see `README.md` for topics and parameters.
#![forbid(unsafe_code)]

pub mod dds;
pub mod image;
pub mod localize;
pub mod msgs;
pub mod params;
pub mod queue;
pub mod sync;
pub mod vio;
