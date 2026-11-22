#![forbid(unsafe_code)]
//! Robot-navigation support for visloc-rs maps.
//!
//! visloc-rs localizes a robot in a visual map; a navigation stack such as
//! Nav2 plans and drives on a 2D occupancy grid. This crate is the bridge:
//!
//! - [`occupancy`]: builds the 2D occupancy grid of a sparse visual map
//!   (landmarks, and the camera-to-landmark rays that show free space), in a
//!   2.5D grid so low furniture survives rays passing above it.
//! - [`grid`]: the grid type, with `nav_msgs/OccupancyGrid` conventions and
//!   reading / writing in the ROS `map_server` format (PGM + YAML) that
//!   Nav2's `map_server` loads.
//! - [`sim`]: a deterministic simulation of a camera-carrying
//!   differential-drive robot (textured floor plan, drifting wheel odometry,
//!   synthetic camera detections, range scans, a mapping run), for testing
//!   localization and navigation end to end without a robot;
//!   [`sim_robot`] localizes that robot in its map with visloc's
//!   localization pipeline and keeps `map -> odom`, as a navigation stack
//!   expects from its localization.
//!
//! Planning and control are left to the navigation stack; see
//! `docs/navigation.md` for running Nav2 on visloc-rs localization.

pub mod grid;
pub mod occupancy;
pub mod sim;
pub mod sim_robot;

pub use grid::{GridCounts, GridGeometry, MapServerError, OccupancyGrid, FREE, OCCUPIED, UNKNOWN};
pub use occupancy::{
    floor_z_from_cameras, occupancy_from_visual_map, OccupancyConfig, OccupancyMapper,
    OccupancyStats,
};
