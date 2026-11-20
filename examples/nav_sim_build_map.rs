//! Builds the map of the simulated apartment (`visloc_navigation::sim`) the
//! way a robot would: a mapping run through every room, saved as a COLMAP
//! model with landmark descriptors, plus the occupancy grid Nav2 plans on.
//!
//! ```text
//! cargo run --release --example nav_sim_build_map -- --out runs/nav_sim/map
//! ```
//!
//! Writes, under `--out`:
//!
//! - `sparse/{cameras,images,points3D}.txt`: the map (landmarks with a 2 cm
//!   error, keyframe poses with 1 cm, standing in for SLAM error);
//! - `landmark_descriptors.txt`: one descriptor per landmark;
//! - `occupancy.{pgm,yaml}`: the occupancy grid, for Nav2's `map_server`,
//!   built from the map alone (no ground-truth walls), exactly as
//!   `visual_map_to_occupancy --map-dir <out> --camera-height 0.5` does;
//! - `mapping_ground_truth.tum`: the true robot poses at the keyframes.
//!
//! The world, the robot and the mapping run are deterministic (`--seed`).
//! `ros2/visloc-ros2`'s `visloc_nav_sim_node` drives a robot in the same
//! world and localizes it in this map; see `docs/navigation.md`.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use visloc_rs::io::colmap::write_colmap_text_model;
use visloc_rs::io::descriptors::write_landmark_descriptors_txt;
use visloc_rs::navigation::sim::{
    apartment_mapping_route, build_map, CameraRig, MappingConfig, World,
};
use visloc_rs::navigation::{occupancy_from_visual_map, OccupancyConfig};
use visloc_rs::types::LandmarkDescriptorStore;

/// Camera height above the floor of the simulated robot.
const CAMERA_HEIGHT: f64 = 0.5;

fn run() -> Result<(), String> {
    let mut out = PathBuf::from("runs/nav_sim/map");
    let mut world_seed = 1;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--out" => out = PathBuf::from(value),
            "--seed" => world_seed = value.parse().map_err(|_| format!("bad seed {value}"))?,
            other => {
                return Err(format!(
                    "unknown argument {other}; usage: --out DIR [--seed N]"
                ))
            }
        }
    }
    let world = World::apartment(world_seed);
    let rig = CameraRig::forward(CAMERA_HEIGHT);
    let run = build_map(
        &world,
        &rig,
        &apartment_mapping_route(),
        &MappingConfig::default(),
    );
    let sparse = out.join("sparse");
    std::fs::create_dir_all(&sparse).map_err(|e| format!("{}: {e}", sparse.display()))?;
    write_colmap_text_model(&run.map, &sparse).map_err(|e| format!("writing the model: {e}"))?;
    write_landmark_descriptors_txt(
        &LandmarkDescriptorStore::from_visual_map(&run.map),
        out.join("landmark_descriptors.txt"),
    )
    .map_err(|e| format!("writing descriptors: {e}"))?;

    let config = OccupancyConfig {
        floor_z: 0.0,
        ..OccupancyConfig::default()
    };
    let (grid, _) = occupancy_from_visual_map(&run.map, config, 0.2, 0.5)
        .ok_or("the mapping run saw nothing")?;
    let yaml = grid
        .write_map_server(out.join("occupancy"))
        .map_err(|e| format!("writing the grid: {e}"))?;

    let mut tum = String::from("# timestamp_s x y z qx qy qz qw (robot base, ground truth)\n");
    for (i, pose) in run.keyframe_ground_truth.iter().enumerate() {
        let (s, c) = (pose.yaw / 2.0).sin_cos();
        writeln!(tum, "{i} {} {} 0 0 0 {s} {c}", pose.x, pose.y).expect("string write");
    }
    std::fs::write(out.join("mapping_ground_truth.tum"), tum)
        .map_err(|e| format!("writing the ground truth: {e}"))?;

    let counts = grid.counts();
    println!(
        "world: {} walls, {} texture points; map: {} landmarks, {} keyframes",
        world.walls.len(),
        world.landmarks.len(),
        run.map.landmarks.len(),
        run.map.keyframes.len()
    );
    println!(
        "occupancy: {} x {} cells, free {} occupied {} unknown {} -> {}",
        grid.geometry.width,
        grid.geometry.height,
        counts.free,
        counts.occupied,
        counts.unknown,
        yaml.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
