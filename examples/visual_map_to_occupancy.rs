//! Converts a visual map into a 2D occupancy grid that Nav2 (or any ROS
//! `map_server`) loads, so a robot can plan on the map it localizes in.
//!
//! ```text
//! cargo run --release --example visual_map_to_occupancy -- \
//!   --map-dir runs/office_map --camera-height 0.55 --out runs/office_map/occupancy
//! ```
//!
//! The input is a COLMAP model (text or binary): a map saved by
//! `basalt_euroc_online_slam_demo --save-map` (`--map-dir`, which reads its
//! `sparse/`) or any model directory (`--sparse`). It must be gravity-aligned
//! with `z` up, as the stereo-inertial SLAM's maps are. Landmarks between
//! `--min-height` and `--max-height` above the floor are obstacles, and the
//! rays from each camera to the landmarks it observed are free space (see
//! `visloc_navigation::occupancy`). Writes `<out>.pgm` and `<out>.yaml`.
//!
//! The floor is `--floor-z`, or the median camera height minus
//! `--camera-height` (the camera's height above the floor on the robot).

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use visloc_rs::io::colmap::{read_colmap_binary_model, read_colmap_text_model};
use visloc_rs::navigation::{floor_z_from_cameras, occupancy_from_visual_map, OccupancyConfig};

const USAGE: &str = "usage: visual_map_to_occupancy (--map-dir DIR | --sparse DIR) --out STEM \
(--camera-height M | --floor-z Z) [--resolution M] [--min-height M] [--max-height M] \
[--max-range M] [--min-hits N] [--min-observations N] [--free-radius M] [--margin M]";

struct Args {
    sparse: PathBuf,
    out: PathBuf,
    camera_height: Option<f64>,
    floor_z: Option<f64>,
    free_radius: f64,
    margin: f64,
    config: OccupancyConfig,
}

fn parse_args() -> Result<Args, String> {
    let mut sparse = None;
    let mut out = None;
    let mut camera_height = None;
    let mut floor_z = None;
    let mut free_radius = 0.2;
    let mut margin = 0.5;
    let mut config = OccupancyConfig::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |v: String| v.parse::<f64>().map_err(|_| format!("bad number: {v}"));
        match flag.as_str() {
            "--map-dir" => sparse = Some(PathBuf::from(value()?).join("sparse")),
            "--sparse" => sparse = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--camera-height" => camera_height = Some(number(value()?)?),
            "--floor-z" => floor_z = Some(number(value()?)?),
            "--resolution" => config.resolution = number(value()?)?,
            "--min-height" => config.min_height = number(value()?)?,
            "--max-height" => config.max_height = number(value()?)?,
            "--max-range" => config.max_range = number(value()?)?,
            "--min-hits" => config.min_hits = number(value()?)? as u32,
            "--min-observations" => config.min_observations = number(value()?)? as usize,
            "--free-radius" => free_radius = number(value()?)?,
            "--margin" => margin = number(value()?)?,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let sparse = sparse.ok_or(format!("--map-dir or --sparse is required\n{USAGE}"))?;
    let out = out.ok_or(format!("--out is required\n{USAGE}"))?;
    if camera_height.is_none() == floor_z.is_none() {
        return Err(format!("give one of --camera-height or --floor-z\n{USAGE}"));
    }
    Ok(Args {
        sparse,
        out,
        camera_height,
        floor_z,
        free_radius,
        margin,
        config,
    })
}

fn read_model(dir: &Path) -> Result<visloc_rs::types::VisualMap, String> {
    let result = if dir.join("points3D.txt").exists() {
        read_colmap_text_model(dir)
    } else {
        read_colmap_binary_model(dir)
    };
    result.map_err(|error| format!("reading the COLMAP model in {}: {error}", dir.display()))
}

fn run(args: Args) -> Result<(), String> {
    let started = Instant::now();
    let map = read_model(&args.sparse)?;
    let mut config = args.config;
    config.floor_z = match (args.floor_z, args.camera_height) {
        (Some(z), _) => z,
        (None, Some(height)) => floor_z_from_cameras(&map, height)
            .ok_or("the model has no posed images to place the floor")?,
        (None, None) => unreachable!("checked in parse_args"),
    };
    let (grid, stats) =
        occupancy_from_visual_map(&map, config.clone(), args.free_radius, args.margin)
            .ok_or("nothing to map: no landmark observed by enough posed images")?;
    let yaml = grid
        .write_map_server(&args.out)
        .map_err(|error| format!("writing {}: {error}", args.out.display()))?;
    let counts = grid.counts();
    let g = grid.geometry;
    println!(
        "{} landmarks, {} images -> {} x {} cells at {} m, origin ({:.2}, {:.2}); floor z {:.3}",
        map.landmarks.len(),
        map.keyframes.len(),
        g.width,
        g.height,
        g.resolution,
        g.origin_x,
        g.origin_y,
        config.floor_z
    );
    println!(
        "free {} occupied {} unknown {}; rays {}, landmarks in band {}, out of band {}, beyond range {}",
        counts.free,
        counts.occupied,
        counts.unknown,
        stats.rays,
        stats.landmarks_in_band,
        stats.landmarks_out_of_band,
        stats.rays_out_of_range
    );
    println!(
        "wrote {} ({:.2} s)",
        yaml.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() -> ExitCode {
    match parse_args().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
