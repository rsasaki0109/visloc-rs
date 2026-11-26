//! Map localization over stdin/stdout, for simulators that render the camera themselves
//! (Robot Native Engine's street example; see `docs/rne_migration.md`).
//!
//! ```text
//! visloc_map_localize_server --map DIR [--camera 0] [--min_inliers 30] [--max_hamming 70]
//!     [--ratio 0.8] [--reprojection_threshold 3] [--projection_radius 20] [--match_threads 1]
//!     [--prior_radius 0] [--no_global_with_prior]
//! ```
//!
//! The map is loaded once (`visloc_map_localize_node`'s core, [`MapLocalizer`]). Each line
//! on stdin is one image: `PATH` or `PATH tx ty tz qx qy qz qw`, an 8-bit binary PGM (`P5`)
//! with the map's camera model, and optionally the predicted camera pose in the map
//! (map <- camera, OpenCV camera axes: x right, y down, z forward), which turns on the
//! projection search. Each answer is one line on stdout:
//! `ok tx ty tz qx qy qz qw INLIERS MS` (the camera pose in the map) or `fail REASON`.
//! A line `quit` (or the end of stdin) stops the server.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use visloc_basalt::RawU16Image;
use visloc_core::geometry::SE3;
use visloc_ros2::map_localize::{load_basalt_map, MapLocalizeOptions, MapLocalizer};

fn main() {
    if let Err(error) = run() {
        eprintln!("visloc_map_localize_server: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |name: &str| -> Option<String> {
        let flag = format!("--{name}");
        args.iter()
            .position(|a| *a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let parse = |name: &str| -> Result<Option<f64>, String> {
        value(name)
            .map(|v| v.parse::<f64>().map_err(|e| format!("--{name}: {e}")))
            .transpose()
    };
    let map_dir = PathBuf::from(value("map").ok_or("--map DIR is required")?);
    let camera = parse("camera")?.map_or(0, |v| v as u16);
    let mut options = MapLocalizeOptions::default();
    if let Some(v) = parse("min_inliers")? {
        options.min_inliers = v as usize;
    }
    if let Some(v) = parse("max_hamming")? {
        options.max_hamming = v as u32;
    }
    if let Some(v) = parse("ratio")? {
        options.ratio = (v > 0.0).then_some(v as f32);
    }
    if let Some(v) = parse("reprojection_threshold")? {
        options.reprojection_px = v;
    }
    if let Some(v) = parse("projection_radius")? {
        options.projection_radius_px = (v > 0.0).then_some(v);
    }
    if let Some(v) = parse("match_threads")? {
        options.match_threads = v as usize;
    }
    if let Some(v) = parse("prior_radius")? {
        options.prior_radius_m = (v > 0.0).then_some(v);
    }
    options.no_global_with_prior = args.iter().any(|a| a == "--no_global_with_prior");

    let map = load_basalt_map(&map_dir)?;
    let localizer = MapLocalizer::new(map, camera, &options)?;
    let imu_from_camera = localizer
        .map()
        .imu_from_camera
        .get(usize::from(camera))
        .cloned()
        .ok_or("the map has no extrinsics for that camera")?;
    let camera_from_imu = imu_from_camera.inverse();
    eprintln!(
        "visloc_map_localize_server: {} landmarks, ready",
        localizer.map().map.landmarks.len()
    );

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" {
            break;
        }
        let answer = match answer(line, &localizer, &imu_from_camera, &camera_from_imu) {
            Ok(answer) => answer,
            Err(reason) => format!("fail {}", reason.replace(char::is_whitespace, "_")),
        };
        writeln!(stdout, "{answer}").map_err(|e| e.to_string())?;
        stdout.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn answer(
    line: &str,
    localizer: &MapLocalizer,
    imu_from_camera: &SE3,
    camera_from_imu: &SE3,
) -> Result<String, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let image = read_pgm(Path::new(fields[0]))?;
    let prior = match fields.len() {
        1 => None,
        8 => {
            let v = fields[1..]
                .iter()
                .map(|f| f.parse::<f64>().map_err(|e| format!("prior: {e}")))
                .collect::<Result<Vec<_>, _>>()?;
            let map_from_camera = SE3 {
                rotation: UnitQuaternion::from_quaternion(Quaternion::new(v[6], v[3], v[4], v[5])),
                translation: Vector3::new(v[0], v[1], v[2]),
            };
            // map <- body = map <- camera <- body
            Some(map_from_camera.compose(camera_from_imu))
        }
        n => {
            return Err(format!(
                "expected PATH or PATH and 7 pose values, got {n} fields"
            ))
        }
    };
    let fix = localizer.localize(&image, prior.as_ref());
    let Some(map_from_body) = fix.map_from_body else {
        return Err(fix
            .failure
            .unwrap_or_else(|| format!("{} inliers", fix.inliers)));
    };
    let map_from_camera = map_from_body.compose(imu_from_camera);
    let (t, q) = (map_from_camera.translation, map_from_camera.rotation);
    Ok(format!(
        "ok {:.6} {:.6} {:.6} {:.9} {:.9} {:.9} {:.9} {} {:.1}",
        t.x, t.y, t.z, q.i, q.j, q.k, q.w, fix.inliers, fix.elapsed_ms
    ))
}

/// An 8-bit binary PGM as Basalt's raw image (`u8 << 8`).
fn read_pgm(path: &Path) -> Result<RawU16Image, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut fields = Vec::new();
    let mut pos = 0;
    while fields.len() < 4 && pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..pos]).into_owned());
    }
    if fields.len() < 4 || fields[0] != "P5" || fields[3] != "255" {
        return Err(format!("{}: not an 8-bit binary PGM", path.display()));
    }
    let (width, height): (usize, usize) = (
        fields[1].parse().map_err(|_| "PGM width")?,
        fields[2].parse().map_err(|_| "PGM height")?,
    );
    let data = &bytes[pos + 1..];
    if data.len() < width * height {
        return Err(format!("{}: short PGM", path.display()));
    }
    let pixels = data[..width * height]
        .iter()
        .map(|&v| u16::from(v) << 8)
        .collect();
    RawU16Image::new(width, height, pixels).map_err(|e| format!("{e:?}"))
}
