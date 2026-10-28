//! Metric sizes and positions of static objects from a VIO run (Phase 3 of
//! `docs/object_metric_slam_plan.md`).
//!
//! ```text
//! # the VIO, keeping its feature tracks
//! cargo run --release --example basalt_euroc_vio_demo -- --euroc-dir SEQ --calibration CALIB \
//!   --config CONFIG --mono --out-dir target/seq_vio --dump-tracks target/seq_vio/tracks.csv
//! # 2D object tracks with timestamps, all classes
//! cargo run --release --features image-io,onnx-inference --example yolox_track_demo -- \
//!   --model yolox_s.onnx --images SEQ/mav0/cam0/data --boxes-csv boxes.csv --all-classes
//! # the objects
//! cargo run --release --example objects_metric_offline -- --tracks target/seq_vio/tracks.csv \
//!   --trajectory target/seq_vio/trajectory.tum --calibration CALIB --boxes boxes.csv
//! ```
//!
//! A feature track belongs to the object whose camera-0 boxes contain it
//! most often, if they do at least 3 times. Each object's tracks are
//! triangulated with the trajectory's poses, and an upright box is fitted
//! (`visloc_objects::metric`). Only static objects come out right: a moving
//! object's rays do not meet, and its tracks are rejected. The trajectory is
//! the IMU pose (`T_w_i`), as `basalt_euroc_vio_demo` writes it; for
//! `--mono-vo` runs, whose trajectory is the camera pose, pass
//! `--trajectory-is-camera`. Writes `objects.csv` next to the boxes unless
//! `--out` says otherwise.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Isometry3, Point2, Quaternion, Translation3, UnitQuaternion, Vector3};
use visloc_rs::objects::{
    estimate_static_object, plausible_size, BearingObservation, BoxFitParams, TriangulationParams,
    COCO_CLASSES,
};

struct Args {
    tracks: PathBuf,
    trajectory: PathBuf,
    calibration: PathBuf,
    boxes: PathBuf,
    out: Option<PathBuf>,
    trajectory_is_camera: bool,
}

struct ObjectBox {
    object_id: u64,
    class_id: u32,
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
}

/// Minimum camera-0 observations inside an object's boxes for a track to
/// belong to it.
const MIN_VOTES: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let calibration = visloc_basalt::BasaltCalibration::from_path(&args.calibration)?;
    let poses = read_trajectory(&args.trajectory)?;
    let boxes = read_boxes(&args.boxes)?;

    // frame_id,timestamp_ns,camera_id,track_id,x,y
    let mut observations: Vec<(i64, u16, u64, Point2<f64>)> = Vec::new();
    for (index, line) in fs::read_to_string(&args.tracks)?
        .lines()
        .enumerate()
        .skip(1)
    {
        let f: Vec<&str> = line.split(',').collect();
        let bad = || format!("{}: line {}", args.tracks.display(), index + 1);
        if f.len() != 6 {
            return Err(bad().into());
        }
        let number = |i: usize| f[i].trim().parse::<f64>().map_err(|_| bad());
        observations.push((
            f[1].trim().parse().map_err(|_| bad())?,
            f[2].trim().parse().map_err(|_| bad())?,
            f[3].trim().parse().map_err(|_| bad())?,
            Point2::new(number(4)?, number(5)?),
        ));
    }

    // Votes: how often each track is seen in camera 0 inside each object's box.
    let mut votes: HashMap<u64, HashMap<u64, usize>> = HashMap::new();
    let mut classes: HashMap<u64, u32> = HashMap::new();
    for &(timestamp_ns, camera_id, track_id, pixel) in &observations {
        if camera_id != 0 {
            continue;
        }
        for b in boxes.get(&timestamp_ns).into_iter().flatten() {
            if pixel.x >= b.x_min && pixel.x <= b.x_max && pixel.y >= b.y_min && pixel.y <= b.y_max
            {
                *votes
                    .entry(track_id)
                    .or_default()
                    .entry(b.object_id)
                    .or_default() += 1;
                classes.insert(b.object_id, b.class_id);
            }
        }
    }
    let owner: HashMap<u64, u64> = votes
        .iter()
        .filter_map(|(&track_id, objects)| {
            let (&object_id, &count) = objects
                .iter()
                .max_by_key(|(id, count)| (**count, std::cmp::Reverse(**id)))?;
            (count >= MIN_VOTES).then_some((track_id, object_id))
        })
        .collect();

    // Bearing observations per object and track, from every camera.
    let mut per_object: BTreeMap<u64, BTreeMap<u64, Vec<BearingObservation>>> = BTreeMap::new();
    for &(timestamp_ns, camera_id, track_id, pixel) in &observations {
        let Some(&object_id) = owner.get(&track_id) else {
            continue;
        };
        let (Some(pose), Some(camera)) = (poses.get(&timestamp_ns), calibration.camera(camera_id))
        else {
            continue;
        };
        let camera_to_world = if args.trajectory_is_camera {
            *pose
        } else {
            let t_i_c = calibration
                .camera_to_imu(camera_id)
                .ok_or("camera without T_i_c")?;
            pose * Isometry3::from_parts(Translation3::from(t_i_c.translation), t_i_c.rotation)
        };
        let Some(bearing) = camera.unproject(&pixel) else {
            continue;
        };
        per_object
            .entry(object_id)
            .or_default()
            .entry(track_id)
            .or_default()
            .push(BearingObservation {
                camera_to_world,
                bearing,
            });
    }

    let mut csv = String::from(
        "object_id,class_id,class,tracks,points,center_x,center_y,center_z,length,width,height,yaw,scale_source,plausible\n",
    );
    println!(
        "{:>6} {:<14} {:>6} {:>6}  {:>22}  {:>24}  plausible",
        "object", "class", "tracks", "points", "size l x w x h (m)", "centre (m)"
    );
    for (object_id, tracks) in &per_object {
        let class_id = classes[object_id];
        let tracks: Vec<Vec<BearingObservation>> = tracks.values().cloned().collect();
        let Some(estimate) = estimate_static_object(
            *object_id,
            class_id,
            &tracks,
            &TriangulationParams::default(),
            &BoxFitParams::default(),
        ) else {
            continue;
        };
        let b = estimate.upright_box;
        let largest = b.length.max(b.width).max(b.height);
        let plausible = match plausible_size(class_id) {
            Some((low, high)) if (low..=high).contains(&largest) => "yes",
            Some(_) => "NO",
            None => "-",
        };
        let class = COCO_CLASSES.get(class_id as usize).copied().unwrap_or("?");
        println!(
            "{object_id:>6} {class:<14} {:>6} {:>6}  {:>6.2} x {:>5.2} x {:>5.2}  ({:>6.2}, {:>6.2}, {:>6.2})  {plausible}",
            tracks.len(),
            estimate.points,
            b.length,
            b.width,
            b.height,
            b.center.x,
            b.center.y,
            b.center.z
        );
        writeln!(
            csv,
            "{object_id},{class_id},{class},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},triangulated,{plausible}",
            tracks.len(),
            estimate.points,
            b.center.x,
            b.center.y,
            b.center.z,
            b.length,
            b.width,
            b.height,
            b.yaw
        )?;
    }
    let out = args
        .out
        .unwrap_or_else(|| args.boxes.with_file_name("objects.csv"));
    fs::write(&out, csv)?;
    println!("objects: {}", out.display());
    Ok(())
}

/// TUM lines `t tx ty tz qx qy qz qw`, `t` in seconds with up to nine
/// decimals, keyed by the exact nanosecond timestamp.
fn read_trajectory(path: &PathBuf) -> Result<HashMap<i64, Isometry3<f64>>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut poses = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let bad = || format!("{}: line {}", path.display(), index + 1);
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 8 {
            return Err(bad());
        }
        // f64 seconds cannot hold nanoseconds at epoch scale: parse the digits.
        let (seconds, fraction) = f[0].split_once('.').unwrap_or((f[0], ""));
        let digits = format!("{fraction:0<9}");
        let timestamp_ns = seconds.parse::<i64>().map_err(|_| bad())? * 1_000_000_000
            + digits[..9].parse::<i64>().map_err(|_| bad())?;
        let v: Vec<f64> = f[1..]
            .iter()
            .map(|s| s.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| bad())?;
        let rotation = UnitQuaternion::from_quaternion(Quaternion::new(v[6], v[3], v[4], v[5]));
        poses.insert(
            timestamp_ns,
            Isometry3::from_parts(Translation3::from(Vector3::new(v[0], v[1], v[2])), rotation),
        );
    }
    Ok(poses)
}

/// `timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max`.
fn read_boxes(path: &PathBuf) -> Result<HashMap<i64, Vec<ObjectBox>>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut boxes: HashMap<i64, Vec<ObjectBox>> = HashMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.starts_with("timestamp_ns") || line.trim().is_empty() {
            continue;
        }
        let bad = || format!("{}: line {}", path.display(), index + 1);
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() != 8 {
            return Err(bad());
        }
        let n = |i: usize| f[i].parse::<f64>().map_err(|_| bad());
        boxes
            .entry(f[0].parse().map_err(|_| bad())?)
            .or_default()
            .push(ObjectBox {
                object_id: f[1].parse().map_err(|_| bad())?,
                class_id: f[2].parse().map_err(|_| bad())?,
                x_min: n(4)?,
                y_min: n(5)?,
                x_max: n(6)?,
                y_max: n(7)?,
            });
    }
    Ok(boxes)
}

fn parse_args() -> Result<Args, String> {
    let usage = "usage: objects_metric_offline --tracks FILE --trajectory FILE --calibration FILE \
                 --boxes FILE [--out FILE] [--trajectory-is-camera]";
    let (mut tracks, mut trajectory, mut calibration, mut boxes, mut out) =
        (None, None, None, None, None);
    let mut trajectory_is_camera = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{arg} needs a value\n{usage}"))
        };
        match arg.as_str() {
            "--tracks" => tracks = Some(value()?),
            "--trajectory" => trajectory = Some(value()?),
            "--calibration" => calibration = Some(value()?),
            "--boxes" => boxes = Some(value()?),
            "--out" => out = Some(value()?),
            "--trajectory-is-camera" => trajectory_is_camera = true,
            _ => return Err(format!("unknown argument {arg}\n{usage}")),
        }
    }
    Ok(Args {
        tracks: tracks.ok_or(usage)?,
        trajectory: trajectory.ok_or(usage)?,
        calibration: calibration.ok_or(usage)?,
        boxes: boxes.ok_or(usage)?,
        out,
        trajectory_is_camera,
    })
}
