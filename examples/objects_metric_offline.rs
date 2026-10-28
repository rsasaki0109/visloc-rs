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
//! (`visloc_objects::metric`). Objects of the same class whose boxes lie
//! within 0.5 m of each other are merged and fitted again: the 2D tracker
//! gives an object a new ID each time it comes back into view. Only static
//! objects triangulate: a moving object's rays do not meet.
//!
//! Objects of a class that moves (people, vehicles, animals) are also placed
//! on the ground frame by frame (`visloc_objects::ground`). The ground
//! follows the camera's path through the triangulated feature tracks seen
//! within 10 s of the object (`PathGround`, so slopes stay sharp), then the
//! points right under its foot (stairs and kerbs). The middle of its box's
//! bottom edge comes from the gravity direction, so a sideways camera works.
//! That gives each one a metric height, width, distance and speed, moving or
//! not. Boxes cut by the image border or scored below 0.5 are skipped. The
//! trajectory is
//! the IMU pose (`T_w_i`), as `basalt_euroc_vio_demo` writes it; for
//! `--mono-vo` runs, whose trajectory is the camera pose, pass
//! `--trajectory-is-camera`. Writes `objects.csv` (static),
//! `ground_objects.csv` (on the ground) and `ground_tracks.csv` (their feet,
//! frame by frame) next to the boxes unless `--out` names the first.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use nalgebra::{
    Isometry3, Point2, Point3, Quaternion, Translation3, UnitQuaternion, Vector2, Vector3,
};
use visloc_rs::objects::{
    estimate_static_object, place_on_local_ground, plausible_height, plausible_size,
    summarize_ground_track, triangulate, BBox, BearingObservation, BoxFitParams, BoxRays,
    GroundParams, GroundTrackParams, GroundView, ObjectEstimate, PathGround, TriangulationParams,
    COCO_CLASSES, MOVABLE_COCO_CLASSES,
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
    score: f64,
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
}

/// Minimum camera-0 observations inside an object's boxes for a track to
/// belong to it.
const MIN_VOTES: usize = 3;

/// Objects of the same class closer than this (centre to centre, metres)
/// are the same object seen under several 2D track IDs.
const MERGE_DISTANCE: f64 = 0.5;

/// Triangulated points a merged object needs to be reported; with fewer the
/// box is mostly noise. Fragments only need the library default to be merged.
const MIN_POINTS: usize = 12;

/// Feature points seen within this long of an object on the ground count
/// for its ground height and the ground under it.
const GROUND_WINDOW_NS: i64 = 10_000_000_000;

/// Boxes placed on the ground: at least this score, at least this far from
/// the image border (a cut box has no foot), over at least this long.
const MIN_GROUND_SCORE: f64 = 0.5;
const BORDER_PX: f64 = 2.0;
const MIN_GROUND_SECONDS: f64 = 1.0;

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

    let camera_pose = |timestamp_ns: i64, camera_id: u16| -> Option<Isometry3<f64>> {
        let pose = pose_at(&poses, timestamp_ns)?;
        if args.trajectory_is_camera {
            return Some(*pose);
        }
        let t_i_c = calibration.camera_to_imu(camera_id)?;
        Some(pose * Isometry3::from_parts(Translation3::from(t_i_c.translation), t_i_c.rotation))
    };

    // Bearing observations per track, from every camera, and the time each
    // track was seen.
    let mut per_track: BTreeMap<u64, (Vec<BearingObservation>, Vec<i64>)> = BTreeMap::new();
    for &(timestamp_ns, camera_id, track_id, pixel) in &observations {
        let (Some(camera_to_world), Some(camera)) = (
            camera_pose(timestamp_ns, camera_id),
            calibration.camera(camera_id),
        ) else {
            continue;
        };
        let Some(bearing) = camera.unproject(&pixel) else {
            continue;
        };
        let (bearings, times) = per_track.entry(track_id).or_default();
        bearings.push(BearingObservation {
            camera_to_world,
            bearing,
        });
        times.push(timestamp_ns);
    }
    let mut per_object: BTreeMap<u64, BTreeMap<u64, Vec<BearingObservation>>> = BTreeMap::new();
    for (track_id, (bearings, _)) in &per_track {
        if let Some(&object_id) = owner.get(track_id) {
            per_object
                .entry(object_id)
                .or_default()
                .insert(*track_id, bearings.clone());
        }
    }

    // First estimate each 2D track's object, then merge objects of the same
    // class whose centres are within MERGE_DISTANCE: the 2D tracker loses an
    // object when it leaves the view and gives it a new ID when it comes back.
    let estimate = |tracks: &BTreeMap<u64, Vec<BearingObservation>>, object_id: u64, min_points| {
        let tracks: Vec<Vec<BearingObservation>> = tracks.values().cloned().collect();
        estimate_static_object(
            object_id,
            classes[&object_id],
            &tracks,
            &TriangulationParams::default(),
            &BoxFitParams {
                min_points,
                ..BoxFitParams::default()
            },
        )
    };
    let first: Vec<(u64, ObjectEstimate)> = per_object
        .iter()
        .filter_map(|(id, tracks)| {
            Some((
                *id,
                estimate(tracks, *id, BoxFitParams::default().min_points)?,
            ))
        })
        .collect();
    let mut group: Vec<usize> = (0..first.len()).collect();
    fn root(group: &mut [usize], i: usize) -> usize {
        if group[i] != i {
            group[i] = root(group, group[i]);
        }
        group[i]
    }
    for i in 0..first.len() {
        for j in i + 1..first.len() {
            let (a, b) = (&first[i].1, &first[j].1);
            if a.class_id == b.class_id
                && (a.upright_box.center - b.upright_box.center).norm() < MERGE_DISTANCE
            {
                let (ri, rj) = (root(&mut group, i), root(&mut group, j));
                group[ri.max(rj)] = ri.min(rj);
            }
        }
    }
    let mut merged: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for (i, (object_id, _)) in first.iter().enumerate() {
        let r = root(&mut group, i);
        merged.entry(r).or_default().push(*object_id);
    }

    let mut csv = String::from(
        "object_id,merged_ids,class_id,class,tracks,points,center_x,center_y,center_z,length,width,height,yaw,scale_source,plausible\n",
    );
    println!(
        "{:>6} {:<14} {:>6} {:>6} {:>6}  {:>22}  {:>24}  plausible",
        "object", "class", "merged", "tracks", "points", "size l x w x h (m)", "centre (m)"
    );
    for ids in merged.values() {
        let object_id = ids[0];
        let class_id = classes[&object_id];
        let mut tracks = BTreeMap::new();
        for id in ids {
            tracks.extend(per_object[id].iter().map(|(t, o)| (*t, o.clone())));
        }
        let Some(estimate) = estimate(&tracks, object_id, MIN_POINTS) else {
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
            "{object_id:>6} {class:<14} {:>6} {:>6} {:>6}  {:>6.2} x {:>5.2} x {:>5.2}  ({:>6.2}, {:>6.2}, {:>6.2})  {plausible}",
            ids.len(),
            tracks.len(),
            estimate.points,
            b.length,
            b.width,
            b.height,
            b.center.x,
            b.center.y,
            b.center.z
        );
        let merged_ids = ids.iter().map(u64::to_string).collect::<Vec<_>>().join(" ");
        writeln!(
            csv,
            "{object_id},{merged_ids},{class_id},{class},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},triangulated,{plausible}",
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
        .clone()
        .unwrap_or_else(|| args.boxes.with_file_name("objects.csv"));
    fs::write(&out, csv)?;
    println!("objects: {}", out.display());

    // The scene: every track that triangulates, at the median time it was seen.
    let scene: Vec<(i64, Point3<f64>)> = per_track
        .values()
        .filter_map(|(bearings, times)| {
            let point = triangulate(bearings, &TriangulationParams::default())?;
            let mut times = times.clone();
            times.sort_unstable();
            Some((times[times.len() / 2], point))
        })
        .collect();
    let out = out.with_file_name("ground_objects.csv");
    let (csv, frames) = ground_objects(&boxes, &scene, &poses, &calibration, &camera_pose)?;
    fs::write(&out, csv)?;
    println!("objects on the ground: {}", out.display());
    let out = out.with_file_name("ground_tracks.csv");
    fs::write(&out, frames)?;
    println!("their feet, frame by frame: {}", out.display());
    Ok(())
}

/// Place each movable object's camera-0 boxes on the ground and summarize
/// them; returns the CSV.
fn ground_objects(
    boxes: &HashMap<i64, Vec<ObjectBox>>,
    scene: &[(i64, Point3<f64>)],
    poses: &[(i64, Isometry3<f64>)],
    calibration: &visloc_basalt::BasaltCalibration,
    camera_pose: &dyn Fn(i64, u16) -> Option<Isometry3<f64>>,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let camera = calibration.camera(0).ok_or("no camera 0")?;
    let (width, height) = calibration.resolutions[0];
    let (width, height) = (f64::from(width), f64::from(height));
    let mut per_object: BTreeMap<u64, (u32, Vec<(i64, BBox)>)> = BTreeMap::new();
    for (&timestamp_ns, frame) in boxes {
        for b in frame {
            let inside = b.x_min > BORDER_PX
                && b.y_min > BORDER_PX
                && b.x_max < width - BORDER_PX
                && b.y_max < height - BORDER_PX;
            if MOVABLE_COCO_CLASSES.contains(&b.class_id) && b.score >= MIN_GROUND_SCORE && inside {
                let entry = per_object
                    .entry(b.object_id)
                    .or_insert((b.class_id, Vec::new()));
                entry
                    .1
                    .push((timestamp_ns, BBox::new(b.x_min, b.y_min, b.x_max, b.y_max)));
            }
        }
    }
    let mut csv = String::from(
        "object_id,class_id,class,frames,seconds,camera_height,ground_shift,x,y,z,height,width,distance,speed,speed_sigma,moving,scale_source,plausible\n",
    );
    let mut frames_csv = String::from(
        "object_id,class_id,timestamp_ns,x,y,z,camera_x,camera_y,camera_z,height,width,distance\n",
    );
    println!(
        "\n{:>6} {:<12} {:>6} {:>7}  {:>8}  {:>6}  {:>13}  {:>8}  {:>7}  plausible",
        "object",
        "class",
        "frames",
        "seconds",
        "camera h",
        "shift",
        "size h x w (m)",
        "distance",
        "speed"
    );
    for (object_id, (class_id, mut frames)) in per_object {
        frames.sort_by_key(|(t, _)| *t);
        let (t0, t1) = (frames[0].0, frames[frames.len() - 1].0);
        let window = (t0 - GROUND_WINDOW_NS)..=(t1 + GROUND_WINDOW_NS);
        let points: Vec<Point3<f64>> = scene
            .iter()
            .filter(|(t, _)| window.contains(t))
            .map(|(_, p)| *p)
            .collect();
        let path: Vec<Point3<f64>> = poses
            .iter()
            .filter(|(t, _)| window.contains(t))
            .step_by(5)
            .filter_map(|(t, _)| camera_pose(*t, 0))
            .map(|pose| Point3::from(pose.translation.vector))
            .collect();
        let params = GroundParams::default();
        let Some(path_ground) = PathGround::estimate(&path, &points, &params) else {
            continue;
        };
        let views: Vec<GroundView> = frames
            .iter()
            .filter_map(|(t, bbox)| {
                let pose = camera_pose(*t, 0)?;
                let rays = BoxRays::from_box(bbox, &pose.rotation, |x, y| {
                    camera.unproject(&Point2::new(x, y))
                })?;
                Some(GroundView {
                    timestamp_ns: *t,
                    camera_center: Point3::from(pose.translation.vector),
                    rays,
                })
            })
            .collect();
        // The ground under the camera first, then under the object's foot.
        let Some(middle) = views.get(views.len() / 2) else {
            continue;
        };
        let xy = |p: &Point3<f64>| Vector2::new(p.x, p.y);
        let Some((mut ground, mut placed)) = place_on_local_ground(
            &views,
            &points,
            path_ground.height_at(&xy(&middle.camera_center)),
            &params,
        ) else {
            continue;
        };
        let mut feet: Vec<Vector2<f64>> = placed.iter().map(|o| xy(&o.contact)).collect();
        feet.sort_by(|a, b| a.x.total_cmp(&b.x));
        let path_at_foot = path_ground.height_at(&feet[feet.len() / 2]);
        if (path_at_foot - ground).abs() > 0.05 {
            if let Some(again) = place_on_local_ground(&views, &points, path_at_foot, &params) {
                (ground, placed) = again;
            }
        }
        let mut camera_heights: Vec<f64> =
            views.iter().map(|v| v.camera_center.z - ground).collect();
        let Some(track) = summarize_ground_track(
            object_id,
            class_id,
            &placed,
            &GroundTrackParams {
                min_seconds: MIN_GROUND_SECONDS,
                ..GroundTrackParams::default()
            },
        ) else {
            continue;
        };
        for o in &placed {
            writeln!(
                frames_csv,
                "{object_id},{class_id},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2}",
                o.timestamp_ns,
                o.contact.x,
                o.contact.y,
                o.contact.z,
                o.camera.x,
                o.camera.y,
                o.camera.z,
                o.height,
                o.width,
                o.distance
            )?;
        }
        // Where it stands: the median of its contact points.
        let median = |axis: usize| {
            let mut v: Vec<f64> = track.path.iter().map(|(_, p)| p[axis]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let position = Point3::new(median(0), median(1), median(2));
        camera_heights.sort_by(f64::total_cmp);
        let camera_height = camera_heights[camera_heights.len() / 2];
        let seconds = (t1 - t0) as f64 * 1e-9;
        let shift = ground - path_at_foot;
        let plausible = match plausible_height(class_id) {
            Some((low, high)) if (low..=high).contains(&track.height) => "yes",
            Some(_) => "NO",
            None => "-",
        };
        let class = COCO_CLASSES.get(class_id as usize).copied().unwrap_or("?");
        println!(
            "{object_id:>6} {class:<12} {:>6} {seconds:>7.1}  {camera_height:>8.2}  {shift:>+6.2}  {:>5.2} x {:>5.2}  {:>8.1}  {:>7.2}  {plausible}",
            placed.len(),
            track.height,
            track.width,
            track.distance,
            track.speed
        );
        writeln!(
            csv,
            "{object_id},{class_id},{class},{},{seconds:.2},{camera_height:.3},{shift:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2},{:.3},{:.3},{},ground_contact,{plausible}",
            placed.len(),
            position.x,
            position.y,
            position.z,
            track.height,
            track.width,
            track.distance,
            track.speed,
            track.speed_sigma,
            track.moving
        )?;
    }
    Ok((csv, frames_csv))
}

/// The pose nearest to `timestamp_ns`, if one is within 1 ms. The demo
/// writes the TUM time in seconds through an f64, which is off by a few
/// nanoseconds at epoch scale, so an exact match would miss.
fn pose_at(poses: &[(i64, Isometry3<f64>)], timestamp_ns: i64) -> Option<&Isometry3<f64>> {
    let i = poses.partition_point(|(t, _)| *t < timestamp_ns);
    [i.checked_sub(1), Some(i)]
        .into_iter()
        .flatten()
        .filter_map(|j| poses.get(j))
        .min_by_key(|(t, _)| (t - timestamp_ns).abs())
        .filter(|(t, _)| (t - timestamp_ns).abs() <= 1_000_000)
        .map(|(_, pose)| pose)
}

/// TUM lines `t tx ty tz qx qy qz qw`, `t` in seconds with up to nine
/// decimals, sorted by time.
fn read_trajectory(path: &PathBuf) -> Result<Vec<(i64, Isometry3<f64>)>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut poses = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let bad = || format!("{}: line {}", path.display(), index + 1);
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() != 8 {
            return Err(bad());
        }
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
        poses.push((
            timestamp_ns,
            Isometry3::from_parts(Translation3::from(Vector3::new(v[0], v[1], v[2])), rotation),
        ));
    }
    poses.sort_by_key(|(t, _)| *t);
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
                score: n(3)?,
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
