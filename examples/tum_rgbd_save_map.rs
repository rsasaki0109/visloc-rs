//! Save a localization map from a TUM RGB-D sequence and a trajectory of it,
//! in the layout `basalt_euroc_online_slam_demo --save-map` writes, so
//! `basalt_map_merge` and `basalt_euroc_localize_demo` read it unchanged.
//!
//! ```text
//! cargo run --release --example tum_rgbd_save_map -- \
//!   --seq-dir rgbd_dataset_freiburg1_room --trajectory runs/fr1_room_vo/est.tum \
//!   --out runs/fr1_room_map
//! ```
//!
//! The trajectory gives the camera's poses (TUM format: world <- camera, the
//! RGB camera's optical frame) in its own world frame, e.g. the RGB-D VO of
//! `scripts/run_tum_rgbd_benchmark.sh`. No ground truth is read.
//!
//! - **Gravity.** `basalt_map_merge` registers maps by a yaw and a
//!   translation, so each map's world must have z up. The Kinect's
//!   accelerometer (`accelerometer.txt`) gives it: each sample is turned
//!   into the camera's optical frame (the accelerometer's x and y point the
//!   other way: a half turn about the optical axis), then into the world
//!   by the trajectory's rotation at its time, and averaged over the whole
//!   sequence, where the motion's own accelerations cancel. The map's world
//!   is the trajectory's, rotated so this mean points up.
//!   Sequences without accelerometer samples (Freiburg 3) are levelled by
//!   the map's largest horizontal plane instead (a floor or a desk top): a
//!   RANSAC over the landmarks, among planes within 30 degrees of the
//!   cameras' mean up, refitted to its inliers. On fr1_room it is 1.4
//!   degrees from the ground truth's vertical, the accelerometer 1.8.
//! - **Keyframes.** The first frame, then each frame that has moved
//!   `--keyframe-m` or turned `--keyframe-deg` from the last keyframe.
//! - **Features.** The Basalt mapper's ORB on each keyframe's grey image:
//!   Shi-Tomasi corners (`--max-points`), angles, 256-bit descriptors, so
//!   the map matches what the localization demo extracts.
//! - **Landmarks.** A corner whose registered depth (`--min-depth` ..
//!   `--max-depth`) puts it in front of the camera is matched to the
//!   landmarks of the last `--track-keyframes` keyframes that project within
//!   `--match-radius-px` of it with a descriptor within `--max-hamming`
//!   bits; it observes the nearest, or starts a landmark at its depth.
//!   Landmarks seen by fewer than `--min-observations` keyframes are
//!   dropped, as the SLAM's map export drops single-view ones. A landmark's
//!   descriptor is its observations' medoid.
//!
//! The camera is Freiburg 1's (FULL_OPENCV, `--fx` ... `--k3` to override;
//! Freiburg 3's images are already undistorted, so pass zero coefficients).

#[path = "basalt_map/map_files.rs"]
mod map_files;

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3, Quaternion, Rotation3, UnitQuaternion, Vector3};
use serde_json::json;
use visloc_basalt::mapper::features::{
    compute_angles, compute_descriptors, detect_keypoints_mapping_with,
};
use visloc_basalt::mapper::localization_map::{localization_camera_id, orb_descriptor_to_f32};
use visloc_basalt::pyramid::RawU16Image;
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{
    Camera, CameraModel, Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation, VisualMap,
};

#[derive(Debug, Clone)]
struct Args {
    seq_dir: PathBuf,
    trajectory: PathBuf,
    out: PathBuf,
    intrinsics: [f64; 9],
    keyframe_m: f64,
    keyframe_deg: f64,
    max_points: usize,
    min_depth: f64,
    max_depth: f64,
    depth_scale: f64,
    track_keyframes: usize,
    match_radius_px: f64,
    max_hamming: u32,
    min_observations: usize,
}

impl Args {
    fn usage() -> String {
        "usage: tum_rgbd_save_map --seq-dir DIR --trajectory FILE --out DIR \
         [--keyframe-m M] [--keyframe-deg D] [--max-points N] [--min-depth M] \
         [--max-depth M] [--depth-scale S] [--track-keyframes N] [--match-radius-px P] \
         [--max-hamming N] [--min-observations N] [--fx F --fy F --cx C --cy C \
         --k1 K --k2 K --p1 P --p2 P --k3 K]"
            .into()
    }

    fn parse() -> Result<Self, String> {
        let mut seq_dir = None;
        let mut trajectory = None;
        let mut out = None;
        // Freiburg 1's RGB camera (the TUM RGB-D calibration page).
        let mut intrinsics = [
            517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633,
        ];
        let mut args = Self {
            seq_dir: PathBuf::new(),
            trajectory: PathBuf::new(),
            out: PathBuf::new(),
            intrinsics,
            keyframe_m: 0.10,
            keyframe_deg: 8.0,
            max_points: 1000,
            min_depth: 0.3,
            max_depth: 4.0,
            depth_scale: 1.0 / 5000.0,
            track_keyframes: 5,
            match_radius_px: 4.0,
            max_hamming: 60,
            min_observations: 2,
        };
        let mut arguments = std::env::args().skip(1);
        while let Some(option) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .ok_or_else(|| format!("{option} requires a value"))
            };
            fn number<T: std::str::FromStr>(option: &str, value: String) -> Result<T, String> {
                value
                    .parse()
                    .map_err(|_| format!("{option}: invalid value {value:?}"))
            }
            let intrinsic = |name: &str| {
                [
                    "--fx", "--fy", "--cx", "--cy", "--k1", "--k2", "--p1", "--p2", "--k3",
                ]
                .iter()
                .position(|candidate| *candidate == name)
            };
            match option.as_str() {
                "--help" | "-h" => return Err(Self::usage()),
                "--seq-dir" => seq_dir = Some(PathBuf::from(value()?)),
                "--trajectory" => trajectory = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--keyframe-m" => args.keyframe_m = number(&option, value()?)?,
                "--keyframe-deg" => args.keyframe_deg = number(&option, value()?)?,
                "--max-points" => args.max_points = number(&option, value()?)?,
                "--min-depth" => args.min_depth = number(&option, value()?)?,
                "--max-depth" => args.max_depth = number(&option, value()?)?,
                "--depth-scale" => args.depth_scale = number(&option, value()?)?,
                "--track-keyframes" => args.track_keyframes = number(&option, value()?)?,
                "--match-radius-px" => args.match_radius_px = number(&option, value()?)?,
                "--max-hamming" => args.max_hamming = number(&option, value()?)?,
                "--min-observations" => args.min_observations = number(&option, value()?)?,
                name if intrinsic(name).is_some() => {
                    intrinsics[intrinsic(name).unwrap()] = number(&option, value()?)?;
                }
                _ => return Err(format!("unknown option {option}\n{}", Self::usage())),
            }
        }
        args.seq_dir = seq_dir.ok_or_else(Self::usage)?;
        args.trajectory = trajectory.ok_or_else(Self::usage)?;
        args.out = out.ok_or_else(Self::usage)?;
        args.intrinsics = intrinsics;
        Ok(args)
    }
}

/// `timestamp file` rows of a TUM list (`rgb.txt`, `depth.txt`).
fn read_list(path: &Path) -> Result<Vec<(f64, String)>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            Some((fields.next()?.parse().ok()?, fields.next()?.to_string()))
        })
        .collect())
}

/// Rows of numbers of a TUM text file (`accelerometer.txt`, a trajectory).
fn read_rows(path: &Path, width: usize) -> Result<Vec<Vec<f64>>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(text
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.split_whitespace()
                .filter_map(|value| value.parse().ok())
                .collect::<Vec<f64>>()
        })
        .filter(|row| row.len() >= width)
        .collect())
}

fn nearest(stamps: &[f64], t: f64) -> usize {
    let i = stamps.partition_point(|&s| s < t).min(stamps.len() - 1);
    if i > 0 && (stamps[i - 1] - t).abs() < (stamps[i] - t).abs() {
        i - 1
    } else {
        i
    }
}

/// The trajectory's rotation and position at `t`, linearly interpolated.
fn interpolate(stamps: &[f64], poses: &[SE3], t: f64) -> Option<SE3> {
    let i = stamps.partition_point(|&s| s < t);
    if i == 0 || i >= stamps.len() {
        return None;
    }
    let u = (t - stamps[i - 1]) / (stamps[i] - stamps[i - 1]);
    let (a, b) = (&poses[i - 1], &poses[i]);
    Some(SE3::new(
        a.rotation.slerp(&b.rotation, u),
        a.translation.lerp(&b.translation, u),
    ))
}

/// The rotation taking `up` to +z.
fn levelling(up: &Vector3<f64>) -> UnitQuaternion<f64> {
    Rotation3::rotation_between(up, &Vector3::z())
        .map(|rotation| UnitQuaternion::from_rotation_matrix(&rotation))
        .unwrap_or_else(UnitQuaternion::identity)
}

/// The normal (towards `camera_up`) and inlier count of the plane through
/// most `points` within `tolerance` metres, among planes whose normal lies
/// within `max_tilt` of `camera_up`: a RANSAC over point triples
/// (deterministic), refitted to its inliers by least squares.
fn horizontal_plane(
    points: &[Vector3<f64>],
    camera_up: &Vector3<f64>,
    tolerance: f64,
    max_tilt: f64,
) -> Option<(Vector3<f64>, usize)> {
    if points.len() < 3 {
        return None;
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % points.len() as u64) as usize
    };
    let inliers_of = |normal: &Vector3<f64>, offset: f64| {
        points
            .iter()
            .filter(|point| (normal.dot(point) - offset).abs() < tolerance)
            .collect::<Vec<_>>()
    };
    let mut best: Option<(Vector3<f64>, f64, usize)> = None;
    for _ in 0..2000 {
        let (a, b, c) = (points[next()], points[next()], points[next()]);
        let Some(mut normal) = (b - a).cross(&(c - a)).try_normalize(1e-9) else {
            continue;
        };
        if normal.dot(camera_up) < 0.0 {
            normal = -normal;
        }
        if normal.angle(camera_up) > max_tilt {
            continue;
        }
        let count = inliers_of(&normal, normal.dot(&a)).len();
        if best.is_none_or(|(_, _, best_count)| count > best_count) {
            best = Some((normal, normal.dot(&a), count));
        }
    }
    let (normal, offset, _) = best?;
    // Refit: the smallest principal axis of the inliers.
    let inliers = inliers_of(&normal, offset);
    let mean = inliers.iter().copied().sum::<Vector3<f64>>() / inliers.len() as f64;
    let covariance = inliers
        .iter()
        .map(|point| (*point - mean) * (*point - mean).transpose())
        .sum::<nalgebra::Matrix3<f64>>();
    let eigen = covariance.symmetric_eigen();
    let smallest = eigen.eigenvalues.imin();
    let mut refitted = eigen.eigenvectors.column(smallest).into_owned();
    if refitted.dot(camera_up) < 0.0 {
        refitted = -refitted;
    }
    Some((refitted, inliers.len()))
}

fn hamming(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn grey_image(path: &Path) -> Result<RawU16Image, String> {
    let image = image::open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .to_luma8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let pixels = image
        .into_raw()
        .into_iter()
        .map(|v| u16::from(v) << 8)
        .collect();
    RawU16Image::new(width, height, pixels)
        .map_err(|error| format!("{}: {error:?}", path.display()))
}

fn depth_image(path: &Path) -> Result<(usize, Vec<u16>), String> {
    let image = image::open(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .to_luma16();
    Ok((image.width() as usize, image.into_raw()))
}

struct Track {
    position: Point3<f64>,
    descriptors: Vec<[u8; 32]>,
    observations: Vec<Observation>,
    last_keyframe: usize,
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let rgb = read_list(&args.seq_dir.join("rgb.txt"))?;
    let depth = read_list(&args.seq_dir.join("depth.txt"))?;
    let depth_stamps = depth.iter().map(|(t, _)| *t).collect::<Vec<_>>();
    let rows = read_rows(&args.trajectory, 8)?;
    let stamps = rows.iter().map(|row| row[0]).collect::<Vec<_>>();
    let poses = rows
        .iter()
        .map(|row| {
            SE3::new(
                UnitQuaternion::from_quaternion(Quaternion::new(row[7], row[4], row[5], row[6])),
                Vector3::new(row[1], row[2], row[3]),
            )
        })
        .collect::<Vec<_>>();
    if poses.len() < 2 {
        return Err(format!("{}: fewer than two poses", args.trajectory.display()).into());
    }

    // Gravity: the accelerometer's mean specific force, in the world.
    let accel_path = args.seq_dir.join("accelerometer.txt");
    let accel = if accel_path.exists() {
        read_rows(&accel_path, 4)?
    } else {
        Vec::new()
    };
    let mut up = Vector3::zeros();
    let mut samples = 0;
    for row in &accel {
        if let Some(world_from_camera) = interpolate(&stamps, &poses, row[0]) {
            // Accelerometer x and y point opposite the optical frame's.
            let in_camera = Vector3::new(-row[1], -row[2], row[3]);
            up += world_from_camera.rotation * in_camera;
            samples += 1;
        }
    }
    // Without accelerometer samples (Freiburg 3 has none), the map is
    // levelled after it is built, by its largest horizontal plane.
    let by_accelerometer = samples > 0;
    let aligned = if by_accelerometer {
        let g = (up / samples as f64).norm();
        let up = up.normalize();
        println!(
            "gravity: {samples} accelerometer samples, mean {g:.2} m/s^2, tilt {:.1} deg",
            up.angle(&Vector3::z()).to_degrees()
        );
        SE3::new(levelling(&up), Vector3::zeros())
    } else {
        SE3::identity()
    };

    let [fx, fy, cx, cy, k1, k2, p1, p2, k3] = args.intrinsics;
    let camera_id = localization_camera_id(0);
    let first = grey_image(&args.seq_dir.join(&rgb[0].1))?;
    let camera = Camera {
        id: camera_id,
        model: CameraModel::FullOpenCv,
        width: first.width() as u32,
        height: first.height() as u32,
        params: vec![fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0],
    };

    let mut map = VisualMap::new();
    map.cameras.insert(camera_id, camera.clone());
    let mut images = Vec::new();
    let mut tracks: BTreeMap<u64, Track> = BTreeMap::new();
    let mut next_landmark = 1_u64;
    let mut last_keyframe_pose: Option<SE3> = None;
    let mut keyframe_index = 0_usize;

    for (frame_id, (t, file)) in rgb.iter().enumerate() {
        let Some(world_from_camera) = interpolate(&stamps, &poses, *t) else {
            continue;
        };
        let world_from_camera = aligned.compose(&world_from_camera);
        if let Some(last) = &last_keyframe_pose {
            let delta = last.inverse().compose(&world_from_camera);
            if delta.translation.norm() < args.keyframe_m
                && delta.rotation.angle().to_degrees() < args.keyframe_deg
            {
                continue;
            }
        }
        let d = nearest(&depth_stamps, *t);
        if (depth_stamps[d] - t).abs() > 0.02 {
            continue;
        }
        let grey = grey_image(&args.seq_dir.join(file))?;
        let (depth_width, depth_pixels) = depth_image(&args.seq_dir.join(&depth[d].1))?;
        let quality = map_files::Frontend::default().quality_level();
        let corners = detect_keypoints_mapping_with(&grey, args.max_points, quality);
        let angles = compute_angles(&grey, &corners, true);
        let descriptors = compute_descriptors(&grey, &corners, &angles)?;
        last_keyframe_pose = Some(world_from_camera.clone());
        keyframe_index += 1;
        let image_id = keyframe_index as u64;
        let world_to_camera = world_from_camera.inverse();

        // Landmarks of the recent keyframes, projected into this one.
        let recent = tracks
            .iter()
            .filter(|(_, track)| track.last_keyframe + args.track_keyframes >= keyframe_index)
            .filter_map(|(&id, track)| {
                let point = world_to_camera.transform_point(&track.position);
                (point.z > 0.0)
                    .then(|| camera.project(&point))
                    .flatten()
                    .map(|xy| (id, xy, point.z))
            })
            .collect::<Vec<_>>();
        let cell = args.match_radius_px.max(1.0);
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (index, (_, xy, _)) in recent.iter().enumerate() {
            grid.entry(((xy.x / cell) as i64, (xy.y / cell) as i64))
                .or_default()
                .push(index);
        }
        let mut used = vec![false; recent.len()];

        let mut frame = Frame::new(image_id, camera_id);
        frame.keypoints = corners.clone();
        frame.pose = Some(Pose {
            world_to_camera: world_to_camera.clone(),
        });
        for (keypoint_index, (corner, descriptor)) in corners.iter().zip(&descriptors).enumerate() {
            let (u, v) = (corner.x.round() as usize, corner.y.round() as usize);
            let raw = depth_pixels.get(v * depth_width + u).copied().unwrap_or(0);
            let z = f64::from(raw) * args.depth_scale;
            if !(args.min_depth..=args.max_depth).contains(&z) {
                continue;
            }
            let Some(normalized) = camera.normalize_pixel(corner) else {
                continue;
            };
            // The nearest recent landmark that projects close by, looks
            // alike and lies at about this depth.
            let (gx, gy) = ((corner.x / cell) as i64, (corner.y / cell) as i64);
            let mut best: Option<(usize, f64)> = None;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for &index in grid.get(&(gx + dx, gy + dy)).into_iter().flatten() {
                        let (id, xy, depth_there) = &recent[index];
                        let distance = (xy - corner).norm();
                        if used[index]
                            || distance > args.match_radius_px
                            || (depth_there - z).abs() > 0.05 * z
                            || hamming(&tracks[id].descriptors[0], descriptor) > args.max_hamming
                        {
                            continue;
                        }
                        if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                            best = Some((index, distance));
                        }
                    }
                }
            }
            let landmark_id = match best {
                Some((index, _)) => {
                    used[index] = true;
                    recent[index].0
                }
                None => {
                    let id = next_landmark;
                    next_landmark += 1;
                    let in_camera = Point3::new(normalized.x * z, normalized.y * z, z);
                    tracks.insert(
                        id,
                        Track {
                            position: world_from_camera.transform_point(&in_camera),
                            descriptors: Vec::new(),
                            observations: Vec::new(),
                            last_keyframe: keyframe_index,
                        },
                    );
                    id
                }
            };
            let track = tracks.get_mut(&landmark_id).expect("track exists");
            track.descriptors.push(*descriptor);
            track.last_keyframe = keyframe_index;
            track.observations.push(Observation {
                frame_id: image_id,
                landmark_id,
                keypoint_index,
                xy: Point2::new(corner.x, corner.y),
            });
        }
        map.keyframes.insert(
            image_id,
            Keyframe {
                frame,
                observations: Vec::new(),
            },
        );
        images.push(map_files::ImageRow {
            image_id,
            frame_id: frame_id as u64,
            cam_id: 0,
            timestamp_ns: Some((t * 1e9).round() as i64),
            session: 0,
        });
    }

    let mut store = LandmarkDescriptorStore::new();
    for (id, track) in tracks {
        if track.observations.len() < args.min_observations {
            continue;
        }
        let medoid = track
            .descriptors
            .iter()
            .min_by_key(|candidate| {
                track
                    .descriptors
                    .iter()
                    .map(|other| hamming(candidate, other))
                    .sum::<u32>()
            })
            .expect("a track has a descriptor");
        store.insert(id, orb_descriptor_to_f32(medoid));
        for observation in &track.observations {
            if let Some(keyframe) = map.keyframes.get_mut(&observation.frame_id) {
                keyframe.observations.push(observation.clone());
            }
        }
        let mut landmark = Landmark::new(id, track.position);
        landmark.observations = track.observations;
        map.landmarks.insert(id, landmark);
    }

    let aligned = if by_accelerometer {
        aligned
    } else {
        // The cameras' mean up (optical -y) picks which planes count as
        // horizontal: within 30 degrees of it.
        let camera_up = map
            .keyframes
            .values()
            .filter_map(|keyframe| keyframe.frame.pose.as_ref())
            .map(|pose| pose.camera_to_world().rotation * -Vector3::y())
            .sum::<Vector3<f64>>()
            .normalize();
        let points = map
            .landmarks
            .values()
            .map(|landmark| landmark.position.coords)
            .collect::<Vec<_>>();
        let (up, inliers) = horizontal_plane(&points, &camera_up, 0.015, 30f64.to_radians())
            .ok_or("no horizontal plane found to level the map")?;
        println!(
            "level: plane of {inliers} of {} landmarks, {:.1} deg from the cameras' mean up",
            points.len(),
            up.angle(&camera_up).to_degrees()
        );
        let level = SE3::new(levelling(&up), Vector3::zeros());
        for keyframe in map.keyframes.values_mut() {
            if let Some(pose) = keyframe.frame.pose.as_mut() {
                pose.world_to_camera = pose.world_to_camera.compose(&level.inverse());
            }
        }
        for landmark in map.landmarks.values_mut() {
            landmark.position = level.transform_point(&landmark.position);
        }
        level
    };

    let mut summary = json!({
        "schema": map_files::MAP_SCHEMA,
        "world_frame": if by_accelerometer {
            "the trajectory's world, levelled by the accelerometer's gravity (z up)"
        } else {
            "the trajectory's world, levelled by its largest horizontal plane (z up)"
        },
        "feature_max_points": args.max_points,
        "cameras": [{
            "camera_id": camera_id,
            "basalt_camera": 0,
            "model": "FULL_OPENCV",
            "params": camera.params,
            "t_imu_cam": map_files::se3_json(&SE3::identity()),
        }],
        "source": {
            "sequence": args.seq_dir.display().to_string(),
            "trajectory": args.trajectory.display().to_string(),
            "levelling": map_files::se3_json(&aligned),
        },
    });
    map_files::Frontend::default().write_json(&mut summary);
    let summary = map_files::write(&args.out, &map, &store, &images, summary)?;
    println!(
        "wrote {}: {} keyframe images, {} landmarks, {} observations",
        args.out.display(),
        summary["images"],
        summary["landmarks"],
        summary["observations"]
    );
    Ok(())
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&args) {
        eprintln!("tum_rgbd_save_map: {error}");
        std::process::exit(1);
    }
}
