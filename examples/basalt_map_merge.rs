//! Merge maps saved by `basalt_euroc_online_slam_demo --save-map` from separate
//! sessions of one place into a single map, in the first map's world frame.
//!
//! ```text
//! cargo run --release --example basalt_map_merge -- \
//!   --base runs/mh01_map --add runs/mh04_map --out runs/mh01_mh04_map
//! ```
//!
//! Each `--add` map is registered against the merged map so far by
//! localizing its keyframes in it
//! ([`register_map`](visloc_rs::localization::register_map): a gravity-aligned
//! yaw and translation, agreed on by several keyframes), then merged into it
//! ([`merge_maps`](visloc_rs::localization::merge_maps)): its keyframes and
//! landmarks are moved into the base frame and renumbered, and its landmarks
//! that coincide with existing ones are fused into them. All maps must come
//! from the same camera rig. Another calibration of it (another day's) is
//! kept: its cameras get new ids in the merged COLMAP model, and its
//! session in `map.json` records its `t_imu_cam`.
//!
//! The merged map keeps the base map's world frame, so poses localized
//! against it, and anything aligned to the base map's SLAM trajectory, stay
//! valid. `map.json` lists the sessions with the transform from each session's
//! world and how it was registered; `images.csv` says which session each image
//! came from. A merged map can be the `--base` (or an `--add`) of another
//! merge.
//!
//! With `--inspect-loops`, inspect registration between the original maps
//! and write `loop_inspection.json` to the output directory without merging.
//! With `--optimize-sessions`, jointly align the original maps in the base
//! session's frame before fusing their landmarks.

#[path = "basalt_map/global_alignment.rs"]
mod global_alignment;
#[path = "basalt_map/keyframe_alignment.rs"]
mod keyframe_alignment;
#[path = "basalt_map/loop_inspection.rs"]
mod loop_inspection;
#[path = "basalt_map/map_files.rs"]
mod map_files;

use std::io;
use std::iter;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};
use visloc_core::geometry::SE3;
use visloc_rs::localization::{merge_maps, register_map, MapMergeConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Sequential,
    Inspect,
    Optimize,
}

#[derive(Debug, Clone, PartialEq)]
struct Args {
    base: PathBuf,
    add: Vec<PathBuf>,
    out: PathBuf,
    config: MapMergeConfig,
    mode: Mode,
    refine_keyframes: bool,
    keyframe_loop_min_separation_m: Option<f64>,
    moving_boxes: Vec<(PathBuf, PathBuf)>,
}

impl Args {
    fn usage() -> String {
        "usage: basalt_map_merge --base DIR --add DIR [--add DIR ...] --out DIR \
         [--min-inliers N] [--min-keyframes N] [--max-hamming N] \
         [--fuse-radius-m M] [--fuse-max-hamming N] [--threads N] \
         [--retrieval-keyframes N] [--retrieval-bits B] [--duplicate-radius-m M] \
         [--inspect-loops | --optimize-sessions [--refine-keyframes [--keyframe-loop-min-separation-m M]] [--moving-boxes MAP_DIR CSV]...]"
            .into()
    }

    fn parse<I>(arguments: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut base = None;
        let mut out = None;
        let mut args = Self {
            base: PathBuf::new(),
            add: Vec::new(),
            out: PathBuf::new(),
            mode: Mode::Sequential,
            refine_keyframes: false,
            keyframe_loop_min_separation_m: None,
            moving_boxes: Vec::new(),
            config: MapMergeConfig {
                threads: std::thread::available_parallelism().map_or(1, usize::from),
                ..MapMergeConfig::default()
            },
        };
        let mut arguments = arguments.into_iter();
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
            let config = &mut args.config;
            match option.as_str() {
                "--help" | "-h" => return Err(Self::usage()),
                "--base" => base = Some(PathBuf::from(value()?)),
                "--add" => args.add.push(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--inspect-loops" if args.mode != Mode::Optimize => args.mode = Mode::Inspect,
                "--optimize-sessions" if args.mode != Mode::Inspect => args.mode = Mode::Optimize,
                "--refine-keyframes" => args.refine_keyframes = true,
                "--keyframe-loop-min-separation-m" => {
                    let meters: f64 = number(&option, value()?)?;
                    if !meters.is_finite() || meters <= 0.0 {
                        return Err(format!("{option} must be finite and positive"));
                    }
                    args.keyframe_loop_min_separation_m = Some(meters);
                }
                "--moving-boxes" => args.moving_boxes.push((value()?.into(), value()?.into())),
                "--inspect-loops" | "--optimize-sessions" => {
                    return Err("choose either --inspect-loops or --optimize-sessions".into());
                }
                "--min-inliers" => config.min_inliers = number(&option, value()?)?,
                "--retrieval-keyframes" => config.retrieval_keyframes = number(&option, value()?)?,
                "--retrieval-bits" => config.retrieval_bits = number(&option, value()?)?,
                "--duplicate-radius-m" => config.duplicate_radius_m = number(&option, value()?)?,
                "--min-keyframes" => config.min_keyframes = number(&option, value()?)?,
                "--max-hamming" => config.gates.max_distance = number(&option, value()?)?,
                "--fuse-radius-m" => config.fuse_radius_m = number(&option, value()?)?,
                "--fuse-max-hamming" => config.fuse_max_hamming = number(&option, value()?)?,
                "--threads" => config.threads = number(&option, value()?)?,
                other => return Err(format!("unknown option {other}\n\n{}", Self::usage())),
            }
        }
        args.base = base.ok_or_else(|| format!("--base is required\n\n{}", Self::usage()))?;
        args.out = out.ok_or_else(|| format!("--out is required\n\n{}", Self::usage()))?;
        if args.mode != Mode::Optimize && (args.refine_keyframes || !args.moving_boxes.is_empty()) {
            return Err("--refine-keyframes and --moving-boxes require --optimize-sessions".into());
        }
        if args.keyframe_loop_min_separation_m.is_some() && !args.refine_keyframes {
            return Err("--keyframe-loop-min-separation-m requires --refine-keyframes".into());
        }
        if args.add.is_empty() {
            return Err(format!(
                "at least one --add is required\n\n{}",
                Self::usage()
            ));
        }
        Ok(args)
    }
}

/// The sessions of a loaded map: `map.json`'s list, or the map itself as the
/// only session.
fn map_sessions(map: &map_files::SavedMap, dir: &std::path::Path) -> Vec<Value> {
    map.summary
        .get("sessions")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| {
            vec![json!({
                "source": dir.display().to_string(),
                "base_from_session": map_files::se3_json(&SE3::identity()),
            })]
        })
}

fn same_rig(lhs: &[SE3], rhs: &[SE3]) -> bool {
    lhs.len() == rhs.len()
        && lhs.iter().zip(rhs).all(|(a, b)| {
            let error = a.inverse().compose(b);
            error.translation.norm() < 1e-9 && error.rotation.angle() < 1e-9
        })
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.mode == Mode::Inspect {
        return loop_inspection::run(args);
    }

    let mut additions = args.add.iter().map(PathBuf::as_path).collect::<Vec<_>>();
    let mut original_maps = Vec::new();
    let optimization = if args.mode == Mode::Optimize {
        let mut sorted = additions
            .iter()
            .map(|&path| Ok((path.canonicalize()?, path)))
            .collect::<io::Result<Vec<_>>>()?;
        sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        additions = sorted.into_iter().map(|(_, path)| path).collect();
        let paths = iter::once(args.base.as_path())
            .chain(additions.iter().copied())
            .collect::<Vec<_>>();
        if let Ok(output) = args.out.canonicalize() {
            for path in &paths {
                if path.canonicalize()? == output {
                    return Err(
                        "optimized output must be a separate directory from the input maps".into(),
                    );
                }
            }
        }
        original_maps = map_files::load_original_maps(&paths)?;
        let mut masked = std::collections::HashSet::new();
        let mut filter_reports = Vec::new();
        for (map_path, csv) in &args.moving_boxes {
            let identity = map_path.canonicalize()?;
            let index = paths
                .iter()
                .position(|path| path.canonicalize().is_ok_and(|p| p == identity))
                .ok_or_else(|| format!("{} is not an input map", map_path.display()))?;
            if !masked.insert(index) {
                return Err(format!("duplicate moving boxes for {}", map_path.display()).into());
            }
            let boxes = visloc_basalt::dynamic_objects::read_object_boxes_csv(csv)?;
            let removed = map_files::exclude_moving_tracks(&mut original_maps[index], &boxes);
            filter_reports.push(json!({"session": index, "removed_landmarks": removed}));
        }
        let (alignment, mut report) = global_alignment::solve(
            &mut original_maps,
            &args.config,
            args.refine_keyframes
                .then_some(args.keyframe_loop_min_separation_m.unwrap_or(0.35)),
        )?;
        if !filter_reports.is_empty() {
            report["moving_box_filter"] = json!(filter_reports);
        }
        Some((alignment, report))
    } else {
        None
    };
    let mut original_maps = original_maps.into_iter();
    let mut merged = original_maps
        .next()
        .map_or_else(|| map_files::load(&args.base), Ok)?;
    let mut sessions = map_sessions(&merged, &args.base);
    let mut images = merged.images.clone();
    println!(
        "base {}: {} images, {} landmarks",
        args.base.display(),
        merged.map.keyframes.len(),
        merged.map.landmarks.len()
    );

    for (addition_index, dir) in additions.into_iter().enumerate() {
        let other = original_maps
            .next()
            .map_or_else(|| map_files::load(dir), Ok)?;
        // Another calibration of the rig (another day's) merges too: its
        // cameras get new ids in the COLMAP model, and its session records
        // its own T_imu_cam (map.json's cameras stay the base map's).
        let other_rig = !same_rig(&merged.imu_from_camera, &other.imu_from_camera);
        if other_rig {
            println!(
                "{}: another calibration of the rig than the base map's",
                dir.display()
            );
        }
        // Features extracted differently (equalized or raw, blurred or not)
        // do not match each other.
        if other.frontend != merged.frontend {
            return Err(format!(
                "{}: features extracted with {:?}, the base map's with {:?}; \
                 build both the same way",
                dir.display(),
                other.frontend,
                merged.frontend
            )
            .into());
        }
        let started = Instant::now();
        let registration = if optimization.is_none() {
            Some(
                register_map(
                    &merged.map,
                    &merged.descriptors,
                    &other.map,
                    &other.descriptors,
                    &args.config,
                )
                .map_err(|error| format!("{}: not registered: {error}", dir.display()))?,
            )
        } else {
            None
        };
        let base_from_other = match (&optimization, &registration) {
            (Some((alignment, _)), _) => &alignment.reference_from_session[addition_index + 1],
            (_, Some(registration)) => &registration.base_from_other,
            _ => unreachable!("each mode provides a session transform"),
        };
        let registered_s = started.elapsed().as_secs_f64();
        let result = merge_maps(
            &merged.map,
            &merged.descriptors,
            &other.map,
            &other.descriptors,
            base_from_other,
            &args.config,
        )?;
        let added_landmarks = result.map.landmarks.len() - merged.map.landmarks.len();
        let yaw = base_from_other.rotation.euler_angles().2;
        let t = base_from_other.translation;
        if let Some(registration) = &registration {
            println!(
                "add {}: {}/{} keyframes localized, {} agree; yaw {:.3} deg, translation \
                 [{:.3}, {:.3}, {:.3}] m; keyframe residual RMS {:.1} cm; {} images, \
                 {} landmarks fused, {} added ({:.1} s)",
                dir.display(),
                registration.localized_keyframes,
                registration.keyframes,
                registration.agreeing_keyframes,
                yaw.to_degrees(),
                t.x,
                t.y,
                t.z,
                registration.keyframe_rms_m * 100.0,
                result.keyframe_ids.len(),
                result.fused_landmarks,
                added_landmarks,
                registered_s,
            );
        } else {
            println!(
                "add {}: optimized yaw {:.3} deg, translation [{:.3}, {:.3}, {:.3}] m; \
                 {} images, {} landmarks fused, {} added",
                dir.display(),
                yaw.to_degrees(),
                t.x,
                t.y,
                t.z,
                result.keyframe_ids.len(),
                result.fused_landmarks,
                added_landmarks,
            );
        }

        // The added map's sessions follow, with their transforms composed.
        let first_session = sessions.len();
        for (index, mut session) in map_sessions(&other, dir).into_iter().enumerate() {
            let other_from_session = session
                .get("base_from_session")
                .and_then(map_files::se3_from_json)
                .ok_or("map.json: session without base_from_session")?;
            session["base_from_session"] =
                map_files::se3_json(&base_from_other.compose(&other_from_session));
            if other_rig && session.get("t_imu_cam").is_none() {
                session["t_imu_cam"] = Value::Array(
                    other
                        .imu_from_camera
                        .iter()
                        .map(map_files::se3_json)
                        .collect(),
                );
            }
            if index == 0 {
                if let Some(registration) = &registration {
                    session["registration"] = json!({
                        "keyframes": registration.keyframes,
                        "localized_keyframes": registration.localized_keyframes,
                        "agreeing_keyframes": registration.agreeing_keyframes,
                        "keyframe_rms_m": registration.keyframe_rms_m,
                        "fused_landmarks": result.fused_landmarks,
                        "added_landmarks": added_landmarks,
                    });
                } else {
                    session["fusion"] = json!({
                        "fused_landmarks": result.fused_landmarks,
                        "added_landmarks": added_landmarks,
                    });
                }
            }
            sessions.push(session);
        }
        for image in &other.images {
            if let Some(&image_id) = result.keyframe_ids.get(&image.image_id) {
                images.push(map_files::ImageRow {
                    image_id,
                    session: first_session + image.session,
                    ..image.clone()
                });
            }
        }
        merged.map = result.map;
        merged.descriptors = result.descriptors;
    }

    let mut summary = merged.summary.clone();
    summary["world_frame"] = json!(
        "Basalt mapper world of the first session (gravity-aligned VIO world of its sequence)"
    );
    summary["sessions"] = Value::Array(sessions);
    if let Some((_, report)) = optimization {
        summary["session_alignment"] = report;
    }
    images.sort_by_key(|image| image.image_id);
    let summary = map_files::write(
        &args.out,
        &merged.map,
        &merged.descriptors,
        &images,
        summary,
    )?;
    println!(
        "wrote {}: {} images, {} landmarks, {} sessions",
        args.out.display(),
        summary["images"],
        summary["landmarks"],
        summary["sessions"].as_array().map_or(0, Vec::len)
    );
    Ok(())
}

fn main() {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&args) {
        eprintln!("basalt_map_merge: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Result<Args, String> {
        Args::parse(arguments.iter().map(|argument| (*argument).to_owned()))
    }

    #[test]
    fn motion_confirmed_boxes_remove_tracks_and_their_descriptors() {
        use nalgebra::{Point2, Point3};
        use std::collections::BTreeMap;
        use visloc_basalt::dynamic_objects::ObjectBox;
        use visloc_core::types::{
            Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation, VisualMap,
        };

        let mut saved = map_files::SavedMap {
            map: VisualMap::new(),
            descriptors: LandmarkDescriptorStore::new(),
            extra_descriptors: Vec::new(),
            imu_from_camera: Vec::new(),
            max_points: 0,
            frontend: Default::default(),
            summary: Value::Null,
            images: Vec::new(),
        };
        let mut observations = Vec::new();
        for (id, x) in [(1, 20.0), (2, 100.0)] {
            let observation = Observation {
                frame_id: 1,
                landmark_id: id,
                keypoint_index: observations.len(),
                xy: Point2::new(x, 20.0),
            };
            let mut landmark = Landmark::new(id, Point3::new(0.0, 0.0, 1.0));
            landmark.observations.push(observation.clone());
            saved.map.landmarks.insert(id, landmark);
            saved.descriptors.insert(id, vec![0.0; 32]);
            observations.push(observation);
        }
        saved.map.keyframes.insert(
            1,
            Keyframe {
                frame: Frame::new(1, 1),
                observations,
            },
        );
        saved.images.push(map_files::ImageRow {
            image_id: 1,
            frame_id: 1,
            cam_id: 0,
            timestamp_ns: Some(123),
            session: 0,
        });
        let boxes = BTreeMap::from([(
            123,
            vec![ObjectBox {
                object_id: 7,
                x_min: 10.0,
                y_min: 10.0,
                x_max: 30.0,
                y_max: 30.0,
            }],
        )]);
        assert_eq!(map_files::exclude_moving_tracks(&mut saved, &boxes), 1);
        assert!(!saved.map.landmarks.contains_key(&1));
        assert!(!saved.descriptors.contains(1));
        assert!(saved.descriptors.contains(2));
        assert_eq!(saved.map.keyframes[&1].observations[0].landmark_id, 2);
    }

    #[test]
    fn arguments_need_a_base_an_addition_and_an_output() {
        let args = parse(&[
            "--base",
            "a",
            "--add",
            "b",
            "--add",
            "c",
            "--out",
            "m",
            "--min-keyframes",
            "5",
            "--fuse-radius-m",
            "0",
        ])
        .expect("parsed");
        assert_eq!(args.mode, Mode::Sequential);
        assert_eq!(args.add, [PathBuf::from("b"), PathBuf::from("c")]);
        assert_eq!(args.config.min_keyframes, 5);
        assert_eq!(args.config.fuse_radius_m, 0.0);
        assert!(parse(&["--base", "a", "--out", "m"]).is_err());
        assert!(parse(&["--add", "b", "--out", "m"]).is_err());
        assert!(parse(&["--base", "a", "--add", "b"]).is_err());
        assert!(parse(&["--base", "a", "--add", "b", "--out", "m", "--bogus", "1"]).is_err());
        let mut optimize = vec![
            "--base",
            "a",
            "--add",
            "b",
            "--out",
            "m",
            "--optimize-sessions",
        ];
        assert_eq!(parse(&optimize).unwrap().mode, Mode::Optimize);
        optimize.push("--inspect-loops");
        assert!(parse(&optimize).is_err());
    }

    #[test]
    fn keyframe_separation_override_requires_refinement_and_positive_distance() {
        let mut args = vec![
            "--base",
            "a",
            "--add",
            "b",
            "--out",
            "m",
            "--optimize-sessions",
        ];
        assert!(parse(&args)
            .unwrap()
            .keyframe_loop_min_separation_m
            .is_none());
        args.extend(["--keyframe-loop-min-separation-m", "0.15"]);
        assert!(parse(&args).is_err());
        args.push("--refine-keyframes");
        assert_eq!(
            parse(&args).unwrap().keyframe_loop_min_separation_m,
            Some(0.15)
        );
        for invalid in ["0", "-1", "NaN", "inf"] {
            args[8] = invalid;
            assert!(parse(&args).is_err());
        }
    }

    #[test]
    fn images_csv_reads_with_and_without_the_session_column() {
        let dir = std::env::temp_dir().join(format!("visloc_map_merge_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let old = dir.join("old.csv");
        std::fs::write(
            &old,
            "image_id,frame_id,cam_id,timestamp_ns\n1,0,0,100\n2,0,1,\n",
        )
        .expect("write");
        let new = dir.join("new.csv");
        std::fs::write(
            &new,
            "image_id,frame_id,cam_id,timestamp_ns,session\n7,3,1,200,2\n",
        )
        .expect("write");
        let old = map_files::read_images(&old).expect("old layout");
        let new = map_files::read_images(&new).expect("new layout");
        let bad = dir.join("bad.csv");
        std::fs::write(&bad, "image_id,frame_id,cam_id,timestamp_ns\n1,x,0,100\n").expect("write");
        assert!(map_files::read_images(&bad).is_err());
        std::fs::remove_dir_all(&dir).expect("cleanup");
        assert_eq!(
            old,
            [
                map_files::ImageRow {
                    image_id: 1,
                    frame_id: 0,
                    cam_id: 0,
                    timestamp_ns: Some(100),
                    session: 0,
                },
                map_files::ImageRow {
                    image_id: 2,
                    frame_id: 0,
                    cam_id: 1,
                    timestamp_ns: None,
                    session: 0,
                },
            ]
        );
        assert_eq!(new[0].session, 2);
        assert_eq!(new[0].timestamp_ns, Some(200));
    }
}
