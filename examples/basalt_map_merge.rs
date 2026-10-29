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
//! from the same camera rig and calibration.
//!
//! The merged map keeps the base map's world frame, so poses localized
//! against it, and anything aligned to the base map's SLAM trajectory, stay
//! valid. `map.json` lists the sessions with the transform from each session's
//! world and how it was registered; `images.csv` says which session each image
//! came from. A merged map can be the `--base` (or an `--add`) of another
//! merge.

#[path = "basalt_map/map_files.rs"]
mod map_files;

use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};
use visloc_core::geometry::SE3;
use visloc_rs::localization::{merge_maps, register_map, MapMergeConfig};

#[derive(Debug, Clone, PartialEq)]
struct Args {
    base: PathBuf,
    add: Vec<PathBuf>,
    out: PathBuf,
    config: MapMergeConfig,
}

impl Args {
    fn usage() -> String {
        "usage: basalt_map_merge --base DIR --add DIR [--add DIR ...] --out DIR \
         [--min-inliers N] [--min-keyframes N] [--max-hamming N] \
         [--fuse-radius-m M] [--fuse-max-hamming N] [--threads N]"
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
                "--min-inliers" => config.min_inliers = number(&option, value()?)?,
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
    let mut merged = map_files::load(&args.base)?;
    let mut sessions = map_sessions(&merged, &args.base);
    let mut images = merged.images.clone();
    println!(
        "base {}: {} images, {} landmarks",
        args.base.display(),
        merged.map.keyframes.len(),
        merged.map.landmarks.len()
    );

    for dir in &args.add {
        let other = map_files::load(dir)?;
        if !same_rig(&merged.imu_from_camera, &other.imu_from_camera) {
            return Err(format!(
                "{}: camera extrinsics differ from the base map's",
                dir.display()
            )
            .into());
        }
        // Descriptors from equalized and raw images do not match each other.
        if other.local_contrast != merged.local_contrast {
            return Err(format!(
                "{}: built with {:?} local contrast equalization, the base map with {:?}; \
                 build both the same way",
                dir.display(),
                other.local_contrast,
                merged.local_contrast
            )
            .into());
        }
        if other.feature_smoothing != merged.feature_smoothing {
            return Err(format!(
                "{}: built with {:?} blur before ORB, the base map with {:?}; \
                 build both the same way",
                dir.display(),
                other.feature_smoothing,
                merged.feature_smoothing
            )
            .into());
        }
        let started = Instant::now();
        let registration = register_map(
            &merged.map,
            &merged.descriptors,
            &other.map,
            &other.descriptors,
            &args.config,
        )
        .map_err(|error| format!("{}: not registered: {error}", dir.display()))?;
        let registered_s = started.elapsed().as_secs_f64();
        let result = merge_maps(
            &merged.map,
            &merged.descriptors,
            &other.map,
            &other.descriptors,
            &registration.base_from_other,
            &args.config,
        )?;
        let added_landmarks = result.map.landmarks.len() - merged.map.landmarks.len();
        let yaw = registration.base_from_other.rotation.euler_angles().2;
        let t = registration.base_from_other.translation;
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

        // The added map's sessions follow, with their transforms composed.
        let first_session = sessions.len();
        for (index, mut session) in map_sessions(&other, dir).into_iter().enumerate() {
            let other_from_session = session
                .get("base_from_session")
                .and_then(map_files::se3_from_json)
                .ok_or("map.json: session without base_from_session")?;
            session["base_from_session"] =
                map_files::se3_json(&registration.base_from_other.compose(&other_from_session));
            if index == 0 {
                session["registration"] = json!({
                    "keyframes": registration.keyframes,
                    "localized_keyframes": registration.localized_keyframes,
                    "agreeing_keyframes": registration.agreeing_keyframes,
                    "keyframe_rms_m": registration.keyframe_rms_m,
                    "fused_landmarks": result.fused_landmarks,
                    "added_landmarks": added_landmarks,
                });
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
        assert_eq!(args.add, [PathBuf::from("b"), PathBuf::from("c")]);
        assert_eq!(args.config.min_keyframes, 5);
        assert_eq!(args.config.fuse_radius_m, 0.0);
        assert!(parse(&["--base", "a", "--out", "m"]).is_err());
        assert!(parse(&["--add", "b", "--out", "m"]).is_err());
        assert!(parse(&["--base", "a", "--add", "b"]).is_err());
        assert!(parse(&["--base", "a", "--add", "b", "--out", "m", "--bogus", "1"]).is_err());
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
