//! Relocalize one query image's features against a map built by the native
//! COLMAP port (`examples/colmap_incremental_mapper`), using a landmark
//! descriptor store exported from the COLMAP database by
//! `scripts/export_openloris_localization_map.py`.
//!
//! Usage:
//!   cargo run --release --example localize_openloris_map -- \
//!     --map-dir <model/model/0> \
//!     --landmark-descriptors <out/landmark_descriptors.txt> \
//!     --query-features <out/query_features/cam1_000001.txt> \
//!     [--camera-id 1] [--pnp-solver dlt|p3p]
//!
//! `--pnp-solver p3p` uses the P3P minimal solver instead of the 6-point DLT
//! (the default); P3P stays well-posed when the matched landmarks are
//! coplanar, so prefer it for maps dominated by walls.
//!
//! Prints one line: either
//!   `SUCCESS inliers=<n> matches=<n> corr=<n> reproj=<f> t=<x> <y> <z> q=<w> <i> <j> <k>`
//! or `FAILURE reason=<..> candidates=<n> matches=<n> corr=<n>`.

use std::collections::HashMap;
use std::path::PathBuf;

use visloc_rs::core::types::QueryImage;
use visloc_rs::io::colmap::read_colmap_text_model;
use visloc_rs::io::descriptors::read_landmark_descriptors_txt;
use visloc_rs::io::query_features::read_query_features_txt;
use visloc_rs::{
    AllLandmarksSelector, BruteForceMatcher, LocalizationPipeline, LocalizationRansac,
    PnpMinimalSolver,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut flags: HashMap<String, String> = HashMap::new();
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        if let Some(name) = arg.strip_prefix("--") {
            flags.insert(name.to_string(), argv.next().unwrap_or_default());
        }
    }
    let get = |key: &str| -> Result<PathBuf, String> {
        flags
            .get(key)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing required --{key}"))
    };
    let map_dir = get("map-dir")?;
    let descriptors_path = get("landmark-descriptors")?;

    let map = read_colmap_text_model(&map_dir)?;
    let descriptor_store = read_landmark_descriptors_txt(&descriptors_path)?;

    let mut camera_ids: Vec<u64> = map.cameras.keys().copied().collect();
    camera_ids.sort_unstable();
    let camera_cam1: u64 = flags
        .get("camera-id")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| camera_ids.first().copied())
        .ok_or("map contains no cameras")?;
    // Default the second sensor to the largest camera id different from cam1
    // (OpenLORIS rig camera 2); a query whose flat name starts with `cam2`
    // uses it. `--camera-id` still sets the cam1 camera explicitly.
    let camera_cam2: u64 = flags
        .get("camera-id-cam2")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| {
            camera_ids
                .iter()
                .rev()
                .copied()
                .find(|id| *id != camera_cam1)
        })
        .unwrap_or(camera_cam1);
    let camera_for = |name: &str| {
        let id = if name.starts_with("cam2") {
            camera_cam2
        } else {
            camera_cam1
        };
        map.cameras
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("map has no camera id {id}"))
    };

    let mut query_paths: Vec<PathBuf> = Vec::new();
    if let Some(single) = flags.get("query-features") {
        query_paths.push(PathBuf::from(single));
    }
    if let Some(dir) = flags.get("query-features-dir") {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        entries.sort();
        query_paths.extend(entries);
    }
    if query_paths.is_empty() {
        return Err("provide --query-features or --query-features-dir".into());
    }

    let pnp_solver: PnpMinimalSolver = match flags.get("pnp-solver") {
        Some(value) => value.parse()?,
        None => PnpMinimalSolver::default(),
    };
    let mut pipeline = LocalizationPipeline::<
        BruteForceMatcher,
        AllLandmarksSelector,
        LocalizationRansac,
    >::default()
    .with_pnp_solver(pnp_solver);
    // `LocalizationConfig::min_inliers` defaults to 0, so a degenerate pose
    // with no inliers is reported as success; require a real floor unless the
    // caller overrides it.
    pipeline.config.min_inliers = flags
        .get("min-inliers")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let prefix = query_paths.len() > 1;
    for path in query_paths {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let features = read_query_features_txt(&path)?;
        let camera = camera_for(&name)?;
        let query = QueryImage {
            camera,
            keypoints: features.keypoints,
            descriptors: features.descriptors,
        };
        let result = pipeline.localize_with_descriptor_store(&query, &map, &descriptor_store);
        let label = if prefix {
            format!("{name} ")
        } else {
            String::new()
        };
        if result.success {
            if let Some(pose) = result.pose {
                let t = pose.world_to_camera.translation;
                let q = pose.world_to_camera.rotation;
                println!(
                    "{label}SUCCESS inliers={} matches={} corr={} reproj={} t={} {} {} q={} {} {} {}",
                    result.inlier_count,
                    result.match_count,
                    result.correspondence_count,
                    result.reprojection_error.unwrap_or(f64::NAN),
                    t.x,
                    t.y,
                    t.z,
                    q.w,
                    q.i,
                    q.j,
                    q.k,
                );
            } else {
                println!("{label}FAILURE reason=no_pose");
            }
        } else {
            println!(
                "{label}FAILURE reason={:?} candidates={} matches={} corr={}",
                result.failure_reason,
                result.candidate_landmark_count,
                result.match_count,
                result.correspondence_count
            );
        }
    }
    Ok(())
}
