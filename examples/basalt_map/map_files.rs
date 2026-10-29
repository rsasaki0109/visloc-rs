//! On-disk layout of a Basalt mapper map saved for localization, shared by
//! `basalt_euroc_online_slam_demo --save-map` (writer) and
//! `basalt_euroc_localize_demo` (reader):
//!
//! ```text
//! <map_dir>/
//!   sparse/{cameras,images,points3D}.txt  COLMAP text model, mapper world frame
//!   landmark_descriptors.txt              LANDMARK_ID + 256 ORB bits as 0/1
//!   landmark_descriptors_extra.txt        optional: more descriptors per landmark,
//!                                         from its views that differ most (ids repeat)
//!   images.csv                            IMAGE_ID -> mapper frame, camera, timestamp, session
//!   map.json                              T_imu_cam per camera, fit errors, counts, sessions
//! ```
//!
//! The COLMAP model opens in COLMAP itself; `map.json` adds what COLMAP has no
//! field for, the IMU-from-camera extrinsics, so a localized camera pose can be
//! reported as an IMU pose comparable with the SLAM trajectories.
//!
//! A map merged by `basalt_map_merge` holds several mapping sessions in the
//! first one's world frame: `map.json`'s `sessions` lists them with the
//! transform from each session's own world, and `images.csv`'s `session`
//! column says which session an image came from (maps saved before the column
//! existed read as session 0).

// The writer, the reader and the merger examples each use part of this module.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde_json::{json, Value};
use visloc_basalt::contrast::LocalContrast;
use visloc_basalt::mapper::{localization_camera_id, LocalizationMapExport};
use visloc_basalt::BasaltCalibration;
use visloc_core::geometry::SE3;
use visloc_core::types::{LandmarkDescriptorStore, Observation, VisualMap};
use visloc_rs::io::colmap::{write_colmap_text_model, ColmapMapProvider};
use visloc_rs::io::descriptors::{
    read_landmark_descriptor_rows, write_landmark_descriptor_rows, write_landmark_descriptors_txt,
};
use visloc_rs::{DescriptorProvider, MapProvider};

pub const MAP_SCHEMA: &str = "visloc.basalt_localization_map.v1";

pub fn sparse_dir(map_dir: &Path) -> PathBuf {
    map_dir.join("sparse")
}

pub fn descriptors_path(map_dir: &Path) -> PathBuf {
    map_dir.join("landmark_descriptors.txt")
}

pub fn extra_descriptors_path(map_dir: &Path) -> PathBuf {
    map_dir.join("landmark_descriptors_extra.txt")
}

/// A saved map loaded back for localization.
pub struct SavedMap {
    pub map: VisualMap,
    pub descriptors: LandmarkDescriptorStore,
    /// `landmark_descriptors_extra.txt`, empty when the file is missing.
    pub extra_descriptors: Vec<(u64, Vec<f32>)>,
    /// `T_imu_cam` per Basalt camera index.
    pub imu_from_camera: Vec<SE3>,
    /// Detector budget the map's keyframes were built with.
    pub max_points: usize,
    /// Local contrast equalization the map's keyframes were built with,
    /// which localization must apply to its images too.
    pub local_contrast: Option<LocalContrast>,
    /// `map.json` as read.
    pub summary: Value,
    /// `images.csv`, empty when the file is missing.
    pub images: Vec<ImageRow>,
}

/// One row of `images.csv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRow {
    pub image_id: u64,
    pub frame_id: u64,
    pub cam_id: u16,
    pub timestamp_ns: Option<i64>,
    /// Index into `map.json`'s `sessions`.
    pub session: usize,
}

pub fn se3_json(pose: &SE3) -> Value {
    let q = pose.rotation.quaternion();
    let t = pose.translation;
    json!({"translation": [t.x, t.y, t.z], "quaternion_wxyz": [q.w, q.i, q.j, q.k]})
}

pub fn se3_from_json(value: &Value) -> Option<SE3> {
    let numbers = |key: &str| -> Option<Vec<f64>> {
        value
            .get(key)?
            .as_array()?
            .iter()
            .map(Value::as_f64)
            .collect()
    };
    let [tx, ty, tz]: [f64; 3] = numbers("translation")?.try_into().ok()?;
    let [qw, qx, qy, qz]: [f64; 4] = numbers("quaternion_wxyz")?.try_into().ok()?;
    Some(SE3::new(
        UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
        Vector3::new(tx, ty, tz),
    ))
}

/// `map.json`'s `feature_local_contrast` entry.
pub fn local_contrast_json(local_contrast: &LocalContrast) -> Value {
    json!({"tiles": local_contrast.tiles, "clip_limit": local_contrast.clip_limit})
}

/// The map's `feature_local_contrast`: `None` when absent (a map built from
/// the raw images).
pub fn local_contrast_from_json(summary: &Value) -> Result<Option<LocalContrast>, String> {
    let Some(value) = summary
        .get("feature_local_contrast")
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let tiles = value.get("tiles").and_then(Value::as_u64);
    let clip_limit = value.get("clip_limit").and_then(Value::as_f64);
    match (tiles, clip_limit) {
        (Some(tiles), Some(clip_limit)) if tiles > 0 && clip_limit > 0.0 => {
            Ok(Some(LocalContrast {
                tiles: tiles as usize,
                clip_limit,
            }))
        }
        _ => Err(format!("map.json: bad feature_local_contrast {value}")),
    }
}

/// Writes `export` under `map_dir` as a one-session map; returns the
/// `map.json` summary. `local_contrast` is the equalization the keyframes'
/// features were extracted with, if any; with `extra_descriptors`, the
/// export's extra descriptors go to `landmark_descriptors_extra.txt`.
pub fn save(
    map_dir: &Path,
    export: &LocalizationMapExport,
    calibration: &BasaltCalibration,
    max_points: usize,
    local_contrast: Option<&LocalContrast>,
    extra_descriptors: bool,
) -> Result<Value, Box<dyn std::error::Error>> {
    let images = export
        .images
        .iter()
        .map(|image| ImageRow {
            image_id: image.image_id,
            frame_id: image.frame_id,
            cam_id: image.cam_id,
            timestamp_ns: image.timestamp_ns,
            session: 0,
        })
        .collect::<Vec<_>>();
    let cameras = export
        .camera_fits
        .iter()
        .enumerate()
        .map(|(cam_id, fit)| {
            json!({
                "camera_id": localization_camera_id(cam_id as u16),
                "basalt_camera": cam_id,
                "model": "OPENCV_FISHEYE",
                "params": fit.params,
                "t_imu_cam": calibration.camera_to_imu(cam_id as u16).map(se3_json),
                "fit_max_angle_deg": fit.max_angle.to_degrees(),
                "fit_max_error_px": fit.max_error_px,
            })
        })
        .collect::<Vec<_>>();
    let mut summary = json!({
        "schema": MAP_SCHEMA,
        "world_frame": "Basalt mapper world (gravity-aligned VIO world of the mapped sequence)",
        "feature_max_points": max_points,
        "cameras": cameras,
    });
    if let Some(local_contrast) = local_contrast {
        summary["feature_local_contrast"] = local_contrast_json(local_contrast);
    }
    let extra_path = extra_descriptors_path(map_dir);
    if extra_descriptors {
        fs::create_dir_all(map_dir)?;
        write_landmark_descriptor_rows(&export.extra_descriptors, &extra_path)?;
        summary["extra_descriptors"] = json!(export.extra_descriptors.len());
    } else if extra_path.exists() {
        fs::remove_file(&extra_path)?;
    }
    write(map_dir, &export.map, &export.descriptors, &images, summary)
}

/// Writes a map under `map_dir`: the COLMAP model, the descriptors,
/// `images` as `images.csv` and `summary` as `map.json`, with its image,
/// landmark and observation counts set; returns the written summary.
pub fn write(
    map_dir: &Path,
    map: &VisualMap,
    descriptors: &LandmarkDescriptorStore,
    images: &[ImageRow],
    mut summary: Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    fs::create_dir_all(map_dir)?;
    write_colmap_text_model(map, sparse_dir(map_dir))?;
    write_landmark_descriptors_txt(descriptors, descriptors_path(map_dir))?;

    let mut images_csv = String::from("image_id,frame_id,cam_id,timestamp_ns,session\n");
    for image in images {
        images_csv.push_str(&format!(
            "{},{},{},{},{}\n",
            image.image_id,
            image.frame_id,
            image.cam_id,
            image
                .timestamp_ns
                .map_or_else(String::new, |value| value.to_string()),
            image.session,
        ));
    }
    fs::write(map_dir.join("images.csv"), images_csv)?;

    summary["images"] = json!(map.keyframes.len());
    summary["landmarks"] = json!(map.landmarks.len());
    summary["observations"] = json!(map
        .landmarks
        .values()
        .map(|landmark| landmark.observations.len())
        .sum::<usize>());
    fs::write(
        map_dir.join("map.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

/// Reads `images.csv`; a missing `session` column reads as session 0.
pub fn read_images(path: &Path) -> Result<Vec<ImageRow>, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    text.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            let bad = || format!("{}: bad row {line:?}", path.display());
            if fields.len() < 4 {
                return Err(bad().into());
            }
            Ok(ImageRow {
                image_id: fields[0].parse().map_err(|_| bad())?,
                frame_id: fields[1].parse().map_err(|_| bad())?,
                cam_id: fields[2].parse().map_err(|_| bad())?,
                timestamp_ns: match fields[3] {
                    "" => None,
                    value => Some(value.parse().map_err(|_| bad())?),
                },
                session: match fields.get(4) {
                    None | Some(&"") => 0,
                    Some(value) => value.parse().map_err(|_| bad())?,
                },
            })
        })
        .collect()
}

pub fn load(map_dir: &Path) -> Result<SavedMap, Box<dyn std::error::Error>> {
    let summary_path = map_dir.join("map.json");
    let summary: Value = serde_json::from_str(
        &fs::read_to_string(&summary_path)
            .map_err(|error| format!("{}: {error}", summary_path.display()))?,
    )?;
    if summary.get("schema").and_then(Value::as_str) != Some(MAP_SCHEMA) {
        return Err(format!("{}: not a {MAP_SCHEMA} map", map_dir.display()).into());
    }
    let imu_from_camera = summary
        .get("cameras")
        .and_then(Value::as_array)
        .ok_or("map.json: missing cameras")?
        .iter()
        .map(|camera| {
            camera
                .get("t_imu_cam")
                .and_then(se3_from_json)
                .ok_or("map.json: camera without t_imu_cam")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_points = summary
        .get("feature_max_points")
        .and_then(Value::as_u64)
        .ok_or("map.json: missing feature_max_points")? as usize;
    let local_contrast = local_contrast_from_json(&summary)?;
    let images_path = map_dir.join("images.csv");
    let images = if images_path.exists() {
        read_images(&images_path)?
    } else {
        Vec::new()
    };

    let provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        sparse_dir(map_dir),
        descriptors_path(map_dir),
    )?;
    let descriptors = provider
        .landmark_descriptor_store()
        .cloned()
        .ok_or("map has no descriptor store")?;
    let extra_path = extra_descriptors_path(map_dir);
    let extra_descriptors = if extra_path.exists() {
        read_landmark_descriptor_rows(&extra_path)
            .map_err(|error| format!("{}: {error}", extra_path.display()))?
    } else {
        Vec::new()
    };
    // The COLMAP text reader fills only the keyframes' observations; rebuild
    // each landmark's (its TRACK[]) from them, so a map written back keeps
    // its tracks.
    let mut map = provider.visual_map().clone();
    let mut tracks = std::collections::HashMap::<u64, Vec<Observation>>::new();
    for keyframe in map.keyframes.values() {
        for observation in &keyframe.observations {
            tracks
                .entry(observation.landmark_id)
                .or_default()
                .push(observation.clone());
        }
    }
    for (id, landmark) in &mut map.landmarks {
        if let (true, Some(mut track)) = (landmark.observations.is_empty(), tracks.remove(id)) {
            track.sort_by_key(|observation| (observation.frame_id, observation.keypoint_index));
            landmark.observations = track;
        }
    }
    Ok(SavedMap {
        map,
        descriptors,
        extra_descriptors,
        imu_from_camera,
        max_points,
        local_contrast,
        summary,
        images,
    })
}
