//! On-disk layout of a Basalt mapper map saved for localization, shared by
//! `basalt_euroc_online_slam_demo --save-map` (writer) and
//! `basalt_euroc_localize_demo` (reader):
//!
//! ```text
//! <map_dir>/
//!   sparse/{cameras,images,points3D}.txt  COLMAP text model, mapper world frame
//!   landmark_descriptors.txt              LANDMARK_ID + 256 ORB bits as 0/1
//!   images.csv                            IMAGE_ID -> mapper frame, camera, timestamp
//!   map.json                              T_imu_cam per camera, fit errors, counts
//! ```
//!
//! The COLMAP model opens in COLMAP itself; `map.json` adds what COLMAP has no
//! field for, the IMU-from-camera extrinsics, so a localized camera pose can be
//! reported as an IMU pose comparable with the SLAM trajectories.

// The writer and the reader example each use one half of this module.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde_json::{json, Value};
use visloc_basalt::mapper::{localization_camera_id, LocalizationMapExport};
use visloc_basalt::BasaltCalibration;
use visloc_core::geometry::SE3;
use visloc_core::types::{LandmarkDescriptorStore, VisualMap};
use visloc_rs::io::colmap::{write_colmap_text_model, ColmapMapProvider};
use visloc_rs::io::descriptors::write_landmark_descriptors_txt;
use visloc_rs::{DescriptorProvider, MapProvider};

pub const MAP_SCHEMA: &str = "visloc.basalt_localization_map.v1";

pub fn sparse_dir(map_dir: &Path) -> PathBuf {
    map_dir.join("sparse")
}

pub fn descriptors_path(map_dir: &Path) -> PathBuf {
    map_dir.join("landmark_descriptors.txt")
}

/// A saved map loaded back for localization.
pub struct SavedMap {
    pub map: VisualMap,
    pub descriptors: LandmarkDescriptorStore,
    /// `T_imu_cam` per Basalt camera index.
    pub imu_from_camera: Vec<SE3>,
    /// Detector budget the map's keyframes were built with.
    pub max_points: usize,
}

fn se3_json(pose: &SE3) -> Value {
    let q = pose.rotation.quaternion();
    let t = pose.translation;
    json!({"translation": [t.x, t.y, t.z], "quaternion_wxyz": [q.w, q.i, q.j, q.k]})
}

fn se3_from_json(value: &Value) -> Option<SE3> {
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

/// Writes `export` under `map_dir`; returns the `map.json` summary.
pub fn save(
    map_dir: &Path,
    export: &LocalizationMapExport,
    calibration: &BasaltCalibration,
    max_points: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    fs::create_dir_all(map_dir)?;
    write_colmap_text_model(&export.map, sparse_dir(map_dir))?;
    write_landmark_descriptors_txt(&export.descriptors, descriptors_path(map_dir))?;

    let mut images_csv = String::from("image_id,frame_id,cam_id,timestamp_ns\n");
    for image in &export.images {
        images_csv.push_str(&format!(
            "{},{},{},{}\n",
            image.image_id,
            image.frame_id,
            image.cam_id,
            image
                .timestamp_ns
                .map_or_else(String::new, |value| value.to_string()),
        ));
    }
    fs::write(map_dir.join("images.csv"), images_csv)?;

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
    let summary = json!({
        "schema": MAP_SCHEMA,
        "world_frame": "Basalt mapper world (gravity-aligned VIO world of the mapped sequence)",
        "feature_max_points": max_points,
        "cameras": cameras,
        "images": export.map.keyframes.len(),
        "landmarks": export.map.landmarks.len(),
        "observations": export
            .map
            .landmarks
            .values()
            .map(|landmark| landmark.observations.len())
            .sum::<usize>(),
    });
    fs::write(
        map_dir.join("map.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
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

    let provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        sparse_dir(map_dir),
        descriptors_path(map_dir),
    )?;
    let descriptors = provider
        .landmark_descriptor_store()
        .cloned()
        .ok_or("map has no descriptor store")?;
    Ok(SavedMap {
        map: provider.visual_map().clone(),
        descriptors,
        imu_from_camera,
        max_points,
    })
}
