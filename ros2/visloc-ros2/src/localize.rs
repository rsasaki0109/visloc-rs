//! Transport-agnostic map-based localization core of `visloc_localize_node`.
//!
//! Same API path as `examples/localize_openloris_map.rs`: a COLMAP model
//! (text or binary) provides cameras and 3D landmarks, a landmark descriptor
//! store (`<landmark_id> <descriptor floats>` per line, as written by
//! `scripts/export_openloris_localization_map.py`) provides the matching
//! descriptors, and `LocalizationPipeline::<BruteForceMatcher,
//! AllLandmarksSelector, PnPRansac>` estimates the pose. The only difference
//! is that query features are extracted here from the incoming image with
//! the repository's pure-Rust SIFT, instead of being read from a text file.
//!
//! The map's descriptors must live in the same descriptor space as the
//! node's extractor (128-D SIFT, L2 or RootSIFT normalization selectable);
//! otherwise matching silently degrades. See the README.

use std::path::Path;
use std::time::Instant;

use visloc_core::geometry::SE3;
use visloc_core::types::{
    Camera, CameraId, CameraModel, LandmarkDescriptorStore, QueryImage, VisualMap,
};
use visloc_io::colmap::{read_colmap_binary_model, read_colmap_text_model};
use visloc_io::descriptors::read_landmark_descriptors_txt;
use visloc_localization::{AllLandmarksSelector, LocalizationConfig, LocalizationPipeline};
use visloc_vision::features::sift::{extract_sift_features, GrayImage, SiftConfig};
use visloc_vision::matching::BruteForceMatcher;
use visloc_vision::ransac::PnPRansac;

use crate::image::LumaImage;
use crate::msgs::{
    CameraInfo, Covariance6, DiagnosticArray, DiagnosticStatus, Header, KeyValue,
    PoseWithCovariance, PoseWithCovarianceStamped, Time,
};
use crate::vio::ros_pose;

/// Tunables of the localization node.
#[derive(Clone, Debug)]
pub struct LocalizeOptions {
    pub sift: SiftConfig,
    pub config: LocalizationConfig,
    /// 1-sigma position / orientation uncertainty written into the
    /// published covariance diagonal (the PnP solver does not provide a
    /// calibrated covariance).
    pub position_stddev_m: f64,
    pub orientation_stddev_rad: f64,
}

impl Default for LocalizeOptions {
    fn default() -> Self {
        Self {
            sift: SiftConfig {
                max_keypoints: 4000,
                ..SiftConfig::default()
            },
            config: LocalizationConfig {
                // `LocalizationConfig::min_inliers` defaults to 0; require a
                // real floor like `localize_openloris_map` does.
                min_inliers: 12,
                ..LocalizationConfig::default()
            },
            position_stddev_m: 0.1,
            orientation_stddev_rad: 0.05,
        }
    }
}

/// Result of localizing one image.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalizeOutcome {
    pub success: bool,
    /// Camera pose in the map frame (`T_map_camera`).
    pub camera_to_map: Option<SE3>,
    pub feature_count: usize,
    pub match_count: usize,
    pub correspondence_count: usize,
    pub inlier_count: usize,
    pub reprojection_error: Option<f64>,
    pub failure: Option<String>,
    pub elapsed_ms: f64,
}

pub struct LocalizeCore {
    map: VisualMap,
    descriptors: LandmarkDescriptorStore,
    default_camera: Camera,
    pipeline: LocalizationPipeline<BruteForceMatcher, AllLandmarksSelector, PnPRansac>,
    options: LocalizeOptions,
}

/// Reads a COLMAP model directory, binary (`cameras.bin`) preferred.
pub fn read_colmap_model(dir: &Path) -> Result<VisualMap, String> {
    if dir.join("cameras.bin").is_file() {
        read_colmap_binary_model(dir)
            .map_err(|error| format!("COLMAP binary model {}: {error}", dir.display()))
    } else {
        read_colmap_text_model(dir)
            .map_err(|error| format!("COLMAP text model {}: {error}", dir.display()))
    }
}

impl LocalizeCore {
    /// Loads a COLMAP model and an optional descriptor store from disk.
    /// Without a store, per-landmark descriptors embedded in the map are
    /// used (COLMAP models normally carry none).
    pub fn load(
        map_dir: &Path,
        descriptors_path: Option<&Path>,
        camera_id: Option<CameraId>,
        options: LocalizeOptions,
    ) -> Result<Self, String> {
        let map = read_colmap_model(map_dir)?;
        let descriptors = match descriptors_path {
            Some(path) => read_landmark_descriptors_txt(path)
                .map_err(|error| format!("descriptor store {}: {error}", path.display()))?,
            None => LandmarkDescriptorStore::from_visual_map(&map),
        };
        Self::new(map, descriptors, camera_id, options)
    }

    pub fn new(
        map: VisualMap,
        descriptors: LandmarkDescriptorStore,
        camera_id: Option<CameraId>,
        options: LocalizeOptions,
    ) -> Result<Self, String> {
        if descriptors.is_empty() {
            return Err(
                "the map has no landmark descriptors: pass a landmark descriptor store".into(),
            );
        }
        let camera_id = match camera_id {
            Some(id) => id,
            None => *map
                .cameras
                .keys()
                .min()
                .ok_or("the map contains no cameras")?,
        };
        let default_camera = map
            .cameras
            .get(&camera_id)
            .cloned()
            .ok_or_else(|| format!("the map has no camera id {camera_id}"))?;
        let pipeline = LocalizationPipeline::new(
            BruteForceMatcher {
                ratio: options.config.ratio,
            },
            options.config.clone(),
        );
        Ok(Self {
            map,
            descriptors,
            default_camera,
            pipeline,
            options,
        })
    }

    pub const fn map(&self) -> &VisualMap {
        &self.map
    }

    pub const fn default_camera(&self) -> &Camera {
        &self.default_camera
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }

    pub const fn options(&self) -> &LocalizeOptions {
        &self.options
    }

    /// Extracts SIFT on `image` and localizes it against the map.
    /// `camera` overrides the map camera (e.g. from `CameraInfo`).
    pub fn localize(&self, image: &LumaImage, camera: Option<&Camera>) -> LocalizeOutcome {
        let started = Instant::now();
        let camera = camera.unwrap_or(&self.default_camera).clone();
        let failure = |message: String, feature_count: usize| LocalizeOutcome {
            success: false,
            camera_to_map: None,
            feature_count,
            match_count: 0,
            correspondence_count: 0,
            inlier_count: 0,
            reprojection_error: None,
            failure: Some(message),
            elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        };
        if (camera.width as usize, camera.height as usize) != (image.width, image.height) {
            return failure(
                format!(
                    "image is {}x{} but camera {} is {}x{}",
                    image.width, image.height, camera.id, camera.width, camera.height
                ),
                0,
            );
        }
        let pixels = image.to_unit_f32();
        let gray = match GrayImage::new(image.width, image.height, &pixels) {
            Ok(gray) => gray,
            Err(error) => return failure(format!("SIFT input: {error}"), 0),
        };
        let features = match extract_sift_features(&gray, &self.options.sift) {
            Ok(features) => features,
            Err(error) => return failure(format!("SIFT: {error}"), 0),
        };
        let feature_count = features.keypoints.len();
        let query = QueryImage {
            camera,
            keypoints: features.keypoints,
            descriptors: features.descriptors,
        };
        let result =
            self.pipeline
                .localize_with_descriptor_store(&query, &self.map, &self.descriptors);
        let camera_to_map = result
            .success
            .then(|| result.pose.as_ref().map(|pose| pose.camera_to_world()))
            .flatten();
        LocalizeOutcome {
            success: camera_to_map.is_some(),
            camera_to_map,
            feature_count,
            match_count: result.match_count,
            correspondence_count: result.correspondence_count,
            inlier_count: result.inlier_count,
            reprojection_error: result.reprojection_error,
            failure: result
                .failure_reason
                .map(|reason| format!("{reason:?}"))
                .or_else(|| (!result.success).then(|| "no pose".to_string())),
            elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        }
    }

    pub fn pose_msg(
        &self,
        outcome: &LocalizeOutcome,
        stamp: Time,
        map_frame: &str,
    ) -> Option<PoseWithCovarianceStamped> {
        let pose = outcome.camera_to_map.as_ref()?;
        Some(PoseWithCovarianceStamped {
            header: Header::new(stamp, map_frame),
            pose: PoseWithCovariance {
                pose: ros_pose(pose),
                covariance: Covariance6::diagonal(
                    self.options.position_stddev_m.powi(2),
                    self.options.orientation_stddev_rad.powi(2),
                ),
            },
        })
    }
}

/// `diagnostic_msgs/DiagnosticArray` describing one localization attempt.
pub fn diagnostics_msg(outcome: &LocalizeOutcome, stamp: Time, node_name: &str) -> DiagnosticArray {
    let value = |key: &str, value: String| KeyValue {
        key: key.into(),
        value,
    };
    DiagnosticArray {
        header: Header::new(stamp, ""),
        status: vec![DiagnosticStatus {
            level: if outcome.success {
                DiagnosticStatus::OK
            } else {
                DiagnosticStatus::WARN
            },
            name: format!("{node_name}: localization"),
            message: if outcome.success {
                format!("localized with {} inliers", outcome.inlier_count)
            } else {
                outcome
                    .failure
                    .clone()
                    .unwrap_or_else(|| "localization failed".into())
            },
            hardware_id: String::new(),
            values: vec![
                value("success", outcome.success.to_string()),
                value("inliers", outcome.inlier_count.to_string()),
                value("matches", outcome.match_count.to_string()),
                value("correspondences", outcome.correspondence_count.to_string()),
                value("features", outcome.feature_count.to_string()),
                value(
                    "mean_reprojection_error_px",
                    outcome
                        .reprojection_error
                        .map(|error| format!("{error:.4}"))
                        .unwrap_or_default(),
                ),
                value("processing_ms", format!("{:.2}", outcome.elapsed_ms)),
            ],
        }],
    }
}

/// Builds a visloc camera from `sensor_msgs/CameraInfo`.
///
/// Uses the **unrectified** intrinsics `K` and distortion `D` (the node
/// expects raw images). `plumb_bob` / `rational_polynomial` map to COLMAP's
/// `OPENCV` / `FULL_OPENCV`, `equidistant` to `OPENCV_FISHEYE`; an all-zero
/// or empty `D` gives a plain pinhole.
pub fn camera_from_info(info: &CameraInfo, id: CameraId) -> Result<Camera, String> {
    let (fx, fy, cx, cy) = (info.k[0], info.k[4], info.k[2], info.k[5]);
    if !(fx > 0.0 && fy > 0.0) || info.width == 0 || info.height == 0 {
        return Err("CameraInfo has no valid intrinsics".into());
    }
    let mut d = info.d.clone();
    let no_distortion = d.iter().all(|value| *value == 0.0);
    let (model, params) = if no_distortion {
        (CameraModel::Pinhole, vec![fx, fy, cx, cy])
    } else {
        match info.distortion_model.as_str() {
            "plumb_bob" | "rational_polynomial" => {
                d.resize(8, 0.0);
                if d[4..].iter().all(|value| *value == 0.0) {
                    (
                        CameraModel::OpenCv,
                        vec![fx, fy, cx, cy, d[0], d[1], d[2], d[3]],
                    )
                } else {
                    let mut params = vec![fx, fy, cx, cy];
                    params.extend_from_slice(&d[..8]);
                    (CameraModel::FullOpenCv, params)
                }
            }
            "equidistant" => {
                d.resize(4, 0.0);
                let mut params = vec![fx, fy, cx, cy];
                params.extend_from_slice(&d[..4]);
                (CameraModel::OpenCvFisheye, params)
            }
            other => return Err(format!("unsupported CameraInfo distortion model `{other}`")),
        }
    };
    Ok(Camera {
        id,
        model,
        width: info.width,
        height: info.height,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(model: &str, d: Vec<f64>) -> CameraInfo {
        CameraInfo {
            width: 640,
            height: 480,
            distortion_model: model.into(),
            d,
            k: [500.0, 0.0, 320.0, 0.0, 501.0, 240.0, 0.0, 0.0, 1.0],
            ..CameraInfo::default()
        }
    }

    #[test]
    fn camera_info_models() {
        let pinhole = camera_from_info(&info("plumb_bob", vec![0.0; 5]), 1).unwrap();
        assert_eq!(pinhole.model, CameraModel::Pinhole);
        assert_eq!(pinhole.params, vec![500.0, 501.0, 320.0, 240.0]);
        let opencv =
            camera_from_info(&info("plumb_bob", vec![-0.1, 0.01, 0.001, 0.002, 0.0]), 1).unwrap();
        assert_eq!(opencv.model, CameraModel::OpenCv);
        assert_eq!(opencv.params.len(), 8);
        let full =
            camera_from_info(&info("plumb_bob", vec![-0.1, 0.01, 0.001, 0.002, 0.003]), 1).unwrap();
        assert_eq!(full.model, CameraModel::FullOpenCv);
        assert_eq!(full.params.len(), 12);
        let fisheye = camera_from_info(&info("equidistant", vec![0.1, 0.01, 0.0, 0.0]), 1).unwrap();
        assert_eq!(fisheye.model, CameraModel::OpenCvFisheye);
        assert!(camera_from_info(&info("unknown", vec![0.1]), 1).is_err());
        let mut bad = info("plumb_bob", vec![]);
        bad.k[0] = 0.0;
        assert!(camera_from_info(&bad, 1).is_err());
    }
}
