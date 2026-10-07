use nalgebra::{Point2, Point3, Quaternion, UnitQuaternion, Vector3};
use std::collections::BTreeMap;
use std::fs;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use thiserror::Error;
use visloc_core::geometry::Pose;
use visloc_core::types::{
    Camera, CameraModel, Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation,
    VisualMap, VisualMapValidationReport,
};
use visloc_localization::{DescriptorProvider, MapProvider};
use visloc_vision::features::FeatureSet;
use visloc_vision::stereo_vo::StereoFeature;

use crate::descriptors::{read_landmark_descriptors_txt, DescriptorStoreError};

#[derive(Debug, Error)]
pub enum ColmapError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer parse error: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("float parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("invalid COLMAP line in {file}: {line}")]
    InvalidLine { file: &'static str, line: String },
    #[error("invalid COLMAP binary in {file}: {message}")]
    InvalidBinary { file: &'static str, message: String },
    #[error("invalid COLMAP export input: {0}")]
    InvalidExportInput(String),
}

/// A merged multi-view landmark for COLMAP export: `(world_position,
/// observations)` where each observation is `(frame, left_keypoint_index,
/// pixel)`. See [`write_colmap_reconstruction_for_3dgs`].
pub type ReconstructionLandmark = (Point3<f64>, Vec<(usize, usize, Point2<f64>)>);

/// Summary of a [`write_colmap_text_model_for_3dgs`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct ColmapExportSummary {
    /// Number of camera viewpoints written to `images.txt`.
    pub frame_count: usize,
    /// Number of distinct 3D landmarks written to `points3D.txt`.
    pub landmark_count: usize,
    /// Total number of (frame, keypoint) observations written across all
    /// landmark TRACK[] tails in `points3D.txt`.
    pub observation_count: usize,
}

#[derive(Debug, Error)]
pub enum ColmapMapProviderError {
    #[error("COLMAP error: {0}")]
    Colmap(#[from] ColmapError),
    #[error("descriptor store error: {0}")]
    DescriptorStore(#[from] DescriptorStoreError),
    #[error("COLMAP map validation failed: {0:?}")]
    InvalidMap(VisualMapValidationReport),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColmapMapProvider {
    pub map: VisualMap,
    pub descriptor_store: Option<LandmarkDescriptorStore>,
}

impl ColmapMapProvider {
    pub fn from_text_model_dir(path: impl AsRef<Path>) -> Result<Self, ColmapMapProviderError> {
        Ok(Self {
            map: read_colmap_text_model(path)?,
            descriptor_store: None,
        })
    }

    pub fn from_binary_model_dir(path: impl AsRef<Path>) -> Result<Self, ColmapMapProviderError> {
        Ok(Self {
            map: read_colmap_binary_model(path)?,
            descriptor_store: None,
        })
    }

    pub fn from_text_model_dir_validated(
        path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        let provider = Self::from_text_model_dir(path)?;
        provider
            .validate_map()
            .into_result()
            .map_err(ColmapMapProviderError::InvalidMap)?;
        Ok(provider)
    }

    pub fn from_binary_model_dir_validated(
        path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        let provider = Self::from_binary_model_dir(path)?;
        provider
            .validate_map()
            .into_result()
            .map_err(ColmapMapProviderError::InvalidMap)?;
        Ok(provider)
    }

    pub fn from_text_model_dir_with_descriptors(
        model_path: impl AsRef<Path>,
        descriptor_path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        Ok(Self {
            map: read_colmap_text_model(model_path)?,
            descriptor_store: Some(read_landmark_descriptors_txt(descriptor_path)?),
        })
    }

    pub fn from_text_model_dir_with_descriptors_validated(
        model_path: impl AsRef<Path>,
        descriptor_path: impl AsRef<Path>,
    ) -> Result<Self, ColmapMapProviderError> {
        let provider = Self::from_text_model_dir_with_descriptors(model_path, descriptor_path)?;
        provider
            .validate_for_localization()
            .into_result()
            .map_err(ColmapMapProviderError::InvalidMap)?;
        Ok(provider)
    }

    pub fn validate_map(&self) -> VisualMapValidationReport {
        self.map.validate()
    }

    pub fn validate_for_localization(&self) -> VisualMapValidationReport {
        self.map
            .validate_with_descriptors(self.descriptor_store.as_ref())
    }
}

impl MapProvider for ColmapMapProvider {
    fn visual_map(&self) -> &VisualMap {
        &self.map
    }
}

impl DescriptorProvider for ColmapMapProvider {
    fn landmark_descriptor_store(&self) -> Option<&LandmarkDescriptorStore> {
        self.descriptor_store.as_ref()
    }
}

pub fn read_colmap_text_model(path: impl AsRef<Path>) -> Result<VisualMap, ColmapError> {
    let path = path.as_ref();
    let mut map = VisualMap::new();

    let cameras = parse_cameras_txt(&fs::read_to_string(path.join("cameras.txt"))?)?;
    map.cameras = cameras
        .into_iter()
        .map(|camera| (camera.id, camera))
        .collect();

    let landmarks = parse_points3d_txt(&fs::read_to_string(path.join("points3D.txt"))?)?;
    map.landmarks = landmarks
        .into_iter()
        .map(|landmark| (landmark.id, landmark))
        .collect();

    let keyframes = parse_images_txt(&fs::read_to_string(path.join("images.txt"))?)?;
    map.keyframes = keyframes
        .into_iter()
        .map(|keyframe| (keyframe.frame.id, keyframe))
        .collect();

    Ok(map)
}

pub fn read_colmap_binary_model(path: impl AsRef<Path>) -> Result<VisualMap, ColmapError> {
    let path = path.as_ref();
    let mut map = VisualMap::new();

    let cameras = parse_cameras_bin(&fs::read(path.join("cameras.bin"))?)?;
    map.cameras = cameras
        .into_iter()
        .map(|camera| (camera.id, camera))
        .collect();

    let landmarks = parse_points3d_bin(&fs::read(path.join("points3D.bin"))?)?;
    map.landmarks = landmarks
        .into_iter()
        .map(|landmark| (landmark.id, landmark))
        .collect();

    let keyframes = parse_images_bin(&fs::read(path.join("images.bin"))?)?;
    map.keyframes = keyframes
        .into_iter()
        .map(|keyframe| (keyframe.frame.id, keyframe))
        .collect();

    Ok(map)
}

pub fn write_colmap_text_model(map: &VisualMap, path: impl AsRef<Path>) -> Result<(), ColmapError> {
    let path = path.as_ref();
    for camera in map.cameras.values() {
        validated_colmap_camera_record(camera)?;
    }
    fs::create_dir_all(path)?;
    fs::write(path.join("cameras.txt"), format_cameras_txt(map))?;
    fs::write(path.join("images.txt"), format_images_txt(map))?;
    fs::write(path.join("points3D.txt"), format_points3d_txt(map))?;
    Ok(())
}

/// Write a COLMAP text model suitable for bootstrapping a 3D Gaussian
/// Splatting (3DGS) / NeRF training pipeline.
///
/// The writer materialises `cameras.txt`, `images.txt`, and `points3D.txt`
/// under `out_dir` from a stereo VO output:
///
/// - `camera`: shared intrinsics (replicated as a single COLMAP camera id).
///   Trailing radial `[k1, k2]` on a `Pinhole` is exported as `OPENCV`
///   (see [`colmap_camera_record`]); any other params must already match the
///   COLMAP layout for `camera.model`, otherwise the export is rejected
/// - `poses`: per-frame `world_to_camera` SE3 (one COLMAP image entry each)
/// - `left_features`: left keypoints per frame; only the keypoints that
///   participate in a stereo feature are written to the per-image 2D point
///   list, paired with the matching 3D landmark id
/// - `stereo_per_frame`: triangulated stereo features whose `point_cam`
///   (left-camera frame) is lifted through `pose.camera_to_world()` and
///   aggregated into the sparse `points3D.txt` cloud
/// - `image_name(frame_idx)` supplies the image filename to embed in
///   `images.txt`; this must match the filenames the downstream trainer
///   (gaussian-splatting / nerfstudio) will find under `<dataset>/images/`
///
/// Each stereo feature contributes one COLMAP landmark with one observation;
/// temporal tracks are NOT merged across frames (the trainer optimises the
/// gaussian primitives anyway, so a slightly denser cloud with duplicated
/// landmark ids is the simpler, lower-friction MVP).
pub fn write_colmap_text_model_for_3dgs<F>(
    out_dir: impl AsRef<Path>,
    camera: &Camera,
    poses: &[Pose],
    left_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    image_name: F,
) -> Result<ColmapExportSummary, ColmapError>
where
    F: Fn(usize) -> String,
{
    if poses.len() != left_features.len() || poses.len() != stereo_per_frame.len() {
        return Err(ColmapError::InvalidExportInput(format!(
            "input length mismatch: poses={}, left_features={}, stereo_per_frame={}",
            poses.len(),
            left_features.len(),
            stereo_per_frame.len(),
        )));
    }

    // Reject CameraModel::Unknown(name) that the binary counterpart would
    // also reject, so a caller driving both writers off the same input
    // either gets both files or the same structured error from each.
    let (colmap_model, colmap_params) = validated_colmap_camera_record(camera)?;

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    // cameras.txt — one shared camera.
    let mut cameras_text = String::from("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n");
    cameras_text.push_str(&format_camera_line(camera, &colmap_model, &colmap_params));
    fs::write(out_dir.join("cameras.txt"), cameras_text)?;

    // Aggregate world-frame landmarks (deduplication is intentionally skipped
    // — see the doc comment above).
    let mut landmark_count: u64 = 0;
    let mut observation_count: usize = 0;
    // For each frame, record `(left_keypoint_index, landmark_id)` so the
    // images.txt second line can emit POINTS2D[] in the COLMAP-mandated
    // `X Y POINT3D_ID` triples.
    let mut frame_landmarks: Vec<Vec<(usize, u64)>> = Vec::with_capacity(poses.len());
    let mut points3d_text =
        String::from("# POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_IDX\n");

    for (frame_idx, (pose, features)) in poses.iter().zip(stereo_per_frame.iter()).enumerate() {
        let cam_to_world = pose.camera_to_world();
        let mut per_frame: Vec<(usize, u64)> = Vec::with_capacity(features.len());
        for feature in features {
            landmark_count += 1;
            let landmark_id = landmark_count;
            let world_point = cam_to_world.transform_point(&feature.point_cam);
            points3d_text.push_str(&format!(
                "{} {} {} {} 255 255 255 0 {} {}\n",
                landmark_id,
                format_f64(world_point.x),
                format_f64(world_point.y),
                format_f64(world_point.z),
                frame_idx,
                feature.left_index,
            ));
            observation_count += 1;
            per_frame.push((feature.left_index, landmark_id));
        }
        frame_landmarks.push(per_frame);
    }
    fs::write(out_dir.join("points3D.txt"), points3d_text)?;

    // images.txt — alternating header line + 2D point list.
    let mut images_text = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    for (frame_idx, (pose, features)) in poses.iter().zip(left_features.iter()).enumerate() {
        let q = pose.world_to_camera.rotation.quaternion();
        let t = pose.world_to_camera.translation;
        let name = image_name(frame_idx);
        validate_colmap_image_name(&name, frame_idx)?;
        images_text.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {}\n",
            frame_idx,
            format_f64(q.w),
            format_f64(q.i),
            format_f64(q.j),
            format_f64(q.k),
            format_f64(t.x),
            format_f64(t.y),
            format_f64(t.z),
            camera.id,
            name,
        ));
        // Emit one `X Y POINT3D_ID` triple per left keypoint that participates
        // in a triangulated stereo feature; the rest map to POINT3D_ID = -1.
        let mut by_kp: Vec<(usize, u64)> = frame_landmarks[frame_idx].clone();
        by_kp.sort_by_key(|(kp, _)| *kp);
        let mut tokens: Vec<String> = Vec::with_capacity(features.len());
        for (kp_idx, kp) in features.keypoints.iter().enumerate() {
            let landmark_id = by_kp
                .iter()
                .find(|(observed, _)| *observed == kp_idx)
                .map_or_else(
                    || "-1".to_owned(),
                    |(_, landmark_id)| landmark_id.to_string(),
                );
            tokens.push(format!(
                "{} {} {}",
                format_f64(kp.x),
                format_f64(kp.y),
                landmark_id
            ));
        }
        images_text.push_str(&tokens.join(" "));
        images_text.push('\n');
    }
    fs::write(out_dir.join("images.txt"), images_text)?;

    Ok(ColmapExportSummary {
        frame_count: poses.len(),
        landmark_count: landmark_count as usize,
        observation_count,
    })
}

/// Binary counterpart of [`write_colmap_text_model_for_3dgs`]. Writes
/// `cameras.bin`, `images.bin`, and `points3D.bin` under `out_dir` in
/// the same little-endian layout COLMAP's reference reader expects (see
/// [`parse_cameras_bin`], [`parse_images_bin`], [`parse_points3d_bin`]
/// for the exact byte layouts). Useful for downstream trainers that
/// only ingest the binary form (Inria 3DGS / nerfstudio both accept it
/// transparently). All other semantics — single shared camera, sparse
/// per-frame stereo landmarks lifted through `pose.camera_to_world()`,
/// `frame_idx` as the COLMAP image id, `image_name(frame_idx)` for the
/// `NAME` field — match the text writer exactly.
pub fn write_colmap_binary_model_for_3dgs<F>(
    out_dir: impl AsRef<Path>,
    camera: &Camera,
    poses: &[Pose],
    left_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    image_name: F,
) -> Result<ColmapExportSummary, ColmapError>
where
    F: Fn(usize) -> String,
{
    if poses.len() != left_features.len() || poses.len() != stereo_per_frame.len() {
        return Err(ColmapError::InvalidExportInput(format!(
            "input length mismatch: poses={}, left_features={}, stereo_per_frame={}",
            poses.len(),
            left_features.len(),
            stereo_per_frame.len(),
        )));
    }

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    // cameras.bin — single shared camera (matches the text writer).
    let (colmap_model, colmap_params) = validated_colmap_camera_record(camera)?;
    let model_id = colmap_id_from_camera_model(&colmap_model)?;
    let camera_id_u32 = u32::try_from(camera.id).map_err(|_| {
        ColmapError::InvalidExportInput(format!(
            "camera id {} does not fit in u32 (COLMAP binary cameras use u32 ids)",
            camera.id
        ))
    })?;
    let mut cameras_bytes: Vec<u8> = Vec::new();
    cameras_bytes.extend_from_slice(&1u64.to_le_bytes());
    cameras_bytes.extend_from_slice(&camera_id_u32.to_le_bytes());
    cameras_bytes.extend_from_slice(&model_id.to_le_bytes());
    cameras_bytes.extend_from_slice(&(camera.width as u64).to_le_bytes());
    cameras_bytes.extend_from_slice(&(camera.height as u64).to_le_bytes());
    for param in &colmap_params {
        cameras_bytes.extend_from_slice(&param.to_le_bytes());
    }
    fs::write(out_dir.join("cameras.bin"), cameras_bytes)?;

    // Aggregate world-frame landmarks (deduplication is intentionally skipped
    // — see the doc comment on `write_colmap_text_model_for_3dgs`).
    let mut landmark_count: u64 = 0;
    let mut observation_count: usize = 0;
    let mut frame_landmarks: Vec<Vec<(usize, u64)>> = Vec::with_capacity(poses.len());
    // We hold landmark records in memory so we can write the total count up
    // front (the binary format requires `count` before the records).
    let mut landmark_records: Vec<(u64, Point3<f64>, u32, u32)> = Vec::new();

    for (frame_idx, (pose, features)) in poses.iter().zip(stereo_per_frame.iter()).enumerate() {
        let image_id_u32 = u32::try_from(frame_idx).map_err(|_| {
            ColmapError::InvalidExportInput(format!(
                "frame index {frame_idx} does not fit in u32 (COLMAP binary uses u32 image ids)"
            ))
        })?;
        let cam_to_world = pose.camera_to_world();
        let mut per_frame: Vec<(usize, u64)> = Vec::with_capacity(features.len());
        for feature in features {
            landmark_count += 1;
            let landmark_id = landmark_count;
            let world_point = cam_to_world.transform_point(&feature.point_cam);
            let kp_index_u32 = u32::try_from(feature.left_index).map_err(|_| {
                ColmapError::InvalidExportInput(format!(
                    "left keypoint index {} does not fit in u32",
                    feature.left_index
                ))
            })?;
            landmark_records.push((landmark_id, world_point, image_id_u32, kp_index_u32));
            observation_count += 1;
            per_frame.push((feature.left_index, landmark_id));
        }
        frame_landmarks.push(per_frame);
    }

    // points3D.bin: u64 count, then for each landmark:
    //   u64 id, f64 x y z, u8 r g b (3 bytes), f64 error, u64 track_length,
    //   [u32 image_id, u32 point2d_idx]*
    let mut points_bytes: Vec<u8> = Vec::new();
    points_bytes.extend_from_slice(&(landmark_records.len() as u64).to_le_bytes());
    for (landmark_id, position, image_id_u32, kp_index_u32) in &landmark_records {
        points_bytes.extend_from_slice(&landmark_id.to_le_bytes());
        points_bytes.extend_from_slice(&position.x.to_le_bytes());
        points_bytes.extend_from_slice(&position.y.to_le_bytes());
        points_bytes.extend_from_slice(&position.z.to_le_bytes());
        points_bytes.extend_from_slice(&[255u8, 255u8, 255u8]); // white RGB
        points_bytes.extend_from_slice(&0.0f64.to_le_bytes()); // error
        points_bytes.extend_from_slice(&1u64.to_le_bytes()); // track length
        points_bytes.extend_from_slice(&image_id_u32.to_le_bytes());
        points_bytes.extend_from_slice(&kp_index_u32.to_le_bytes());
    }
    fs::write(out_dir.join("points3D.bin"), points_bytes)?;

    // images.bin: u64 count, then for each image:
    //   u32 frame_id, f64 qw qx qy qz, f64 tx ty tz, u32 camera_id,
    //   NULL-terminated NAME, u64 points2d_count, [f64 x, f64 y, i64 point3d_id]*
    let mut images_bytes: Vec<u8> = Vec::new();
    images_bytes.extend_from_slice(&(poses.len() as u64).to_le_bytes());
    for (frame_idx, (pose, features)) in poses.iter().zip(left_features.iter()).enumerate() {
        let image_id_u32 = u32::try_from(frame_idx).map_err(|_| {
            ColmapError::InvalidExportInput(format!(
                "frame index {frame_idx} does not fit in u32 (COLMAP binary uses u32 image ids)"
            ))
        })?;
        images_bytes.extend_from_slice(&image_id_u32.to_le_bytes());
        let q = pose.world_to_camera.rotation.quaternion();
        let t = pose.world_to_camera.translation;
        for v in [q.w, q.i, q.j, q.k, t.x, t.y, t.z] {
            images_bytes.extend_from_slice(&v.to_le_bytes());
        }
        images_bytes.extend_from_slice(&camera_id_u32.to_le_bytes());
        let name = image_name(frame_idx);
        validate_colmap_image_name(&name, frame_idx)?;
        images_bytes.extend_from_slice(name.as_bytes());
        images_bytes.push(0u8); // NUL terminator

        let by_kp = &frame_landmarks[frame_idx];
        images_bytes.extend_from_slice(&(features.keypoints.len() as u64).to_le_bytes());
        for (kp_idx, kp) in features.keypoints.iter().enumerate() {
            images_bytes.extend_from_slice(&kp.x.to_le_bytes());
            images_bytes.extend_from_slice(&kp.y.to_le_bytes());
            let landmark_id = by_kp
                .iter()
                .find(|(observed, _)| *observed == kp_idx)
                .map_or(-1, |(_, lid)| *lid as i64);
            images_bytes.extend_from_slice(&landmark_id.to_le_bytes());
        }
    }
    fs::write(out_dir.join("images.bin"), images_bytes)?;

    Ok(ColmapExportSummary {
        frame_count: poses.len(),
        landmark_count: landmark_count as usize,
        observation_count,
    })
}

/// Write a COLMAP text model from a *merged-track* sparse reconstruction.
///
/// Unlike [`write_colmap_text_model_for_3dgs`] — which lifts one fresh landmark
/// per frame (every `POINT3D` has a single-element `TRACK[]`) — this writer takes
/// landmarks that are already merged across views by a global bundle adjustment:
/// each `landmarks[i]` is `(world_position, observations)` where `observations`
/// is the full `(frame, left_keypoint_index, pixel)` track. The emitted
/// `points3D.txt` therefore carries genuine multi-view `TRACK[]` tails, which is
/// what lets a downstream 3DGS optimizer converge to crisp geometry instead of
/// the per-frame depth-lift fog.
///
/// `left_features[frame]` supplies the full keypoint list so `images.txt` can map
/// each keypoint to its `POINT3D_ID` (or `-1` when untracked); the COLMAP
/// `POINT2D_IDX` recorded in each track equals the keypoint index, since
/// `POINTS2D[]` is emitted in keypoint order.
pub fn write_colmap_reconstruction_for_3dgs<F>(
    out_dir: impl AsRef<Path>,
    camera: &Camera,
    poses: &[Pose],
    left_features: &[FeatureSet],
    landmarks: &[ReconstructionLandmark],
    image_name: F,
) -> Result<ColmapExportSummary, ColmapError>
where
    F: Fn(usize) -> String,
{
    if poses.len() != left_features.len() {
        return Err(ColmapError::InvalidExportInput(format!(
            "input length mismatch: poses={}, left_features={}",
            poses.len(),
            left_features.len(),
        )));
    }
    let (colmap_model, colmap_params) = validated_colmap_camera_record(camera)?;

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    // cameras.txt — one shared camera.
    let mut cameras_text = String::from("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n");
    cameras_text.push_str(&format_camera_line(camera, &colmap_model, &colmap_params));
    fs::write(out_dir.join("cameras.txt"), cameras_text)?;

    // For each frame, map left-keypoint index -> assigned POINT3D_ID so the
    // images.txt second line can emit the `X Y POINT3D_ID` triples.
    let mut frame_kp_to_point: Vec<std::collections::HashMap<usize, u64>> =
        vec![std::collections::HashMap::new(); poses.len()];
    let mut observation_count: usize = 0;

    // points3D.txt — one line per merged landmark, with the full multi-view TRACK[].
    let mut points3d_text =
        String::from("# POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_IDX\n");
    for (idx, (position, observations)) in landmarks.iter().enumerate() {
        let point_id = (idx as u64) + 1;
        points3d_text.push_str(&format!(
            "{} {} {} {} 255 255 255 0",
            point_id,
            format_f64(position.x),
            format_f64(position.y),
            format_f64(position.z),
        ));
        for &(frame, left_idx, _) in observations {
            if frame >= poses.len() {
                continue;
            }
            // POINT2D_IDX == keypoint index, because images.txt emits POINTS2D[]
            // in keypoint order below.
            points3d_text.push_str(&format!(" {} {}", frame, left_idx));
            frame_kp_to_point[frame].insert(left_idx, point_id);
            observation_count += 1;
        }
        points3d_text.push('\n');
    }
    fs::write(out_dir.join("points3D.txt"), points3d_text)?;

    // images.txt — alternating header line + 2D point list.
    let mut images_text = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    for (frame_idx, (pose, features)) in poses.iter().zip(left_features.iter()).enumerate() {
        let q = pose.world_to_camera.rotation.quaternion();
        let t = pose.world_to_camera.translation;
        let name = image_name(frame_idx);
        validate_colmap_image_name(&name, frame_idx)?;
        images_text.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {}\n",
            frame_idx,
            format_f64(q.w),
            format_f64(q.i),
            format_f64(q.j),
            format_f64(q.k),
            format_f64(t.x),
            format_f64(t.y),
            format_f64(t.z),
            camera.id,
            name,
        ));
        let kp_to_point = &frame_kp_to_point[frame_idx];
        let mut tokens: Vec<String> = Vec::with_capacity(features.keypoints.len());
        for (kp_idx, kp) in features.keypoints.iter().enumerate() {
            let point_id = kp_to_point
                .get(&kp_idx)
                .map_or_else(|| "-1".to_owned(), |id| id.to_string());
            tokens.push(format!(
                "{} {} {}",
                format_f64(kp.x),
                format_f64(kp.y),
                point_id
            ));
        }
        images_text.push_str(&tokens.join(" "));
        images_text.push('\n');
    }
    fs::write(out_dir.join("images.txt"), images_text)?;

    Ok(ColmapExportSummary {
        frame_count: poses.len(),
        landmark_count: landmarks.len(),
        observation_count,
    })
}

/// Write a merged-track COLMAP text model with one calibrated camera per
/// output image.  This is the multi-camera counterpart to
/// [`write_colmap_reconstruction_for_3dgs`]; all camera ids and intrinsics are
/// retained, and each `images.txt` header points at the camera corresponding
/// to that frame.  The legacy single-camera writer is intentionally untouched.
pub fn write_colmap_reconstruction_for_3dgs_with_cameras<F>(
    out_dir: impl AsRef<Path>,
    cameras: &[Camera],
    poses: &[Pose],
    left_features: &[FeatureSet],
    landmarks: &[ReconstructionLandmark],
    image_name: F,
) -> Result<ColmapExportSummary, ColmapError>
where
    F: Fn(usize) -> String,
{
    if cameras.len() != poses.len() || poses.len() != left_features.len() {
        return Err(ColmapError::InvalidExportInput(format!(
            "input length mismatch: cameras={}, poses={}, left_features={}",
            cameras.len(),
            poses.len(),
            left_features.len(),
        )));
    }
    let mut unique_cameras = BTreeMap::<u64, Camera>::new();
    for camera in cameras {
        validated_colmap_camera_record(camera)?;
        if camera.width == 0 || camera.height == 0 || camera.params.iter().any(|p| !p.is_finite()) {
            return Err(ColmapError::InvalidExportInput(format!(
                "camera {} has invalid dimensions or non-finite parameters",
                camera.id
            )));
        }
        if let Some(previous) = unique_cameras.insert(camera.id, camera.clone()) {
            if previous != *camera {
                return Err(ColmapError::InvalidExportInput(format!(
                    "camera id {} is assigned incompatible definitions",
                    camera.id
                )));
            }
        }
    }

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let mut cameras_text = String::from("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n");
    for camera in unique_cameras.values() {
        let (model, params) = colmap_camera_record(camera);
        cameras_text.push_str(&format_camera_line(camera, &model, &params));
    }
    fs::write(out_dir.join("cameras.txt"), cameras_text)?;

    let mut frame_kp_to_point: Vec<BTreeMap<usize, u64>> = vec![BTreeMap::new(); poses.len()];
    let mut observation_count = 0usize;
    let mut points3d_text =
        String::from("# POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_IDX\n");
    for (landmark_index, (position, observations)) in landmarks.iter().enumerate() {
        let point_id = landmark_index as u64 + 1;
        points3d_text.push_str(&format!(
            "{} {} {} {} 255 255 255 0",
            point_id,
            format_f64(position.x),
            format_f64(position.y),
            format_f64(position.z),
        ));
        for &(frame, keypoint, _) in observations {
            if frame >= poses.len() || keypoint >= left_features[frame].keypoints.len() {
                return Err(ColmapError::InvalidExportInput(format!(
                    "landmark {} references invalid observation ({}, {})",
                    point_id, frame, keypoint
                )));
            }
            if frame_kp_to_point[frame]
                .insert(keypoint, point_id)
                .is_some()
            {
                return Err(ColmapError::InvalidExportInput(format!(
                    "keypoint ({}, {}) belongs to multiple exported landmarks",
                    frame, keypoint
                )));
            }
            points3d_text.push_str(&format!(" {} {}", frame, keypoint));
            observation_count += 1;
        }
        points3d_text.push('\n');
    }
    fs::write(out_dir.join("points3D.txt"), points3d_text)?;

    let mut images_text = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    for (frame, ((pose, features), camera)) in
        poses.iter().zip(left_features).zip(cameras).enumerate()
    {
        let q = pose.world_to_camera.rotation.quaternion();
        let t = pose.world_to_camera.translation;
        let name = image_name(frame);
        validate_colmap_image_name(&name, frame)?;
        images_text.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {}\n",
            frame,
            format_f64(q.w),
            format_f64(q.i),
            format_f64(q.j),
            format_f64(q.k),
            format_f64(t.x),
            format_f64(t.y),
            format_f64(t.z),
            camera.id,
            name,
        ));
        let point_ids = &frame_kp_to_point[frame];
        let points = features
            .keypoints
            .iter()
            .enumerate()
            .map(|(keypoint, pixel)| {
                format!(
                    "{} {} {}",
                    format_f64(pixel.x),
                    format_f64(pixel.y),
                    point_ids
                        .get(&keypoint)
                        .map_or_else(|| "-1".to_owned(), ToString::to_string)
                )
            })
            .collect::<Vec<_>>();
        images_text.push_str(&points.join(" "));
        images_text.push('\n');
    }
    fs::write(out_dir.join("images.txt"), images_text)?;

    Ok(ColmapExportSummary {
        frame_count: poses.len(),
        landmark_count: landmarks.len(),
        observation_count,
    })
}

pub fn format_cameras_txt(map: &VisualMap) -> String {
    let mut output = String::from("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n");
    let mut cameras = map.cameras.values().collect::<Vec<_>>();
    cameras.sort_by_key(|camera| camera.id);
    for camera in cameras {
        let (model, params) = colmap_camera_record(camera);
        output.push_str(&format_camera_line(camera, &model, &params));
    }
    output
}

pub fn format_images_txt(map: &VisualMap) -> String {
    let mut output = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    let mut keyframes = map.keyframes.values().collect::<Vec<_>>();
    keyframes.sort_by_key(|keyframe| keyframe.frame.id);
    for keyframe in keyframes {
        let pose = keyframe.frame.pose.clone().unwrap_or_default();
        let q = pose.world_to_camera.rotation.quaternion();
        let t = pose.world_to_camera.translation;
        output.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} image_{}.jpg\n",
            keyframe.frame.id,
            format_f64(q.w),
            format_f64(q.i),
            format_f64(q.j),
            format_f64(q.k),
            format_f64(t.x),
            format_f64(t.y),
            format_f64(t.z),
            keyframe.frame.camera_id,
            keyframe.frame.id,
        ));

        let mut observation_by_keypoint = keyframe
            .observations
            .iter()
            .map(|observation| (observation.keypoint_index, observation.landmark_id))
            .collect::<Vec<_>>();
        observation_by_keypoint.sort_by_key(|(keypoint_index, _)| *keypoint_index);
        let point_tokens = keyframe
            .frame
            .keypoints
            .iter()
            .enumerate()
            .map(|(keypoint_index, xy)| {
                let landmark_id = observation_by_keypoint
                    .iter()
                    .find(|(observed_index, _)| *observed_index == keypoint_index)
                    .map_or_else(
                        || "-1".to_owned(),
                        |(_, landmark_id)| landmark_id.to_string(),
                    );
                format!("{} {} {}", format_f64(xy.x), format_f64(xy.y), landmark_id)
            })
            .collect::<Vec<_>>();
        output.push_str(&point_tokens.join(" "));
        output.push('\n');
    }
    output
}

pub fn format_points3d_txt(map: &VisualMap) -> String {
    let mut output =
        String::from("# POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_IDX\n");
    let mut landmarks = map.landmarks.values().collect::<Vec<_>>();
    landmarks.sort_by_key(|landmark| landmark.id);
    for landmark in landmarks {
        output.push_str(&format!(
            "{} {} {} {} 255 255 255 0",
            landmark.id,
            format_f64(landmark.position.x),
            format_f64(landmark.position.y),
            format_f64(landmark.position.z),
        ));
        let mut observations = landmark.observations.iter().collect::<Vec<_>>();
        observations.sort_by_key(|observation| (observation.frame_id, observation.keypoint_index));
        for observation in observations {
            output.push_str(&format!(
                " {} {}",
                observation.frame_id, observation.keypoint_index
            ));
        }
        output.push('\n');
    }
    output
}

pub fn parse_cameras_txt(contents: &str) -> Result<Vec<Camera>, ColmapError> {
    let mut cameras = Vec::new();
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 5 {
            return Err(ColmapError::InvalidLine {
                file: "cameras.txt",
                line: line.to_owned(),
            });
        }
        let id = tokens[0].parse()?;
        let model = CameraModel::from_colmap_name(tokens[1]);
        let width = tokens[2].parse()?;
        let height = tokens[3].parse()?;
        let params = tokens[4..]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        cameras.push(Camera {
            id,
            model,
            width,
            height,
            params,
        });
    }
    Ok(cameras)
}

pub fn parse_points3d_txt(contents: &str) -> Result<Vec<Landmark>, ColmapError> {
    let mut landmarks = Vec::new();
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 8 {
            return Err(ColmapError::InvalidLine {
                file: "points3D.txt",
                line: line.to_owned(),
            });
        }
        landmarks.push(Landmark::new(
            tokens[0].parse()?,
            Point3::new(tokens[1].parse()?, tokens[2].parse()?, tokens[3].parse()?),
        ));
    }
    Ok(landmarks)
}

pub fn parse_images_txt(contents: &str) -> Result<Vec<Keyframe>, ColmapError> {
    let mut keyframes = Vec::new();
    let mut lines = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with('#'));

    while let Some(header) = lines.next() {
        if header.is_empty() {
            continue;
        }

        let tokens = header.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 10 {
            return Err(ColmapError::InvalidLine {
                file: "images.txt",
                line: header.to_owned(),
            });
        }

        let frame_id = tokens[0].parse()?;
        let qw = tokens[1].parse()?;
        let qx = tokens[2].parse()?;
        let qy = tokens[3].parse()?;
        let qz = tokens[4].parse()?;
        let tx = tokens[5].parse()?;
        let ty = tokens[6].parse()?;
        let tz = tokens[7].parse()?;
        let camera_id = tokens[8].parse()?;
        let points_line = lines.next().unwrap_or("");

        let mut frame = Frame::new(frame_id, camera_id);
        frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
            Vector3::new(tx, ty, tz),
        ));

        let point_tokens = points_line.split_whitespace().collect::<Vec<_>>();
        if point_tokens.len() % 3 != 0 {
            return Err(ColmapError::InvalidLine {
                file: "images.txt",
                line: points_line.to_owned(),
            });
        }

        let mut observations = Vec::new();
        for (keypoint_index, chunk) in point_tokens.chunks(3).enumerate() {
            let xy = Point2::new(chunk[0].parse()?, chunk[1].parse()?);
            frame.keypoints.push(xy);

            let point_id = chunk[2].parse::<i64>()?;
            if point_id >= 0 {
                observations.push(Observation {
                    frame_id,
                    landmark_id: point_id as u64,
                    keypoint_index,
                    xy,
                });
            }
        }

        keyframes.push(Keyframe {
            frame,
            observations,
        });
    }

    Ok(keyframes)
}

pub fn parse_cameras_bin(contents: &[u8]) -> Result<Vec<Camera>, ColmapError> {
    // A camera record contains the fixed fields plus at least the three
    // parameters required by every model accepted by `camera_model_from_colmap_id`.
    const MIN_CAMERA_RECORD_BYTES: usize = 4 + 4 + 8 + 8 + (3 * 8);
    let mut reader = BinaryReader::new("cameras.bin", contents);
    let camera_count = reader.read_count_with_min_bytes("camera_count", MIN_CAMERA_RECORD_BYTES)?;
    let mut cameras = Vec::new();
    reader.try_reserve(&mut cameras, camera_count, "camera_count")?;

    for _ in 0..camera_count {
        let id = reader.read_u32()? as u64;
        let model_id = reader.read_i32()?;
        let width = u32::try_from(reader.read_u64()?).map_err(|_| ColmapError::InvalidBinary {
            file: "cameras.bin",
            message: "camera width does not fit in u32".to_owned(),
        })?;
        let height = u32::try_from(reader.read_u64()?).map_err(|_| ColmapError::InvalidBinary {
            file: "cameras.bin",
            message: "camera height does not fit in u32".to_owned(),
        })?;
        let (model, parameter_count) = camera_model_from_colmap_id(model_id)?;
        let mut params = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count {
            params.push(reader.read_f64()?);
        }
        cameras.push(Camera {
            id,
            model,
            width,
            height,
            params,
        });
    }

    reader.finish()?;
    Ok(cameras)
}

pub fn parse_images_bin(contents: &[u8]) -> Result<Vec<Keyframe>, ColmapError> {
    // An image with an empty NAME and no POINTS2D still occupies these fixed
    // fields. Each POINTS2D entry is two f64 coordinates plus one i64 id.
    const MIN_IMAGE_RECORD_BYTES: usize = 4 + (7 * 8) + 4 + 1 + 8;
    const IMAGE_POINT2D_BYTES: usize = 3 * 8;
    let mut reader = BinaryReader::new("images.bin", contents);
    let image_count = reader.read_count_with_min_bytes("image_count", MIN_IMAGE_RECORD_BYTES)?;
    let mut keyframes = Vec::new();
    reader.try_reserve(&mut keyframes, image_count, "image_count")?;

    for _ in 0..image_count {
        let frame_id = reader.read_u32()? as u64;
        let qw = reader.read_f64()?;
        let qx = reader.read_f64()?;
        let qy = reader.read_f64()?;
        let qz = reader.read_f64()?;
        let tx = reader.read_f64()?;
        let ty = reader.read_f64()?;
        let tz = reader.read_f64()?;
        let camera_id = reader.read_u32()? as u64;
        reader.read_null_terminated_string()?;

        let mut frame = Frame::new(frame_id, camera_id);
        frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
            Vector3::new(tx, ty, tz),
        ));

        let point_count = reader.read_count_with_min_bytes("point_count", IMAGE_POINT2D_BYTES)?;
        reader.try_reserve(&mut frame.keypoints, point_count, "point_count keypoints")?;
        let mut observations = Vec::new();
        reader.try_reserve(&mut observations, point_count, "point_count observations")?;
        for keypoint_index in 0..point_count {
            let xy = Point2::new(reader.read_f64()?, reader.read_f64()?);
            frame.keypoints.push(xy);

            let point_id = reader.read_i64()?;
            if point_id >= 0 {
                observations.push(Observation {
                    frame_id,
                    landmark_id: point_id as u64,
                    keypoint_index,
                    xy,
                });
            }
        }

        keyframes.push(Keyframe {
            frame,
            observations,
        });
    }

    reader.finish()?;
    Ok(keyframes)
}

pub fn parse_points3d_bin(contents: &[u8]) -> Result<Vec<Landmark>, ColmapError> {
    // A POINT3D record contains id, xyz, RGB, error, and track_length before
    // its variable-length track entries. Each track entry is (image_id,
    // point2D_idx), two u32 values.
    const MIN_POINT3D_RECORD_BYTES: usize = 8 + (3 * 8) + 3 + 8 + 8;
    const POINT3D_TRACK_ENTRY_BYTES: usize = 2 * 4;
    let mut reader = BinaryReader::new("points3D.bin", contents);
    let point_count =
        reader.read_count_with_min_bytes("point3D_count", MIN_POINT3D_RECORD_BYTES)?;
    let mut landmarks = Vec::new();
    reader.try_reserve(&mut landmarks, point_count, "point3D_count")?;

    for _ in 0..point_count {
        let id = reader.read_u64()?;
        let position = Point3::new(reader.read_f64()?, reader.read_f64()?, reader.read_f64()?);
        reader.skip(3)?;
        let _error = reader.read_f64()?;
        let track_length =
            reader.read_count_with_min_bytes("track_length", POINT3D_TRACK_ENTRY_BYTES)?;
        for _ in 0..track_length {
            let _image_id = reader.read_u32()?;
            let _point2d_index = reader.read_u32()?;
        }
        landmarks.push(Landmark::new(id, position));
    }

    reader.finish()?;
    Ok(landmarks)
}

/// Number of `PARAMS[]` the COLMAP camera model carries in `cameras.txt` /
/// `cameras.bin`, or `None` for models COLMAP does not define.
fn colmap_param_count(model: &CameraModel) -> Option<usize> {
    Some(match model {
        CameraModel::SimplePinhole => 3,
        CameraModel::Pinhole => 4,
        CameraModel::SimpleRadial => 4,
        CameraModel::Radial => 5,
        CameraModel::OpenCv => 8,
        CameraModel::OpenCvFisheye => 8,
        CameraModel::FullOpenCv => 12,
        CameraModel::Fov => 5,
        CameraModel::SimpleRadialFisheye => 4,
        CameraModel::RadialFisheye => 5,
        CameraModel::DoubleSphere | CameraModel::Unknown(_) => return None,
    })
}

/// Map a camera onto the `(model, params)` record COLMAP expects.
///
/// visloc keeps self-calibrated radial distortion as two trailing `[k1, k2]`
/// slots on a `Pinhole` (see [`Camera::pinhole_radial`]) or a 6-parameter
/// `OpenCv`. COLMAP's `PINHOLE` has exactly four parameters and `OPENCV`
/// exactly eight, so writing those params verbatim produces a model file
/// that COLMAP rejects (or whose distortion other readers silently drop).
/// Such cameras are exported as `OPENCV` with `p1 = p2 = 0`, which is the
/// identical projection; an all-zero distortion tail on a `Pinhole`
/// collapses back to plain `PINHOLE`. Every other camera is returned as is.
pub fn colmap_camera_record(camera: &Camera) -> (CameraModel, Vec<f64>) {
    let params = &camera.params;
    match camera.model {
        CameraModel::Pinhole if params.len() > 4 && params[4..].iter().all(|&k| k == 0.0) => {
            (CameraModel::Pinhole, params[..4].to_vec())
        }
        CameraModel::Pinhole | CameraModel::OpenCv if params.len() > 4 && params.len() < 8 => {
            let mut opencv = params.clone();
            opencv.resize(8, 0.0);
            (CameraModel::OpenCv, opencv)
        }
        _ => (camera.model.clone(), params.clone()),
    }
}

/// [`colmap_camera_record`] plus the checks every fallible writer needs: the
/// model must exist in COLMAP and carry exactly COLMAP's parameter count.
fn validated_colmap_camera_record(camera: &Camera) -> Result<(CameraModel, Vec<f64>), ColmapError> {
    let (model, params) = colmap_camera_record(camera);
    colmap_id_from_camera_model(&model)?;
    let expected = colmap_param_count(&model).ok_or_else(|| {
        ColmapError::InvalidExportInput(format!(
            "camera {} model {} is not a COLMAP camera model",
            camera.id,
            camera_model_to_colmap_name(&model)
        ))
    })?;
    if params.len() != expected {
        return Err(ColmapError::InvalidExportInput(format!(
            "camera {} model {} needs {expected} params for COLMAP, got {}",
            camera.id,
            camera_model_to_colmap_name(&model),
            params.len()
        )));
    }
    Ok((model, params))
}

/// One `cameras.txt` line (with trailing newline).
fn format_camera_line(camera: &Camera, model: &CameraModel, params: &[f64]) -> String {
    let mut line = format!(
        "{} {} {} {}",
        camera.id,
        camera_model_to_colmap_name(model),
        camera.width,
        camera.height,
    );
    for param in params {
        line.push_str(&format!(" {}", format_f64(*param)));
    }
    line.push('\n');
    line
}

fn camera_model_to_colmap_name(model: &CameraModel) -> &str {
    match model {
        CameraModel::Pinhole => "PINHOLE",
        CameraModel::SimplePinhole => "SIMPLE_PINHOLE",
        CameraModel::SimpleRadial => "SIMPLE_RADIAL",
        CameraModel::Radial => "RADIAL",
        CameraModel::OpenCv => "OPENCV",
        CameraModel::FullOpenCv => "FULL_OPENCV",
        CameraModel::OpenCvFisheye => "OPENCV_FISHEYE",
        CameraModel::SimpleRadialFisheye => "SIMPLE_RADIAL_FISHEYE",
        CameraModel::RadialFisheye => "RADIAL_FISHEYE",
        CameraModel::Fov => "FOV",
        // Double Sphere is not a COLMAP model; the writers reject it through
        // `colmap_id_from_camera_model` before this name is ever emitted.
        CameraModel::DoubleSphere => "DOUBLE_SPHERE",
        CameraModel::Unknown(name) => name.as_str(),
    }
}

fn format_f64(value: f64) -> String {
    let formatted = format!("{value:.12}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

/// Reject image NAME characters that would corrupt either writer surface.
///
/// - NUL terminates the binary NAME field early, silently truncating it.
/// - ASCII whitespace breaks the text format's space-separated tokens.
/// - LF / CR would inject a spurious image record into the text file (the
///   reader's alternating header + 2D-points line layout depends on
///   exactly one image per pair of lines).
///
/// Sharing the rule across both writers means any `image_name` closure
/// accepted by one is also accepted by the other — what makes it safe to
/// drive the text/binary writers from a single VO run against the same
/// input (as the KITTI → COLMAP → 3DGS smoke harness does).
fn validate_colmap_image_name(name: &str, frame_idx: usize) -> Result<(), ColmapError> {
    if let Some(bad_idx) = name
        .bytes()
        .position(|b| matches!(b, 0 | b' ' | b'\t' | b'\n' | b'\r'))
    {
        return Err(ColmapError::InvalidExportInput(format!(
            "image name for frame {frame_idx} contains an invalid character at byte {bad_idx} (NUL or ASCII whitespace would corrupt the COLMAP export); got {name:?}"
        )));
    }
    Ok(())
}

fn colmap_id_from_camera_model(model: &CameraModel) -> Result<i32, ColmapError> {
    Ok(match model {
        CameraModel::SimplePinhole => 0,
        CameraModel::Pinhole => 1,
        CameraModel::SimpleRadial => 2,
        CameraModel::Radial => 3,
        CameraModel::OpenCv => 4,
        CameraModel::FullOpenCv => 6,
        CameraModel::OpenCvFisheye => 5,
        CameraModel::Fov => 7,
        CameraModel::SimpleRadialFisheye => 8,
        CameraModel::RadialFisheye => 9,
        CameraModel::DoubleSphere => {
            return Err(ColmapError::InvalidExportInput(
                "Double Sphere is not a COLMAP camera model".to_owned(),
            ));
        }
        CameraModel::Unknown(name) => match name.as_str() {
            "OPENCV_FISHEYE" => 5,
            "FULL_OPENCV" => 6,
            "FOV" => 7,
            "SIMPLE_RADIAL_FISHEYE" => 8,
            "RADIAL_FISHEYE" => 9,
            "THIN_PRISM_FISHEYE" => 10,
            other => {
                return Err(ColmapError::InvalidExportInput(format!(
                    "cannot encode CameraModel::Unknown({other:?}) as a COLMAP binary model id"
                )));
            }
        },
    })
}

fn camera_model_from_colmap_id(model_id: i32) -> Result<(CameraModel, usize), ColmapError> {
    match model_id {
        0 => Ok((CameraModel::SimplePinhole, 3)),
        1 => Ok((CameraModel::Pinhole, 4)),
        2 => Ok((CameraModel::SimpleRadial, 4)),
        3 => Ok((CameraModel::Radial, 5)),
        4 => Ok((CameraModel::OpenCv, 8)),
        5 => Ok((CameraModel::OpenCvFisheye, 8)),
        6 => Ok((CameraModel::FullOpenCv, 12)),
        7 => Ok((CameraModel::Fov, 5)),
        8 => Ok((CameraModel::SimpleRadialFisheye, 4)),
        9 => Ok((CameraModel::RadialFisheye, 5)),
        10 => Ok((CameraModel::Unknown("THIN_PRISM_FISHEYE".to_owned()), 12)),
        other => Err(ColmapError::InvalidBinary {
            file: "cameras.bin",
            message: format!("unsupported camera model id {other}"),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod multi_camera_writer_tests {
    use super::*;

    #[test]
    fn fisheye_camera_models_round_trip_through_colmap_ids() {
        for (model, id, parameter_count) in [
            (CameraModel::OpenCvFisheye, 5, 8),
            (CameraModel::FullOpenCv, 6, 12),
            (CameraModel::Fov, 7, 5),
            (CameraModel::SimpleRadialFisheye, 8, 4),
            (CameraModel::RadialFisheye, 9, 5),
        ] {
            assert_eq!(colmap_id_from_camera_model(&model).unwrap(), id);
            let (decoded, count) = camera_model_from_colmap_id(id).unwrap();
            assert_eq!(decoded, model);
            assert_eq!(count, parameter_count);
            assert_eq!(
                camera_model_to_colmap_name(&model),
                decoded.colmap_name().unwrap()
            );
        }
        // Double Sphere is not a COLMAP model.
        assert!(colmap_id_from_camera_model(&CameraModel::DoubleSphere).is_err());
    }

    #[test]
    fn reconstruction_writer_preserves_per_image_camera_ids() {
        let root =
            std::env::temp_dir().join(format!("visloc_multi_camera_writer_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let cameras = vec![
            Camera::pinhole(7, 100, 80, 50.0, 50.0, 50.0, 40.0),
            Camera::pinhole(9, 200, 160, 100.0, 80.0, 100.0, 80.0),
        ];
        let poses = vec![Pose::default(), Pose::default()];
        let features = vec![
            FeatureSet::new(vec![Point2::new(50.0, 40.0)], vec![vec![1.0]]).unwrap(),
            FeatureSet::new(vec![Point2::new(100.0, 80.0)], vec![vec![2.0]]).unwrap(),
        ];
        let landmarks = vec![
            (
                Point3::new(0.0, 0.0, 5.0),
                vec![(0usize, 0usize, Point2::new(50.0, 40.0))],
            ),
            (
                Point3::new(0.0, 0.0, 5.0),
                vec![(1usize, 0usize, Point2::new(100.0, 80.0))],
            ),
        ];
        let summary = write_colmap_reconstruction_for_3dgs_with_cameras(
            &root,
            &cameras,
            &poses,
            &features,
            &landmarks,
            |index| format!("image_{index}.png"),
        )
        .unwrap();
        assert_eq!(summary.frame_count, 2);
        assert_eq!(
            parse_cameras_txt(&fs::read_to_string(root.join("cameras.txt")).unwrap())
                .unwrap()
                .iter()
                .map(|camera| camera.id)
                .collect::<Vec<_>>(),
            vec![7, 9]
        );
        let headers = fs::read_to_string(root.join("images.txt"))
            .unwrap()
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .step_by(2)
            .map(|line| {
                line.split_whitespace()
                    .nth(8)
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(headers, vec![7, 9]);
        let _ = fs::remove_dir_all(root);
    }
}

struct BinaryReader<'a> {
    file: &'static str,
    contents: &'a [u8],
    offset: usize,
}

impl<'a> BinaryReader<'a> {
    const fn new(file: &'static str, contents: &'a [u8]) -> Self {
        Self {
            file,
            contents,
            offset: 0,
        }
    }

    fn read_u32(&mut self) -> Result<u32, ColmapError> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, ColmapError> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, ColmapError> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64, ColmapError> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64, ColmapError> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }

    /// Read a count and reject it before any allocation or long iteration when
    /// the remaining input cannot contain even the minimum-sized records.
    /// The bound is derived from this input's remaining bytes rather than a
    /// fixed model-size limit, so valid large models remain supported.
    fn read_count_with_min_bytes(
        &mut self,
        field: &'static str,
        min_record_bytes: usize,
    ) -> Result<usize, ColmapError> {
        let raw = self.read_u64()?;
        let count = usize::try_from(raw).map_err(|_| ColmapError::InvalidBinary {
            file: self.file,
            message: format!("{field}={raw} does not fit in usize"),
        })?;
        let required_bytes =
            count
                .checked_mul(min_record_bytes)
                .ok_or_else(|| ColmapError::InvalidBinary {
                    file: self.file,
                    message: format!("{field}={count} overflows minimum record-byte calculation"),
                })?;
        let remaining_bytes = self.contents.len().saturating_sub(self.offset);
        if required_bytes > remaining_bytes {
            return Err(ColmapError::InvalidBinary {
                file: self.file,
                message: format!("{field}={count} requires at least {required_bytes} remaining byte(s), but only {remaining_bytes} remain"),
            });
        }
        Ok(count)
    }

    fn try_reserve<T>(
        &self,
        values: &mut Vec<T>,
        additional: usize,
        field: &'static str,
    ) -> Result<(), ColmapError> {
        values
            .try_reserve(additional)
            .map_err(|error| ColmapError::InvalidBinary {
                file: self.file,
                message: format!("{field}={additional} allocation failed: {error:?}"),
            })
    }

    fn read_null_terminated_string(&mut self) -> Result<String, ColmapError> {
        let start = self.offset;
        while self.offset < self.contents.len() && self.contents[self.offset] != 0 {
            self.offset += 1;
        }
        if self.offset >= self.contents.len() {
            return Err(ColmapError::InvalidBinary {
                file: self.file,
                message: "unterminated string".to_owned(),
            });
        }
        let name = std::str::from_utf8(&self.contents[start..self.offset])
            .map_err(|error| ColmapError::InvalidBinary {
                file: self.file,
                message: format!("invalid UTF-8 string: {error}"),
            })?
            .to_owned();
        self.offset += 1;
        Ok(name)
    }

    fn skip(&mut self, byte_count: usize) -> Result<(), ColmapError> {
        self.read_bytes(byte_count).map(|_| ())
    }

    fn finish(&self) -> Result<(), ColmapError> {
        if self.offset == self.contents.len() {
            Ok(())
        } else {
            Err(ColmapError::InvalidBinary {
                file: self.file,
                message: format!(
                    "{} trailing byte(s) after parsing",
                    self.contents.len() - self.offset
                ),
            })
        }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], ColmapError> {
        let bytes = self.read_bytes(N)?;
        let mut array = [0; N];
        array.copy_from_slice(bytes);
        Ok(array)
    }

    fn read_bytes(&mut self, byte_count: usize) -> Result<&'a [u8], ColmapError> {
        let end =
            self.offset
                .checked_add(byte_count)
                .ok_or_else(|| ColmapError::InvalidBinary {
                    file: self.file,
                    message: "reader offset overflow".to_owned(),
                })?;
        if end > self.contents.len() {
            return Err(ColmapError::InvalidBinary {
                file: self.file,
                message: format!("unexpected end of file at byte {}", self.offset),
            });
        }
        let bytes = &self.contents[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }
}
