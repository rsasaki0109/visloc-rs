use std::path::Path;

use thiserror::Error;
use visloc_core::types::{Camera, CameraId};

use crate::calibration::{read_kitti_pinhole_camera, CalibrationError};
use crate::images::{
    common_image_sequence_summary, read_common_image_sequence_dir,
    read_common_image_sequence_dir_with_timestamp_file, validate_common_image_sequence_dimensions,
    validate_common_image_sequence_timestamps, ImageSequenceError, ImageSequenceSummary,
    ImageSequenceValidationIssue, LoadedImageFrame,
};

#[derive(Debug, Error)]
pub enum KittiDatasetError {
    #[error("KITTI image sequence error: {0}")]
    ImageSequence(#[from] ImageSequenceError),
    #[error("KITTI calibration error: {0}")]
    Calibration(#[from] CalibrationError),
    #[error("KITTI image sequence is empty")]
    EmptyImageSequence,
    #[error("KITTI image dimensions are too large for Camera: {width}x{height}")]
    CameraDimensionTooLarge { width: usize, height: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiImageSequence {
    pub camera: Camera,
    pub frames: Vec<LoadedImageFrame>,
    pub summary: ImageSequenceSummary,
    pub dimension_issues: Vec<ImageSequenceValidationIssue>,
    pub timestamp_issues: Vec<ImageSequenceValidationIssue>,
}

pub fn read_kitti_image_sequence_dir(
    image_dir: impl AsRef<Path>,
    calibration_path: impl AsRef<Path>,
    projection_label: &str,
    camera_id: CameraId,
) -> Result<KittiImageSequence, KittiDatasetError> {
    let frames = read_common_image_sequence_dir(image_dir)?;
    build_kitti_image_sequence(frames, calibration_path, projection_label, camera_id)
}

pub fn read_kitti_image_sequence_dir_with_timestamp_file(
    image_dir: impl AsRef<Path>,
    timestamp_path: impl AsRef<Path>,
    calibration_path: impl AsRef<Path>,
    projection_label: &str,
    camera_id: CameraId,
) -> Result<KittiImageSequence, KittiDatasetError> {
    let frames = read_common_image_sequence_dir_with_timestamp_file(image_dir, timestamp_path)?;
    build_kitti_image_sequence(frames, calibration_path, projection_label, camera_id)
}

fn build_kitti_image_sequence(
    frames: Vec<LoadedImageFrame>,
    calibration_path: impl AsRef<Path>,
    projection_label: &str,
    camera_id: CameraId,
) -> Result<KittiImageSequence, KittiDatasetError> {
    let first = frames
        .first()
        .ok_or(KittiDatasetError::EmptyImageSequence)?;
    let width = u32::try_from(first.image.width()).map_err(|_| {
        KittiDatasetError::CameraDimensionTooLarge {
            width: first.image.width(),
            height: first.image.height(),
        }
    })?;
    let height = u32::try_from(first.image.height()).map_err(|_| {
        KittiDatasetError::CameraDimensionTooLarge {
            width: first.image.width(),
            height: first.image.height(),
        }
    })?;
    let camera =
        read_kitti_pinhole_camera(calibration_path, projection_label, camera_id, width, height)?;
    let summary = common_image_sequence_summary(&frames);
    let dimension_issues = validate_common_image_sequence_dimensions(&frames);
    let timestamp_issues = validate_common_image_sequence_timestamps(&frames);

    Ok(KittiImageSequence {
        camera,
        frames,
        summary,
        dimension_issues,
        timestamp_issues,
    })
}
