use std::fs;
use std::path::Path;

use thiserror::Error;
use visloc_core::types::{Camera, CameraId};

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid KITTI calibration line {line_number}: {line} ({message})")]
    InvalidKittiLine {
        line_number: usize,
        line: String,
        message: String,
    },
    #[error("KITTI projection {label} was not found")]
    KittiProjectionNotFound { label: String },
    #[error("invalid KITTI projection {label}: {message}")]
    InvalidKittiProjection { label: String, message: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiProjection {
    pub label: String,
    pub values: [f64; 12],
}

impl KittiProjection {
    pub fn fx(&self) -> f64 {
        self.values[0]
    }

    pub fn fy(&self) -> f64 {
        self.values[5]
    }

    pub fn cx(&self) -> f64 {
        self.values[2]
    }

    pub fn cy(&self) -> f64 {
        self.values[6]
    }

    pub fn to_pinhole_camera(
        &self,
        camera_id: CameraId,
        width: u32,
        height: u32,
    ) -> Result<Camera, CalibrationError> {
        let fx = self.fx();
        let fy = self.fy();
        if !fx.is_finite() || fx <= 0.0 {
            return Err(CalibrationError::InvalidKittiProjection {
                label: self.label.clone(),
                message: format!("fx must be positive and finite, got {fx}"),
            });
        }
        if !fy.is_finite() || fy <= 0.0 {
            return Err(CalibrationError::InvalidKittiProjection {
                label: self.label.clone(),
                message: format!("fy must be positive and finite, got {fy}"),
            });
        }

        Ok(Camera::pinhole(
            camera_id,
            width,
            height,
            fx,
            fy,
            self.cx(),
            self.cy(),
        ))
    }
}

pub fn read_kitti_calibration_txt(
    path: impl AsRef<Path>,
) -> Result<Vec<KittiProjection>, CalibrationError> {
    parse_kitti_calibration_txt(&fs::read_to_string(path)?)
}

pub fn read_kitti_pinhole_camera(
    path: impl AsRef<Path>,
    label: &str,
    camera_id: CameraId,
    width: u32,
    height: u32,
) -> Result<Camera, CalibrationError> {
    let projections = read_kitti_calibration_txt(path)?;
    kitti_projection_to_pinhole_camera(&projections, label, camera_id, width, height)
}

pub fn parse_kitti_calibration_txt(text: &str) -> Result<Vec<KittiProjection>, CalibrationError> {
    let mut projections = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((label, rest)) = trimmed.split_once(':') else {
            continue;
        };
        let label = label.trim();
        if !is_kitti_projection_label(label) {
            continue;
        }

        let tokens: Vec<_> = rest.split_whitespace().collect();
        if tokens.len() != 12 {
            return Err(CalibrationError::InvalidKittiLine {
                line_number,
                line: line.to_owned(),
                message: format!("expected 12 projection values, got {}", tokens.len()),
            });
        }

        let mut values = [0.0; 12];
        for (index, token) in tokens.iter().enumerate() {
            values[index] =
                token
                    .parse::<f64>()
                    .map_err(|error| CalibrationError::InvalidKittiLine {
                        line_number,
                        line: line.to_owned(),
                        message: format!("invalid projection value {index}: {error}"),
                    })?;
        }

        projections.push(KittiProjection {
            label: label.to_owned(),
            values,
        });
    }
    Ok(projections)
}

pub fn kitti_projection_to_pinhole_camera(
    projections: &[KittiProjection],
    label: &str,
    camera_id: CameraId,
    width: u32,
    height: u32,
) -> Result<Camera, CalibrationError> {
    projections
        .iter()
        .find(|projection| projection.label == label)
        .ok_or_else(|| CalibrationError::KittiProjectionNotFound {
            label: label.to_owned(),
        })?
        .to_pinhole_camera(camera_id, width, height)
}

fn is_kitti_projection_label(label: &str) -> bool {
    let bytes = label.as_bytes();
    bytes.len() == 2 && bytes[0] == b'P' && bytes[1].is_ascii_digit()
}
