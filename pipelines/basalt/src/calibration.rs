//! Reader for Basalt EuRoC calibration JSON files.
//!
//! Basalt's EuRoC files wrap the calibration object in a `value0` member and
//! use `T_imu_cam` for a camera-to-IMU transform. This module keeps that
//! direction explicit: `t_imu_cam` maps a point in camera coordinates into
//! IMU coordinates (`p_i = T_imu_cam * p_c`). The inverse is exposed only via
//! [`BasaltCalibration::imu_to_camera`].

use std::{fs, path::Path};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use serde::Deserialize;
use thiserror::Error;
use visloc_core::geometry::SE3;

use crate::{
    camera::{CameraModelError, DoubleSphereCamera},
    time::{TimeInterval, TimeIntervalError},
    types::CameraId,
};

/// A validated Basalt EuRoC calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct BasaltCalibration {
    /// `T_imu_cam`, mapping camera-frame points into the IMU frame.
    pub t_imu_cam: Vec<SE3>,
    /// Double Sphere intrinsics, in the same order as `t_imu_cam`.
    pub cameras: Vec<DoubleSphereCamera>,
    /// Image resolutions `(width, height)`, in the same order as `cameras`.
    pub resolutions: Vec<(u32, u32)>,
    pub calib_accel_bias: Vec<f64>,
    pub calib_gyro_bias: Vec<f64>,
    pub imu_update_rate_hz: f64,
    pub accel_noise_std: Vector3<f64>,
    pub gyro_noise_std: Vector3<f64>,
    pub accel_bias_std: Vector3<f64>,
    pub gyro_bias_std: Vector3<f64>,
    /// `T_mocap_world` as supplied by Basalt's calibration file.
    pub t_mocap_world: SE3,
    /// `T_imu_marker` as supplied by Basalt's calibration file.
    pub t_imu_marker: SE3,
    pub mocap_time_offset_ns: i64,
    pub mocap_to_imu_offset_ns: i64,
    /// Added to camera timestamps to obtain the corrected IMU clock.
    pub cam_time_offset_ns: i64,
}

impl BasaltCalibration {
    /// Parses either the standard `{ "value0": { ... } }` form or the inner
    /// calibration object itself.
    pub fn from_json_str(json: &str) -> Result<Self, CalibrationError> {
        let document: RawDocument = serde_json::from_str(json)?;
        Self::from_raw(document.into_calibration()?)
    }

    /// Reads a Basalt EuRoC calibration file without depending on `visloc-io`.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, CalibrationError> {
        let json = fs::read_to_string(path)?;
        Self::from_json_str(&json)
    }

    /// Returns the camera model for a numeric camera stream identifier.
    pub fn camera(&self, camera_id: CameraId) -> Option<&DoubleSphereCamera> {
        self.cameras.get(camera_id as usize)
    }

    /// Returns `T_imu_cam`, mapping camera-frame points into IMU coordinates.
    pub fn camera_to_imu(&self, camera_id: CameraId) -> Option<&SE3> {
        self.t_imu_cam.get(camera_id as usize)
    }

    /// Returns the inverse transform `T_cam_imu`, mapping IMU-frame points
    /// into camera coordinates.
    pub fn imu_to_camera(&self, camera_id: CameraId) -> Option<SE3> {
        self.camera_to_imu(camera_id).map(SE3::inverse)
    }

    /// Returns a copy that keeps only the first `count` cameras (intrinsics,
    /// extrinsics and resolutions together); IMU and mocap fields are
    /// unchanged.
    ///
    /// `retain_cameras(1)` turns a stereo calibration into the
    /// monocular-inertial (cam0 + IMU) rig used by the opt-in `--mono`
    /// replay.  `count` must be in `1..=self.cameras.len()`.
    pub fn retain_cameras(&self, count: usize) -> Result<Self, CalibrationError> {
        if count == 0 || count > self.cameras.len() {
            return Err(CalibrationError::Invalid {
                reason: format!(
                    "cannot retain {count} of {} calibrated cameras",
                    self.cameras.len()
                ),
            });
        }
        let mut retained = self.clone();
        retained.t_imu_cam.truncate(count);
        retained.cameras.truncate(count);
        retained.resolutions.truncate(count);
        Ok(retained)
    }

    /// Converts a camera timestamp interval into the corrected IMU clock.
    ///
    /// Basalt's `cam_time_offset_ns` convention is additive:
    /// `t_corrected = t_camera + cam_time_offset_ns`.
    pub fn camera_interval_to_imu(
        &self,
        camera_interval: TimeInterval,
    ) -> Result<TimeInterval, TimeIntervalError> {
        camera_interval.checked_shift(self.cam_time_offset_ns)
    }

    fn from_raw(raw: RawCalibration) -> Result<Self, CalibrationError> {
        if raw.t_imu_cam.is_empty() {
            return Err(CalibrationError::Invalid {
                reason: "T_imu_cam must contain at least one camera".to_owned(),
            });
        }
        if raw.t_imu_cam.len() != raw.intrinsics.len()
            || raw.t_imu_cam.len() != raw.resolution.len()
        {
            return Err(CalibrationError::Invalid {
                reason: format!(
                    "camera metadata lengths differ: transforms={}, intrinsics={}, resolutions={}",
                    raw.t_imu_cam.len(),
                    raw.intrinsics.len(),
                    raw.resolution.len()
                ),
            });
        }
        if !raw.imu_update_rate.is_finite() || raw.imu_update_rate <= 0.0 {
            return Err(CalibrationError::Invalid {
                reason: format!(
                    "imu_update_rate must be positive, got {}",
                    raw.imu_update_rate
                ),
            });
        }

        let mut t_imu_cam = Vec::with_capacity(raw.t_imu_cam.len());
        for (index, transform) in raw.t_imu_cam.into_iter().enumerate() {
            t_imu_cam.push(parse_transform(transform, &format!("T_imu_cam[{index}]"))?);
        }

        let mut cameras = Vec::with_capacity(raw.intrinsics.len());
        let mut resolutions = Vec::with_capacity(raw.resolution.len());
        for (index, (intrinsic, resolution)) in raw
            .intrinsics
            .into_iter()
            .zip(raw.resolution.into_iter())
            .enumerate()
        {
            if !intrinsic.camera_type.eq_ignore_ascii_case("ds") {
                return Err(CalibrationError::UnsupportedCameraType {
                    camera_id: index,
                    camera_type: intrinsic.camera_type,
                });
            }
            let [width, height] = resolution;
            let camera = DoubleSphereCamera::new(
                intrinsic.intrinsics.fx,
                intrinsic.intrinsics.fy,
                intrinsic.intrinsics.cx,
                intrinsic.intrinsics.cy,
                intrinsic.intrinsics.xi,
                intrinsic.intrinsics.alpha,
                width,
                height,
            )
            .map_err(|source| CalibrationError::InvalidCamera {
                camera_id: index,
                source,
            })?;
            cameras.push(camera);
            resolutions.push((width, height));
        }

        let calib_accel_bias = finite_vector(raw.calib_accel_bias, "calib_accel_bias")?;
        let calib_gyro_bias = finite_vector(raw.calib_gyro_bias, "calib_gyro_bias")?;
        let accel_noise_std = parse_vector3(raw.accel_noise_std, "accel_noise_std")?;
        let gyro_noise_std = parse_vector3(raw.gyro_noise_std, "gyro_noise_std")?;
        let accel_bias_std = parse_vector3(raw.accel_bias_std, "accel_bias_std")?;
        let gyro_bias_std = parse_vector3(raw.gyro_bias_std, "gyro_bias_std")?;

        Ok(Self {
            t_imu_cam,
            cameras,
            resolutions,
            calib_accel_bias,
            calib_gyro_bias,
            imu_update_rate_hz: raw.imu_update_rate,
            accel_noise_std,
            gyro_noise_std,
            accel_bias_std,
            gyro_bias_std,
            t_mocap_world: parse_transform(raw.t_mocap_world, "T_mocap_world")?,
            t_imu_marker: parse_transform(raw.t_imu_marker, "T_imu_marker")?,
            mocap_time_offset_ns: raw.mocap_time_offset_ns,
            mocap_to_imu_offset_ns: raw.mocap_to_imu_offset_ns,
            cam_time_offset_ns: raw.cam_time_offset_ns,
        })
    }
}

fn finite_vector(values: Vec<f64>, field: &str) -> Result<Vec<f64>, CalibrationError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(CalibrationError::Invalid {
            reason: format!("{field} must be a non-empty finite array"),
        });
    }
    Ok(values)
}

fn parse_vector3(values: [f64; 3], field: &str) -> Result<Vector3<f64>, CalibrationError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CalibrationError::Invalid {
            reason: format!("{field} contains a non-finite value"),
        });
    }
    Ok(Vector3::new(values[0], values[1], values[2]))
}

fn parse_transform(raw: RawTransform, field: &str) -> Result<SE3, CalibrationError> {
    let translation = [raw.px, raw.py, raw.pz];
    let quaternion = [raw.qx, raw.qy, raw.qz, raw.qw];
    if translation
        .iter()
        .chain(quaternion.iter())
        .any(|value| !value.is_finite())
    {
        return Err(CalibrationError::Invalid {
            reason: format!("{field} contains a non-finite value"),
        });
    }
    let norm_sq = raw.qx * raw.qx + raw.qy * raw.qy + raw.qz * raw.qz + raw.qw * raw.qw;
    if norm_sq <= f64::EPSILON {
        return Err(CalibrationError::Invalid {
            reason: format!("{field} quaternion has zero norm"),
        });
    }
    Ok(SE3::new(
        UnitQuaternion::new_normalize(Quaternion::new(raw.qw, raw.qx, raw.qy, raw.qz)),
        Vector3::new(raw.px, raw.py, raw.pz),
    ))
}

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("failed to read Basalt calibration: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse Basalt calibration JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Basalt calibration: {reason}")]
    Invalid { reason: String },
    #[error("camera {camera_id} uses unsupported model `{camera_type}`")]
    UnsupportedCameraType {
        camera_id: usize,
        camera_type: String,
    },
    #[error("camera {camera_id} has invalid Double Sphere parameters: {source}")]
    InvalidCamera {
        camera_id: usize,
        source: CameraModelError,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawDocument {
    Wrapped { value0: RawCalibration },
    Direct(RawCalibration),
}

impl RawDocument {
    fn into_calibration(self) -> Result<RawCalibration, CalibrationError> {
        match self {
            Self::Wrapped { value0 } => Ok(value0),
            Self::Direct(calibration) => Ok(calibration),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawCalibration {
    #[serde(rename = "T_imu_cam")]
    t_imu_cam: Vec<RawTransform>,
    intrinsics: Vec<RawIntrinsic>,
    resolution: Vec<[u32; 2]>,
    calib_accel_bias: Vec<f64>,
    calib_gyro_bias: Vec<f64>,
    #[serde(rename = "imu_update_rate")]
    imu_update_rate: f64,
    accel_noise_std: [f64; 3],
    gyro_noise_std: [f64; 3],
    accel_bias_std: [f64; 3],
    gyro_bias_std: [f64; 3],
    #[serde(rename = "T_mocap_world")]
    t_mocap_world: RawTransform,
    #[serde(rename = "T_imu_marker")]
    t_imu_marker: RawTransform,
    mocap_time_offset_ns: i64,
    mocap_to_imu_offset_ns: i64,
    cam_time_offset_ns: i64,
}

#[derive(Debug, Deserialize)]
struct RawIntrinsic {
    camera_type: String,
    intrinsics: RawDoubleSphere,
}

#[derive(Debug, Deserialize)]
struct RawDoubleSphere {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    xi: f64,
    alpha: f64,
}

#[derive(Debug, Deserialize)]
struct RawTransform {
    px: f64,
    py: f64,
    pz: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    qw: f64,
}
