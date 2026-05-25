//! Loader for the EuRoC MAV stereo-inertial dataset layout.
//!
//! The dataset ships a per-MAV root directory (e.g. `MH_01_easy/`) whose
//! `mav0/` subdirectory groups the sensors:
//!
//! - `mav0/cam0/data/<timestamp_ns>.png` + `mav0/cam0/data.csv`
//!   (`#timestamp [ns], filename`) and `mav0/cam0/sensor.yaml` (intrinsics +
//!   body-to-camera extrinsics)
//! - `mav0/cam1/...` mirrors `cam0` for the right rectified camera
//! - `mav0/imu0/data.csv` carries the body-frame gyro and accel readings as
//!   `#timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z`
//!   plus `mav0/imu0/sensor.yaml` with the noise densities + random walks
//! - `mav0/state_groundtruth_estimate0/data.csv` provides position +
//!   orientation (Hamilton quaternion `qw, qx, qy, qz`) and, for the EuRoC
//!   Vicon sequences, also linear velocity and per-axis IMU biases.
//!
//! This module exposes typed readers for each piece plus a composite
//! [`read_euroc_dataset_dir`] that pulls everything together. The CSV/YAML
//! formats are small and stable, so the implementation parses them by hand
//! rather than pulling in a new dependency. Image pixels are NOT decoded
//! here — pair the returned [`EurocImageEntry`] timestamps and filenames with
//! `visloc_io::images::read_common_image` when running the actual harness.

use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix4, UnitQuaternion, Vector3};
use thiserror::Error;

/// Error type for every EuRoC reader.
#[derive(Debug, Error)]
pub enum EurocError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("missing EuRoC file: {0}")]
    MissingFile(PathBuf),
    #[error("invalid EuRoC CSV {path}: line {line_number}: {message}")]
    InvalidCsv {
        path: PathBuf,
        line_number: usize,
        message: String,
    },
    #[error("invalid EuRoC sensor.yaml {path}: {message}")]
    InvalidYaml { path: PathBuf, message: String },
    #[error("EuRoC sensor.yaml {path} is missing required field '{field}'")]
    MissingYamlField { path: PathBuf, field: String },
}

fn read_to_string(path: &Path) -> Result<String, EurocError> {
    fs::read_to_string(path).map_err(|source| EurocError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// One image-manifest row from `mav0/camN/data.csv`.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocImageEntry {
    /// Image acquisition time in nanoseconds since the UNIX epoch.
    pub timestamp_nanoseconds: i128,
    /// Filename inside the matching `data/` directory (e.g. `1403636579763555584.png`).
    pub filename: String,
}

/// Read a `mav0/camN/data.csv` image manifest.
///
/// Lines starting with `#` are treated as headers/comments. Each non-comment
/// row must have exactly two comma-separated fields: an integer nanosecond
/// timestamp and a filename.
pub fn read_euroc_image_manifest(path: &Path) -> Result<Vec<EurocImageEntry>, EurocError> {
    if !path.exists() {
        return Err(EurocError::MissingFile(path.to_path_buf()));
    }
    let text = read_to_string(path)?;
    let mut out = Vec::new();
    for (line_number, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != 2 {
            return Err(EurocError::InvalidCsv {
                path: path.to_path_buf(),
                line_number: line_number + 1,
                message: format!(
                    "expected 2 columns (timestamp, filename), got {}",
                    parts.len()
                ),
            });
        }
        let timestamp_nanoseconds =
            parts[0]
                .trim()
                .parse::<i128>()
                .map_err(|err| EurocError::InvalidCsv {
                    path: path.to_path_buf(),
                    line_number: line_number + 1,
                    message: format!("cannot parse timestamp '{}': {err}", parts[0]),
                })?;
        out.push(EurocImageEntry {
            timestamp_nanoseconds,
            filename: parts[1].trim().to_string(),
        });
    }
    Ok(out)
}

/// One body-frame gyro + accel sample from `mav0/imu0/data.csv`.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocImuSample {
    pub timestamp_nanoseconds: i128,
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

/// Read `mav0/imu0/data.csv` and return every sample in file order.
///
/// The expected header is
/// `#timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z`.
/// The reader skips comment lines and accepts arbitrary whitespace around
/// commas.
pub fn read_euroc_imu_csv(path: &Path) -> Result<Vec<EurocImuSample>, EurocError> {
    if !path.exists() {
        return Err(EurocError::MissingFile(path.to_path_buf()));
    }
    let text = read_to_string(path)?;
    let mut out = Vec::new();
    for (line_number, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let nums = parse_numeric_csv_row(trimmed).map_err(|message| EurocError::InvalidCsv {
            path: path.to_path_buf(),
            line_number: line_number + 1,
            message,
        })?;
        if nums.len() != 7 {
            return Err(EurocError::InvalidCsv {
                path: path.to_path_buf(),
                line_number: line_number + 1,
                message: format!(
                    "expected 7 columns (ts, gx, gy, gz, ax, ay, az), got {}",
                    nums.len()
                ),
            });
        }
        let timestamp_nanoseconds = nums[0] as i128;
        out.push(EurocImuSample {
            timestamp_nanoseconds,
            gyro: Vector3::new(nums[1], nums[2], nums[3]),
            accel: Vector3::new(nums[4], nums[5], nums[6]),
        });
    }
    Ok(out)
}

/// One ground-truth state row from
/// `mav0/state_groundtruth_estimate0/data.csv` (Vicon-derived for EuRoC).
///
/// The Hamilton quaternion is stored on disk as `(qw, qx, qy, qz)` and is
/// converted here into `nalgebra::UnitQuaternion`. Linear velocity and bias
/// fields are optional — the dataset's two "MH" lab sequences carry them,
/// the "V1"/"V2" recordings drop the bias columns, and some derived files
/// keep only the 7-column pose-only form.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocGroundTruthSample {
    pub timestamp_nanoseconds: i128,
    pub position_world: Vector3<f64>,
    pub orientation_world: UnitQuaternion<f64>,
    pub velocity_world: Option<Vector3<f64>>,
    pub bias_gyro: Option<Vector3<f64>>,
    pub bias_acc: Option<Vector3<f64>>,
}

/// Read `mav0/state_groundtruth_estimate0/data.csv`.
///
/// Accepts the canonical 17-column EuRoC layout (timestamp + 3 position + 4
/// quaternion + 3 velocity + 3 gyro bias + 3 accel bias), the 13-column
/// pose+velocity form, and the 8-column pose-only form.
pub fn read_euroc_ground_truth_csv(path: &Path) -> Result<Vec<EurocGroundTruthSample>, EurocError> {
    if !path.exists() {
        return Err(EurocError::MissingFile(path.to_path_buf()));
    }
    let text = read_to_string(path)?;
    let mut out = Vec::new();
    for (line_number, raw) in text.lines().enumerate() {
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let nums = parse_numeric_csv_row(trimmed).map_err(|message| EurocError::InvalidCsv {
            path: path.to_path_buf(),
            line_number: line_number + 1,
            message,
        })?;
        if !matches!(nums.len(), 8 | 13 | 17) {
            return Err(EurocError::InvalidCsv {
                path: path.to_path_buf(),
                line_number: line_number + 1,
                message: format!(
                    "expected 8, 13, or 17 columns (ts + pose + optional velocity/bias), got {}",
                    nums.len()
                ),
            });
        }
        let timestamp_nanoseconds = nums[0] as i128;
        let position_world = Vector3::new(nums[1], nums[2], nums[3]);
        let qw = nums[4];
        let qx = nums[5];
        let qy = nums[6];
        let qz = nums[7];
        let orientation_world =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let velocity_world = if nums.len() >= 13 {
            Some(Vector3::new(nums[8], nums[9], nums[10]))
        } else {
            None
        };
        let (bias_gyro, bias_acc) = if nums.len() == 17 {
            (
                Some(Vector3::new(nums[11], nums[12], nums[13])),
                Some(Vector3::new(nums[14], nums[15], nums[16])),
            )
        } else {
            (None, None)
        };
        out.push(EurocGroundTruthSample {
            timestamp_nanoseconds,
            position_world,
            orientation_world,
            velocity_world,
            bias_gyro,
            bias_acc,
        });
    }
    Ok(out)
}

fn parse_numeric_csv_row(line: &str) -> Result<Vec<f64>, String> {
    line.split(',')
        .enumerate()
        .map(|(index, token)| {
            token
                .trim()
                .parse::<f64>()
                .map_err(|err| format!("column {}: cannot parse '{}': {err}", index, token))
        })
        .collect()
}

/// Camera calibration parsed from `mav0/camN/sensor.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocCameraCalibration {
    /// `T_BS` body-to-sensor transform (`4×4`, row-major in the YAML file).
    pub t_body_sensor: Matrix4<f64>,
    /// Nominal frame rate in Hz.
    pub rate_hz: f64,
    /// Image resolution `(width, height)` in pixels.
    pub resolution: (u32, u32),
    /// Camera model identifier (`pinhole` for every EuRoC release).
    pub camera_model: String,
    /// Pinhole intrinsics `(fu, fv, cu, cv)`.
    pub intrinsics: [f64; 4],
    /// Distortion model identifier (`radial-tangential` for EuRoC).
    pub distortion_model: String,
    /// Distortion coefficients in the order declared by `distortion_model`
    /// (4-vector `(k1, k2, p1, p2)` for `radial-tangential`).
    pub distortion_coefficients: Vec<f64>,
}

/// IMU calibration parsed from `mav0/imu0/sensor.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct EurocImuCalibration {
    pub t_body_sensor: Matrix4<f64>,
    pub rate_hz: f64,
    pub gyroscope_noise_density: f64,
    pub gyroscope_random_walk: f64,
    pub accelerometer_noise_density: f64,
    pub accelerometer_random_walk: f64,
}

/// Parse a `mav0/camN/sensor.yaml` file.
pub fn read_euroc_camera_sensor_yaml(path: &Path) -> Result<EurocCameraCalibration, EurocError> {
    if !path.exists() {
        return Err(EurocError::MissingFile(path.to_path_buf()));
    }
    let text = read_to_string(path)?;
    let t_body_sensor = read_t_bs(path, &text)?;
    let rate_hz = read_scalar_f64(path, &text, "rate_hz")?;
    let resolution_vec = read_inline_array_f64(path, &text, "resolution")?;
    if resolution_vec.len() != 2 {
        return Err(EurocError::InvalidYaml {
            path: path.to_path_buf(),
            message: format!(
                "resolution must have 2 entries, got {}",
                resolution_vec.len()
            ),
        });
    }
    let resolution = (resolution_vec[0] as u32, resolution_vec[1] as u32);
    let camera_model = read_scalar_string(path, &text, "camera_model")?;
    let intrinsics_vec = read_inline_array_f64(path, &text, "intrinsics")?;
    if intrinsics_vec.len() != 4 {
        return Err(EurocError::InvalidYaml {
            path: path.to_path_buf(),
            message: format!(
                "intrinsics must have 4 entries (fu, fv, cu, cv), got {}",
                intrinsics_vec.len()
            ),
        });
    }
    let intrinsics = [
        intrinsics_vec[0],
        intrinsics_vec[1],
        intrinsics_vec[2],
        intrinsics_vec[3],
    ];
    let distortion_model = read_scalar_string(path, &text, "distortion_model")?;
    let distortion_coefficients = read_inline_array_f64(path, &text, "distortion_coefficients")?;
    Ok(EurocCameraCalibration {
        t_body_sensor,
        rate_hz,
        resolution,
        camera_model,
        intrinsics,
        distortion_model,
        distortion_coefficients,
    })
}

/// Parse a `mav0/imu0/sensor.yaml` file.
pub fn read_euroc_imu_sensor_yaml(path: &Path) -> Result<EurocImuCalibration, EurocError> {
    if !path.exists() {
        return Err(EurocError::MissingFile(path.to_path_buf()));
    }
    let text = read_to_string(path)?;
    let t_body_sensor = read_t_bs(path, &text)?;
    let rate_hz = read_scalar_f64(path, &text, "rate_hz")?;
    let gyroscope_noise_density = read_scalar_f64(path, &text, "gyroscope_noise_density")?;
    let gyroscope_random_walk = read_scalar_f64(path, &text, "gyroscope_random_walk")?;
    let accelerometer_noise_density = read_scalar_f64(path, &text, "accelerometer_noise_density")?;
    let accelerometer_random_walk = read_scalar_f64(path, &text, "accelerometer_random_walk")?;
    Ok(EurocImuCalibration {
        t_body_sensor,
        rate_hz,
        gyroscope_noise_density,
        gyroscope_random_walk,
        accelerometer_noise_density,
        accelerometer_random_walk,
    })
}

/// Aggregate handle returned by [`read_euroc_dataset_dir`].
#[derive(Debug, Clone, PartialEq)]
pub struct EurocDataset {
    /// Root directory passed in (kept for relative path resolution).
    pub root: PathBuf,
    /// Resolved `mav0/cam0/data/` directory.
    pub cam0_image_dir: PathBuf,
    /// Resolved `mav0/cam1/data/` directory.
    pub cam1_image_dir: PathBuf,
    pub cam0_images: Vec<EurocImageEntry>,
    pub cam1_images: Vec<EurocImageEntry>,
    pub cam0_calibration: EurocCameraCalibration,
    pub cam1_calibration: EurocCameraCalibration,
    pub imu_samples: Vec<EurocImuSample>,
    pub imu_calibration: EurocImuCalibration,
    pub ground_truth: Vec<EurocGroundTruthSample>,
}

/// Read a complete EuRoC MAV recording rooted at `dir` (the directory that
/// contains the `mav0/` subdirectory). Ground truth is optional — if
/// `mav0/state_groundtruth_estimate0/data.csv` is missing the returned
/// dataset has an empty `ground_truth` vector.
pub fn read_euroc_dataset_dir(dir: &Path) -> Result<EurocDataset, EurocError> {
    let mav0 = dir.join("mav0");
    let cam0_dir = mav0.join("cam0");
    let cam1_dir = mav0.join("cam1");
    let imu0_dir = mav0.join("imu0");
    let gt_dir = mav0.join("state_groundtruth_estimate0");

    let cam0_images = read_euroc_image_manifest(&cam0_dir.join("data.csv"))?;
    let cam1_images = read_euroc_image_manifest(&cam1_dir.join("data.csv"))?;
    let cam0_calibration = read_euroc_camera_sensor_yaml(&cam0_dir.join("sensor.yaml"))?;
    let cam1_calibration = read_euroc_camera_sensor_yaml(&cam1_dir.join("sensor.yaml"))?;
    let imu_samples = read_euroc_imu_csv(&imu0_dir.join("data.csv"))?;
    let imu_calibration = read_euroc_imu_sensor_yaml(&imu0_dir.join("sensor.yaml"))?;

    let gt_csv = gt_dir.join("data.csv");
    let ground_truth = if gt_csv.exists() {
        read_euroc_ground_truth_csv(&gt_csv)?
    } else {
        Vec::new()
    };

    Ok(EurocDataset {
        root: dir.to_path_buf(),
        cam0_image_dir: cam0_dir.join("data"),
        cam1_image_dir: cam1_dir.join("data"),
        cam0_images,
        cam1_images,
        cam0_calibration,
        cam1_calibration,
        imu_samples,
        imu_calibration,
        ground_truth,
    })
}

// -------------------------------------------------------------------
// Minimal YAML helpers specialised for EuRoC sensor.yaml files.
// The format is small and stable: top-level `key: value` plus an indented
// `T_BS:` block with a bracketed `data:` array. We avoid a general YAML
// parser and just walk the lines.
// -------------------------------------------------------------------

fn read_scalar_f64(path: &Path, text: &str, key: &str) -> Result<f64, EurocError> {
    let raw = read_scalar_raw(text, key).ok_or_else(|| EurocError::MissingYamlField {
        path: path.to_path_buf(),
        field: key.to_string(),
    })?;
    raw.parse::<f64>().map_err(|err| EurocError::InvalidYaml {
        path: path.to_path_buf(),
        message: format!("cannot parse '{key}: {raw}' as f64: {err}"),
    })
}

fn read_scalar_string(path: &Path, text: &str, key: &str) -> Result<String, EurocError> {
    let raw = read_scalar_raw(text, key).ok_or_else(|| EurocError::MissingYamlField {
        path: path.to_path_buf(),
        field: key.to_string(),
    })?;
    Ok(strip_quotes(&raw).to_string())
}

fn read_scalar_raw(text: &str, key: &str) -> Option<String> {
    let needle = format!("{key}:");
    for raw_line in text.lines() {
        let trimmed = strip_comment(raw_line).trim();
        let Some(remainder) = trimmed.strip_prefix(&needle) else {
            continue;
        };
        let value = remainder.trim();
        if value.is_empty() || value.starts_with('[') {
            // Either a sub-block opener (handled separately) or an inline
            // array; the scalar helper only returns plain key/value rows.
            return None;
        }
        return Some(value.to_string());
    }
    None
}

fn read_inline_array_f64(path: &Path, text: &str, key: &str) -> Result<Vec<f64>, EurocError> {
    let raw = read_array_raw(text, key, 0).ok_or_else(|| EurocError::MissingYamlField {
        path: path.to_path_buf(),
        field: key.to_string(),
    })?;
    parse_bracketed_floats(&raw).map_err(|message| EurocError::InvalidYaml {
        path: path.to_path_buf(),
        message: format!("'{key}': {message}"),
    })
}

fn read_t_bs(path: &Path, text: &str) -> Result<Matrix4<f64>, EurocError> {
    // T_BS is an indented block:
    //   T_BS:
    //     cols: 4
    //     rows: 4
    //     data: [a, b, c, d,
    //            e, f, g, h, ...]
    let block =
        extract_indented_block(text, "T_BS").ok_or_else(|| EurocError::MissingYamlField {
            path: path.to_path_buf(),
            field: "T_BS".to_string(),
        })?;
    let raw = read_array_raw(&block, "data", 0).ok_or_else(|| EurocError::MissingYamlField {
        path: path.to_path_buf(),
        field: "T_BS.data".to_string(),
    })?;
    let values = parse_bracketed_floats(&raw).map_err(|message| EurocError::InvalidYaml {
        path: path.to_path_buf(),
        message: format!("'T_BS.data': {message}"),
    })?;
    if values.len() != 16 {
        return Err(EurocError::InvalidYaml {
            path: path.to_path_buf(),
            message: format!(
                "'T_BS.data' must hold 16 entries (4x4 row-major), got {}",
                values.len()
            ),
        });
    }
    let mut t = Matrix4::<f64>::identity();
    for row in 0..4 {
        for col in 0..4 {
            t[(row, col)] = values[row * 4 + col];
        }
    }
    Ok(t)
}

/// Slice the lines that belong to `key:` as an indented sub-block (every
/// following line whose indent is strictly greater than the block opener).
/// Returns the joined inner text or `None` if the key has no sub-block.
fn extract_indented_block(text: &str, key: &str) -> Option<String> {
    let needle = format!("{key}:");
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let raw = lines[i];
        let stripped = strip_comment(raw);
        let indent = leading_spaces(stripped);
        if stripped.trim() == needle {
            // Collect all following lines with indent > opener's indent.
            let mut block = String::new();
            let mut j = i + 1;
            while j < lines.len() {
                let nxt = strip_comment(lines[j]);
                if nxt.trim().is_empty() {
                    block.push_str(lines[j]);
                    block.push('\n');
                    j += 1;
                    continue;
                }
                if leading_spaces(nxt) <= indent {
                    break;
                }
                block.push_str(lines[j]);
                block.push('\n');
                j += 1;
            }
            return Some(block);
        }
        i += 1;
    }
    None
}

/// Read a (possibly multi-line) inline array under `key:`. The opening `[`
/// must appear on the same line as `key:`; subsequent physical lines are
/// concatenated until the matching `]` is found.
fn read_array_raw(text: &str, key: &str, _min_indent: usize) -> Option<String> {
    let needle = format!("{key}:");
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let stripped = strip_comment(lines[i]);
        let trimmed = stripped.trim();
        let Some(remainder) = trimmed.strip_prefix(&needle) else {
            i += 1;
            continue;
        };
        let value = remainder.trim();
        if !value.starts_with('[') {
            i += 1;
            continue;
        }
        let mut buf = String::from(value);
        if buf.contains(']') {
            return Some(buf);
        }
        let mut j = i + 1;
        while j < lines.len() {
            let nxt = strip_comment(lines[j]);
            buf.push(' ');
            buf.push_str(nxt.trim());
            if nxt.contains(']') {
                return Some(buf);
            }
            j += 1;
        }
        return Some(buf);
    }
    None
}

fn parse_bracketed_floats(text: &str) -> Result<Vec<f64>, String> {
    let start = text.find('[').ok_or_else(|| "missing '['".to_string())?;
    let end = text.rfind(']').ok_or_else(|| "missing ']'".to_string())?;
    if end <= start {
        return Err("']' precedes '['".to_string());
    }
    let inner = &text[start + 1..end];
    let mut out = Vec::new();
    for token in inner.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(
            trimmed
                .parse::<f64>()
                .map_err(|err| format!("cannot parse '{trimmed}': {err}"))?,
        );
    }
    Ok(out)
}

fn strip_comment(line: &str) -> &str {
    // YAML comments start with '#' but only when not inside quotes. EuRoC
    // sensor.yaml files only carry simple '# ...' trailers, so a plain split
    // on the first '#' is sufficient here.
    line.find('#').map(|idx| &line[..idx]).unwrap_or(line)
}

fn leading_spaces(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count()
}

fn strip_quotes(value: &str) -> &str {
    let trimmed = value.trim();
    if (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
        || (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2)
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(dir: &Path, rel: &str, contents: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
    }

    fn cam_sensor_yaml() -> &'static str {
        // Trimmed-down replica of the canonical EuRoC cam0 sensor.yaml.
        "%YAML:1.0\n\
         sensor_type: camera\n\
         comment: VI-Sensor cam0\n\
         T_BS:\n  \
           cols: 4\n  \
           rows: 4\n  \
           data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,\n         \
                  0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,\n         \
                  -0.0257744366974, 0.00375618835797, 0.999660727927, 0.00981073058949,\n         \
                  0.0, 0.0, 0.0, 1.0]\n\
         rate_hz: 20.0\n\
         resolution: [752, 480]\n\
         camera_model: pinhole\n\
         intrinsics: [458.654, 457.296, 367.215, 248.375]\n\
         distortion_model: radial-tangential\n\
         distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 0.0000176187114]\n"
    }

    fn imu_sensor_yaml() -> &'static str {
        "%YAML:1.0\n\
         sensor_type: imu\n\
         comment: VI-Sensor IMU (ADIS16448)\n\
         T_BS:\n  \
           cols: 4\n  \
           rows: 4\n  \
           data: [1.0, 0.0, 0.0, 0.0,\n         \
                  0.0, 1.0, 0.0, 0.0,\n         \
                  0.0, 0.0, 1.0, 0.0,\n         \
                  0.0, 0.0, 0.0, 1.0]\n\
         rate_hz: 200\n\
         gyroscope_noise_density: 1.6968e-04\n\
         gyroscope_random_walk: 1.9393e-05\n\
         accelerometer_noise_density: 2.0e-3\n\
         accelerometer_random_walk: 3.0e-3\n"
    }

    #[test]
    fn parses_camera_sensor_yaml() {
        let dir = tempdir();
        write_file(&dir, "sensor.yaml", cam_sensor_yaml());
        let calib = read_euroc_camera_sensor_yaml(&dir.join("sensor.yaml")).unwrap();
        assert_eq!(calib.resolution, (752, 480));
        assert_eq!(calib.camera_model, "pinhole");
        assert_eq!(calib.distortion_model, "radial-tangential");
        assert!((calib.intrinsics[0] - 458.654).abs() < 1.0e-9);
        assert!((calib.intrinsics[3] - 248.375).abs() < 1.0e-9);
        assert_eq!(calib.distortion_coefficients.len(), 4);
        assert!((calib.distortion_coefficients[0] + 0.28340811).abs() < 1.0e-9);
        assert!((calib.t_body_sensor[(3, 3)] - 1.0).abs() < 1.0e-12);
        assert!((calib.t_body_sensor[(0, 1)] + 0.999880929698).abs() < 1.0e-9);
        assert!((calib.rate_hz - 20.0).abs() < 1.0e-12);
    }

    #[test]
    fn parses_imu_sensor_yaml() {
        let dir = tempdir();
        write_file(&dir, "sensor.yaml", imu_sensor_yaml());
        let calib = read_euroc_imu_sensor_yaml(&dir.join("sensor.yaml")).unwrap();
        assert!((calib.rate_hz - 200.0).abs() < 1.0e-12);
        assert!((calib.gyroscope_noise_density - 1.6968e-04).abs() < 1.0e-12);
        assert!((calib.accelerometer_random_walk - 3.0e-3).abs() < 1.0e-12);
        assert!((calib.t_body_sensor[(0, 0)] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn parses_imu_csv() {
        let dir = tempdir();
        let csv = "#timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z\n\
                   1403636579758555392, -0.09913, 0.14730, 0.05217, 9.04, -0.71, 1.21\n\
                   1403636579763555584, -0.09800, 0.14920, 0.05030, 9.05, -0.70, 1.22\n";
        write_file(&dir, "data.csv", csv);
        let samples = read_euroc_imu_csv(&dir.join("data.csv")).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].timestamp_nanoseconds, 1403636579758555392);
        assert!((samples[1].accel.x - 9.05).abs() < 1.0e-9);
        assert!((samples[0].gyro.y - 0.14730).abs() < 1.0e-9);
    }

    #[test]
    fn parses_image_manifest_and_ground_truth_variants() {
        let dir = tempdir();
        let cam_csv = "#timestamp [ns],filename\n\
                       1403636579763555584,1403636579763555584.png\n\
                       1403636579813555456,1403636579813555456.png\n";
        write_file(&dir, "data.csv", cam_csv);
        let manifest = read_euroc_image_manifest(&dir.join("data.csv")).unwrap();
        assert_eq!(manifest.len(), 2);
        assert_eq!(manifest[1].filename, "1403636579813555456.png");

        // 17-column row carries velocity + biases.
        let gt_full = "#ts,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n\
                       100,0.1,0.2,0.3,1.0,0.0,0.0,0.0,0.01,0.02,0.03,0.001,0.002,0.003,0.04,0.05,0.06\n";
        write_file(&dir, "gt_full.csv", gt_full);
        let full = read_euroc_ground_truth_csv(&dir.join("gt_full.csv")).unwrap();
        assert_eq!(full.len(), 1);
        let row = &full[0];
        assert_eq!(row.timestamp_nanoseconds, 100);
        assert!((row.position_world.x - 0.1).abs() < 1.0e-12);
        assert!(row.velocity_world.is_some());
        assert!(row.bias_gyro.is_some());
        assert!((row.bias_acc.unwrap().z - 0.06).abs() < 1.0e-12);

        // 8-column pose-only form drops velocity + biases.
        let gt_pose_only = "#ts,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n\
                            200,1.0,2.0,3.0,0.7071,0.0,0.7071,0.0\n";
        write_file(&dir, "gt_pose.csv", gt_pose_only);
        let pose = read_euroc_ground_truth_csv(&dir.join("gt_pose.csv")).unwrap();
        assert_eq!(pose.len(), 1);
        assert!(pose[0].velocity_world.is_none());
        assert!(pose[0].bias_gyro.is_none());
        // Hamilton quaternion was stored as (qw, qx, qy, qz); nalgebra packs
        // this into its inner `Quaternion::new(w, x, y, z)` constructor.
        assert!(
            (pose[0].orientation_world.w.abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1.0e-4
        );
    }

    #[test]
    fn reads_complete_dataset_dir() {
        let dir = tempdir();
        for cam in ["cam0", "cam1"] {
            let cam_dir = dir.join("mav0").join(cam);
            write_file(&cam_dir, "sensor.yaml", cam_sensor_yaml());
            let csv = "#timestamp,filename\n\
                       100,100.png\n\
                       200,200.png\n";
            write_file(&cam_dir, "data.csv", csv);
        }
        let imu_dir = dir.join("mav0").join("imu0");
        write_file(&imu_dir, "sensor.yaml", imu_sensor_yaml());
        let imu_csv =
            "#ts,gx,gy,gz,ax,ay,az\n100,0.0,0.0,0.0,0.0,0.0,9.81\n150,0.0,0.0,0.0,0.0,0.0,9.81\n";
        write_file(&imu_dir, "data.csv", imu_csv);

        let dataset = read_euroc_dataset_dir(&dir).unwrap();
        assert_eq!(dataset.cam0_images.len(), 2);
        assert_eq!(dataset.cam1_images.len(), 2);
        assert_eq!(dataset.imu_samples.len(), 2);
        assert!(dataset.ground_truth.is_empty());
        assert_eq!(dataset.cam0_calibration.resolution, (752, 480));
        assert_eq!(
            dataset.cam0_image_dir,
            dir.join("mav0").join("cam0").join("data")
        );
    }

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "visloc_io_euroc_{}_{}",
            std::process::id(),
            random_tag()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).ok();
        }
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn random_tag() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}
