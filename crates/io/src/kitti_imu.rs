//! Loader for KITTI raw OXTS / IMU logs.
//!
//! The KITTI raw recordings ship each OXTS sample as a 30-field whitespace
//! separated text file under `<sequence>/oxts/data/<10-digit>.txt`, paired with
//! `<sequence>/oxts/timestamps.txt`. The 30 fields follow the layout described
//! in `dataformat.txt` distributed with the dataset; this loader parses the
//! whole row but exposes the gyroscope / accelerometer triplets (and a couple
//! of useful navigation accuracies) as a typed record so downstream
//! pre-integration code does not need to recount columns.

use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::Vector3;
use thiserror::Error;

/// Number of whitespace-separated fields expected in a KITTI OXTS data row.
pub const KITTI_OXTS_FIELD_COUNT: usize = 30;

#[derive(Debug, Error)]
pub enum KittiOxtsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("OXTS data directory is missing or not a directory: {0}")]
    MissingDataDirectory(PathBuf),
    #[error("OXTS timestamps file is missing: {0}")]
    MissingTimestampsFile(PathBuf),
    #[error(
        "OXTS data file count does not match timestamps: {data_count} data files, {timestamp_count} timestamps"
    )]
    DataTimestampCountMismatch {
        data_count: usize,
        timestamp_count: usize,
    },
    #[error("invalid OXTS data file {path}: {message}")]
    InvalidDataFile { path: PathBuf, message: String },
    #[error("invalid OXTS timestamp line {line_number} in {path}: {line} ({message})")]
    InvalidTimestampLine {
        path: PathBuf,
        line_number: usize,
        line: String,
        message: String,
    },
}

/// One synchronised OXTS sample (parsed from a single
/// `oxts/data/<frame_id>.txt`).
#[derive(Debug, Clone, PartialEq)]
pub struct KittiOxtsSample {
    /// Geodetic latitude / longitude / altitude (degrees, degrees, metres).
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    /// Roll / pitch / yaw of the IMU body w.r.t. the navigation frame (rad).
    pub roll_rad: f64,
    pub pitch_rad: f64,
    pub yaw_rad: f64,
    /// North / east / forward / left / up velocity (m/s).
    pub velocity_north_mps: f64,
    pub velocity_east_mps: f64,
    pub velocity_forward_mps: f64,
    pub velocity_left_mps: f64,
    pub velocity_up_mps: f64,
    /// Linear acceleration in the IMU body frame `(ax, ay, az)` (m/s²),
    /// directly usable for pre-integration once gravity is supplied in the
    /// world frame.
    pub acceleration_body_mps2: Vector3<f64>,
    /// Linear acceleration in the navigation frame `(af, al, au)` (m/s²).
    pub acceleration_nav_mps2: Vector3<f64>,
    /// Angular rate in the IMU body frame `(wx, wy, wz)` (rad/s).
    pub angular_rate_body_rps: Vector3<f64>,
    /// Angular rate in the navigation frame `(wf, wl, wu)` (rad/s).
    pub angular_rate_nav_rps: Vector3<f64>,
    /// Position / velocity accuracy reported by the GNSS/INS integrator.
    pub position_accuracy_m: f64,
    pub velocity_accuracy_mps: f64,
    /// GNSS navigation status flags (kept as integers for fidelity).
    pub navigation_status: i32,
    pub number_of_satellites: i32,
    pub position_mode: i32,
    pub velocity_mode: i32,
    pub orientation_mode: i32,
}

/// OXTS sample with its synchronised timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct KittiOxtsRecord {
    /// Wall-clock nanoseconds parsed from `oxts/timestamps.txt`.
    pub timestamp_nanoseconds: i128,
    pub sample: KittiOxtsSample,
    pub path: PathBuf,
}

/// Read every synchronised OXTS record under `<oxts_root>/data/` together with
/// the matching `<oxts_root>/timestamps.txt` entries.
///
/// Returns the records in lexicographic order of their data file names so that
/// `records[i]` aligns with the i-th camera frame for the same sequence.
pub fn read_kitti_oxts_dir(
    oxts_root: impl AsRef<Path>,
) -> Result<Vec<KittiOxtsRecord>, KittiOxtsError> {
    let oxts_root = oxts_root.as_ref();
    let data_dir = oxts_root.join("data");
    let timestamps_path = oxts_root.join("timestamps.txt");

    if !data_dir.is_dir() {
        return Err(KittiOxtsError::MissingDataDirectory(data_dir));
    }
    if !timestamps_path.is_file() {
        return Err(KittiOxtsError::MissingTimestampsFile(timestamps_path));
    }

    let data_paths = collect_oxts_data_paths(&data_dir)?;
    let timestamps = parse_kitti_oxts_timestamps_txt_with_path(
        &fs::read_to_string(&timestamps_path)?,
        &timestamps_path,
    )?;

    if data_paths.len() != timestamps.len() {
        return Err(KittiOxtsError::DataTimestampCountMismatch {
            data_count: data_paths.len(),
            timestamp_count: timestamps.len(),
        });
    }

    let mut records = Vec::with_capacity(data_paths.len());
    for (path, timestamp) in data_paths.into_iter().zip(timestamps.into_iter()) {
        let text = fs::read_to_string(&path)?;
        let sample =
            parse_kitti_oxts_sample(&text).map_err(|message| KittiOxtsError::InvalidDataFile {
                path: path.clone(),
                message,
            })?;
        records.push(KittiOxtsRecord {
            timestamp_nanoseconds: timestamp,
            sample,
            path,
        });
    }

    Ok(records)
}

/// Parse a single OXTS data line into a [`KittiOxtsSample`].
///
/// Lines starting with `#` and blank lines are skipped; the parser uses the
/// first non-comment line as the data row.
pub fn parse_kitti_oxts_sample(text: &str) -> Result<KittiOxtsSample, String> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .ok_or_else(|| "no data row in OXTS file".to_owned())?;

    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.len() < KITTI_OXTS_FIELD_COUNT {
        return Err(format!(
            "expected {KITTI_OXTS_FIELD_COUNT} fields, got {} ({line})",
            tokens.len(),
        ));
    }

    let f = |index: usize, field: &str| -> Result<f64, String> {
        tokens[index]
            .parse::<f64>()
            .map_err(|error| format!("invalid {field}: {error}"))
    };
    let i = |index: usize, field: &str| -> Result<i32, String> {
        let value = tokens[index]
            .parse::<f64>()
            .map_err(|error| format!("invalid {field}: {error}"))?;
        if !value.is_finite() || value.fract().abs() > f64::EPSILON {
            return Err(format!(
                "invalid {field}: expected integer-valued flag, got {value}"
            ));
        }
        if value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
            return Err(format!(
                "invalid {field}: integer flag out of range, got {value}"
            ));
        }
        Ok(value as i32)
    };

    Ok(KittiOxtsSample {
        lat_deg: f(0, "lat")?,
        lon_deg: f(1, "lon")?,
        alt_m: f(2, "alt")?,
        roll_rad: f(3, "roll")?,
        pitch_rad: f(4, "pitch")?,
        yaw_rad: f(5, "yaw")?,
        velocity_north_mps: f(6, "vn")?,
        velocity_east_mps: f(7, "ve")?,
        velocity_forward_mps: f(8, "vf")?,
        velocity_left_mps: f(9, "vl")?,
        velocity_up_mps: f(10, "vu")?,
        acceleration_body_mps2: Vector3::new(f(11, "ax")?, f(12, "ay")?, f(13, "az")?),
        acceleration_nav_mps2: Vector3::new(f(14, "af")?, f(15, "al")?, f(16, "au")?),
        angular_rate_body_rps: Vector3::new(f(17, "wx")?, f(18, "wy")?, f(19, "wz")?),
        angular_rate_nav_rps: Vector3::new(f(20, "wf")?, f(21, "wl")?, f(22, "wu")?),
        position_accuracy_m: f(23, "pos_accuracy")?,
        velocity_accuracy_mps: f(24, "vel_accuracy")?,
        navigation_status: i(25, "navstat")?,
        number_of_satellites: i(26, "numsats")?,
        position_mode: i(27, "posmode")?,
        velocity_mode: i(28, "velmode")?,
        orientation_mode: i(29, "orimode")?,
    })
}

/// Parse the textual KITTI `timestamps.txt` (one row per frame, formatted as
/// `YYYY-MM-DD HH:MM:SS.fffffffff`) into a list of wall-clock nanoseconds since
/// the Unix epoch.
pub fn parse_kitti_oxts_timestamps_txt(text: &str) -> Result<Vec<i128>, String> {
    let mut output = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let timestamp = parse_kitti_oxts_timestamp_line(trimmed).map_err(|message| {
            format!(
                "line {line_number}: {line} ({message})",
                line_number = line_index + 1
            )
        })?;
        output.push(timestamp);
    }
    Ok(output)
}

/// Parse one KITTI `timestamps.txt` row into wall-clock nanoseconds.
pub fn parse_kitti_oxts_timestamp_line(line: &str) -> Result<i128, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err("empty timestamp line".to_owned());
    }

    // Split into `YYYY-MM-DD` and `HH:MM:SS.fffffffff` parts.
    let mut parts = trimmed.split_ascii_whitespace();
    let date_part = parts.next().ok_or_else(|| "missing date".to_owned())?;
    let time_part = parts.next().ok_or_else(|| "missing time".to_owned())?;
    if parts.next().is_some() {
        return Err("unexpected trailing tokens".to_owned());
    }

    let mut date_iter = date_part.split('-');
    let year = parse_signed_i64(date_iter.next(), "year")?;
    let month = parse_unsigned_u32(date_iter.next(), "month")?;
    let day = parse_unsigned_u32(date_iter.next(), "day")?;
    if date_iter.next().is_some() {
        return Err("malformed date (expected YYYY-MM-DD)".to_owned());
    }

    let (hms, frac) = match time_part.split_once('.') {
        Some((hms, frac)) => (hms, frac),
        None => (time_part, ""),
    };
    let mut time_iter = hms.split(':');
    let hour = parse_unsigned_u32(time_iter.next(), "hour")?;
    let minute = parse_unsigned_u32(time_iter.next(), "minute")?;
    let second = parse_unsigned_u32(time_iter.next(), "second")?;
    if time_iter.next().is_some() {
        return Err("malformed time (expected HH:MM:SS[.fraction])".to_owned());
    }

    let fractional_nanoseconds = parse_fractional_nanoseconds(frac)?;
    let day_seconds = i128::from(hour) * 3600 + i128::from(minute) * 60 + i128::from(second);
    let date_seconds = days_since_unix_epoch(year, month, day)? * 86_400;
    let total_seconds = date_seconds + day_seconds;
    let total_nanoseconds = total_seconds * 1_000_000_000 + i128::from(fractional_nanoseconds);

    Ok(total_nanoseconds)
}

fn parse_kitti_oxts_timestamps_txt_with_path(
    text: &str,
    path: &Path,
) -> Result<Vec<i128>, KittiOxtsError> {
    let mut output = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let timestamp = parse_kitti_oxts_timestamp_line(trimmed).map_err(|message| {
            KittiOxtsError::InvalidTimestampLine {
                path: path.to_path_buf(),
                line_number: line_index + 1,
                line: line.to_owned(),
                message,
            }
        })?;
        output.push(timestamp);
    }
    Ok(output)
}

fn collect_oxts_data_paths(data_dir: &Path) -> Result<Vec<PathBuf>, KittiOxtsError> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(data_dir)? {
        let path = entry?.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| extension.eq_ignore_ascii_case("txt"))
                .unwrap_or(false)
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn parse_signed_i64(token: Option<&str>, field: &str) -> Result<i64, String> {
    token
        .ok_or_else(|| format!("missing {field}"))?
        .parse::<i64>()
        .map_err(|error| format!("invalid {field}: {error}"))
}

fn parse_unsigned_u32(token: Option<&str>, field: &str) -> Result<u32, String> {
    token
        .ok_or_else(|| format!("missing {field}"))?
        .parse::<u32>()
        .map_err(|error| format!("invalid {field}: {error}"))
}

fn parse_fractional_nanoseconds(fraction: &str) -> Result<u64, String> {
    if fraction.is_empty() {
        return Ok(0);
    }
    if !fraction.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("invalid fractional seconds: {fraction}"));
    }
    let mut padded = String::with_capacity(9);
    if fraction.len() >= 9 {
        padded.push_str(&fraction[..9]);
    } else {
        padded.push_str(fraction);
        padded.extend(std::iter::repeat('0').take(9 - fraction.len()));
    }
    padded
        .parse::<u64>()
        .map_err(|error| format!("invalid fractional seconds: {error}"))
}

/// Convert a `(year, month, day)` calendar date to the number of days elapsed
/// since the Unix epoch (1970-01-01). Negative for dates before the epoch.
fn days_since_unix_epoch(year: i64, month: u32, day: u32) -> Result<i128, String> {
    if !(1..=12).contains(&month) {
        return Err(format!("invalid month: {month}"));
    }
    let days_in_month = days_in_month(year, month);
    if day < 1 || day > days_in_month {
        return Err(format!("invalid day {day} for {year}-{month:02}"));
    }

    // Howard Hinnant's days_from_civil algorithm.
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let m = u32::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    let days = i128::from(era) * 146_097 + i128::from(doe) - 719_468;
    Ok(days)
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}
