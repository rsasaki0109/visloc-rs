//! IMU samples from a CSV file, `time, gx, gy, gz, ax, ay, az` per line.
//!
//! Three layouts, told apart by the header line:
//! - EuRoC (`#timestamp [ns],w_RS_S_x,...`): nanoseconds, rad/s and m/s².
//! - OverMaps360's `insta360_imu.csv` (`t_sec,gyro_x_dps,...,accel_x_mg,...`):
//!   seconds and, despite the labels, the Insta360 X5's raw counts, 16.4 per
//!   deg/s and 1024 per g (docs/insta360_imu_scale.md).
//! - Anything else (or no header): seconds, rad/s and m/s².
//!
//! Lines starting with `#` after the first are comments.

use std::fs::read_to_string;
use std::path::{Path, PathBuf};

use nalgebra::Vector3;

/// Insta360 X5 sensitivities (±2000 dps, ±32 g).
pub const INSTA360_GYRO_COUNTS_PER_DPS: f64 = 16.4;
pub const INSTA360_ACCEL_COUNTS_PER_G: f64 = 1024.0;
pub const STANDARD_GRAVITY: f64 = 9.80665;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuCsvLayout {
    /// Seconds, rad/s, m/s².
    Si,
    /// EuRoC `imu0/data.csv`: nanoseconds, rad/s, m/s².
    Euroc,
    /// OverMaps360 `insta360_imu.csv`: seconds and raw Insta360 X5 counts.
    OverMaps360Counts,
}

/// IMU samples converted to seconds, rad/s and m/s².
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImuCsv {
    pub time: Vec<f64>,
    pub gyro: Vec<Vector3<f64>>,
    pub accel: Vec<Vector3<f64>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ImuCsvError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid IMU CSV {path}: line {line_number}: {message}")]
    InvalidCsv {
        path: PathBuf,
        line_number: usize,
        message: String,
    },
    #[error("IMU CSV {0} has no samples")]
    Empty(PathBuf),
}

/// The layout a header line names.
pub fn imu_csv_layout(header: &str) -> ImuCsvLayout {
    let h = header.to_ascii_lowercase();
    if h.contains("gyro_x_dps") && h.contains("accel_x_mg") {
        ImuCsvLayout::OverMaps360Counts
    } else if h.starts_with('#') && h.contains("[ns]") {
        ImuCsvLayout::Euroc
    } else {
        ImuCsvLayout::Si
    }
}

/// Reads an IMU CSV file (see the module docs) and returns its samples in
/// file order, with the layout it detected.
pub fn read_imu_csv(path: &Path) -> Result<(ImuCsv, ImuCsvLayout), ImuCsvError> {
    let text = read_to_string(path).map_err(|source| ImuCsvError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut lines = text.lines().enumerate().peekable();
    let first = lines.peek().map(|(_, l)| l.trim()).unwrap_or("");
    let numeric = first
        .split(',')
        .next()
        .is_some_and(|f| f.trim().parse::<f64>().is_ok());
    let layout = if numeric {
        ImuCsvLayout::Si
    } else {
        imu_csv_layout(first)
    };
    if !numeric {
        lines.next();
    }
    let (time_scale, gyro_scale, accel_scale) = match layout {
        ImuCsvLayout::Si => (1.0, 1.0, 1.0),
        ImuCsvLayout::Euroc => (1e-9, 1.0, 1.0),
        ImuCsvLayout::OverMaps360Counts => (
            1.0,
            (1.0 / INSTA360_GYRO_COUNTS_PER_DPS).to_radians(),
            STANDARD_GRAVITY / INSTA360_ACCEL_COUNTS_PER_G,
        ),
    };
    let mut out = ImuCsv::default();
    for (index, raw) in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let invalid = |message: String| ImuCsvError::InvalidCsv {
            path: path.to_path_buf(),
            line_number: index + 1,
            message,
        };
        let values: Vec<f64> = line
            .split(',')
            .map(|f| f.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| invalid(e.to_string()))?;
        if values.len() < 7 {
            return Err(invalid(format!("{} columns, need 7", values.len())));
        }
        out.time.push(values[0] * time_scale);
        out.gyro
            .push(Vector3::new(values[1], values[2], values[3]) * gyro_scale);
        out.accel
            .push(Vector3::new(values[4], values[5], values[6]) * accel_scale);
    }
    if out.time.is_empty() {
        return Err(ImuCsvError::Empty(path.to_path_buf()));
    }
    Ok((out, layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> (ImuCsv, ImuCsvLayout) {
        let dir =
            std::env::temp_dir().join(format!("imu_csv_{}_{}", std::process::id(), text.len()));
        std::fs::write(&dir, text).unwrap();
        let out = read_imu_csv(&dir).unwrap();
        std::fs::remove_file(&dir).unwrap();
        out
    }

    #[test]
    fn reads_the_three_layouts() {
        let (si, layout) = read("0.0,0.1,0.2,0.3,0,0,9.8\n0.001,0.1,0.2,0.3,0,0,9.8\n");
        assert_eq!(layout, ImuCsvLayout::Si);
        assert_eq!(si.time, vec![0.0, 0.001]);
        assert_eq!(si.accel[1], Vector3::new(0.0, 0.0, 9.8));

        let (euroc, layout) = read(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n\
             1403636579758555392, 0.1, 0.2, 0.3, 9.0, 0.1, -3.0\n",
        );
        assert_eq!(layout, ImuCsvLayout::Euroc);
        assert!((euroc.time[0] - 1_403_636_579.758_555).abs() < 1e-5);
        assert_eq!(euroc.gyro[0], Vector3::new(0.1, 0.2, 0.3));

        let (counts, layout) = read(
            "t_sec,gyro_x_dps,gyro_y_dps,gyro_z_dps,accel_x_mg,accel_y_mg,accel_z_mg\n\
             -0.27,16.4,0,-32.8,0,-1024,0\n",
        );
        assert_eq!(layout, ImuCsvLayout::OverMaps360Counts);
        assert!((counts.gyro[0].x - 1f64.to_radians()).abs() < 1e-12);
        assert!((counts.gyro[0].z + 2f64.to_radians()).abs() < 1e-12);
        assert!((counts.accel[0].y + STANDARD_GRAVITY).abs() < 1e-12);
    }
}
