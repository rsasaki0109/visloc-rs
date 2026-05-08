use std::fs;
use std::path::Path;

use nalgebra::Point3;
use thiserror::Error;
use visloc_fusion::{GnssMeasurement, Timestamp};

#[derive(Debug, Error)]
pub enum SensorLogError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GNSS measurement line {line_number}: {line} ({message})")]
    InvalidGnssLine {
        line_number: usize,
        line: String,
        message: String,
    },
}

pub fn read_gnss_measurements_txt(
    path: impl AsRef<Path>,
) -> Result<Vec<GnssMeasurement>, SensorLogError> {
    parse_gnss_measurements_txt(&fs::read_to_string(path)?)
}

pub fn parse_gnss_measurements_txt(text: &str) -> Result<Vec<GnssMeasurement>, SensorLogError> {
    let mut measurements = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || is_gnss_header_line(trimmed) {
            continue;
        }

        let tokens = split_sensor_tokens(trimmed);
        if tokens.len() < 4 {
            return Err(SensorLogError::InvalidGnssLine {
                line_number,
                line: line.to_owned(),
                message: "expected timestamp_ns x y z [horizontal_accuracy] [vertical_accuracy]"
                    .to_owned(),
            });
        }

        let timestamp = parse_i128_token(tokens[0], line_number, line, "timestamp_ns")?;
        let x = parse_f64_token(tokens[1], line_number, line, "x")?;
        let y = parse_f64_token(tokens[2], line_number, line, "y")?;
        let z = parse_f64_token(tokens[3], line_number, line, "z")?;
        let horizontal_accuracy = tokens
            .get(4)
            .map(|token| parse_f64_token(token, line_number, line, "horizontal_accuracy"))
            .transpose()?;
        let vertical_accuracy = tokens
            .get(5)
            .map(|token| parse_f64_token(token, line_number, line, "vertical_accuracy"))
            .transpose()?;

        measurements.push(
            GnssMeasurement::new(Timestamp::from_nanoseconds(timestamp), Point3::new(x, y, z))
                .with_accuracy(horizontal_accuracy, vertical_accuracy),
        );
    }
    Ok(measurements)
}

fn split_sensor_tokens(line: &str) -> Vec<&str> {
    line.split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|token| !token.is_empty())
        .collect()
}

fn is_gnss_header_line(line: &str) -> bool {
    let Some(first_token) = split_sensor_tokens(line).first().copied() else {
        return false;
    };
    matches!(
        first_token.to_ascii_lowercase().as_str(),
        "timestamp" | "timestamp_ns" | "time_ns"
    )
}

fn parse_i128_token(
    token: &str,
    line_number: usize,
    line: &str,
    field: &str,
) -> Result<i128, SensorLogError> {
    token
        .parse::<i128>()
        .map_err(|error| SensorLogError::InvalidGnssLine {
            line_number,
            line: line.to_owned(),
            message: format!("invalid {field}: {error}"),
        })
}

fn parse_f64_token(
    token: &str,
    line_number: usize,
    line: &str,
    field: &str,
) -> Result<f64, SensorLogError> {
    token
        .parse::<f64>()
        .map_err(|error| SensorLogError::InvalidGnssLine {
            line_number,
            line: line.to_owned(),
            message: format!("invalid {field}: {error}"),
        })
}
