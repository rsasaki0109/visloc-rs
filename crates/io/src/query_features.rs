use std::fs;
use std::num::ParseFloatError;
use std::path::Path;
use thiserror::Error;
use visloc_vision::features::{FeatureSet, FeatureSetError};

use nalgebra::Point2;

#[derive(Debug, Error)]
pub enum QueryFeatureError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("float parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("feature set error: {0}")]
    FeatureSet(#[from] FeatureSetError),
    #[error("invalid query feature line: {line}")]
    InvalidLine { line: String },
}

pub fn read_query_features_txt(path: impl AsRef<Path>) -> Result<FeatureSet, QueryFeatureError> {
    parse_query_features_txt(&fs::read_to_string(path)?)
}

pub fn parse_query_features_txt(contents: &str) -> Result<FeatureSet, QueryFeatureError> {
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 3 {
            return Err(QueryFeatureError::InvalidLine {
                line: line.to_owned(),
            });
        }

        keypoints.push(Point2::new(tokens[0].parse()?, tokens[1].parse()?));
        descriptors.push(
            tokens[2..]
                .iter()
                .map(|value| value.parse::<f32>())
                .collect::<Result<Vec<_>, _>>()?,
        );
    }

    Ok(FeatureSet::new(keypoints, descriptors)?)
}
