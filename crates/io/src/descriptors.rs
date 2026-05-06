use std::fs;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use thiserror::Error;
use visloc_core::types::LandmarkDescriptorStore;

#[derive(Debug, Error)]
pub enum DescriptorStoreError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integer parse error: {0}")]
    ParseInt(#[from] ParseIntError),
    #[error("float parse error: {0}")]
    ParseFloat(#[from] ParseFloatError),
    #[error("invalid descriptor line: {line}")]
    InvalidLine { line: String },
}

pub fn read_landmark_descriptors_txt(
    path: impl AsRef<Path>,
) -> Result<LandmarkDescriptorStore, DescriptorStoreError> {
    parse_landmark_descriptors_txt(&fs::read_to_string(path)?)
}

pub fn parse_landmark_descriptors_txt(
    contents: &str,
) -> Result<LandmarkDescriptorStore, DescriptorStoreError> {
    let mut store = LandmarkDescriptorStore::new();

    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.len() < 2 {
            return Err(DescriptorStoreError::InvalidLine {
                line: line.to_owned(),
            });
        }

        let landmark_id = tokens[0].parse()?;
        let descriptor = tokens[1..]
            .iter()
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        if descriptor.is_empty() {
            return Err(DescriptorStoreError::InvalidLine {
                line: line.to_owned(),
            });
        }

        store.insert(landmark_id, descriptor);
    }

    Ok(store)
}
