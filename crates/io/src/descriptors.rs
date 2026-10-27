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

/// Formats a store in the `LANDMARK_ID D[0] D[1] ...` text layout read by
/// [`parse_landmark_descriptors_txt`], one landmark per line in ascending id
/// order. Values use Rust's shortest round-trip `f32` formatting, so binary
/// descriptors unpacked to `0.0`/`1.0` are written as `0`/`1`.
pub fn format_landmark_descriptors_txt(store: &LandmarkDescriptorStore) -> String {
    let mut entries = store.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(landmark_id, _)| *landmark_id);
    let mut output = String::from("# LANDMARK_ID DESCRIPTOR[]\n");
    for (landmark_id, descriptor) in entries {
        output.push_str(&landmark_id.to_string());
        for value in descriptor {
            output.push(' ');
            output.push_str(&value.to_string());
        }
        output.push('\n');
    }
    output
}

pub fn write_landmark_descriptors_txt(
    store: &LandmarkDescriptorStore,
    path: impl AsRef<Path>,
) -> Result<(), DescriptorStoreError> {
    fs::write(path, format_landmark_descriptors_txt(store))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatted_store_round_trips() {
        let mut store = LandmarkDescriptorStore::new();
        store.insert(7, vec![1.0, 0.0, 1.0]);
        store.insert(3, vec![0.25, -1.5e-7, 1.0e10]);
        let text = format_landmark_descriptors_txt(&store);
        assert!(text.starts_with("# LANDMARK_ID DESCRIPTOR[]\n3 "));
        assert!(text.ends_with("7 1 0 1\n"));
        assert_eq!(parse_landmark_descriptors_txt(&text).unwrap(), store);
    }
}
