use std::fs;
use std::num::{ParseFloatError, ParseIntError};
use std::path::Path;
use thiserror::Error;
use visloc_core::types::{LandmarkDescriptorStore, LandmarkId};

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

/// Descriptor rows in the `LANDMARK_ID D[0] D[1] ...` layout of
/// [`parse_landmark_descriptors_txt`], with ids repeated freely (several
/// descriptors per landmark), in file order.
pub fn parse_landmark_descriptor_rows(
    contents: &str,
) -> Result<Vec<(LandmarkId, Vec<f32>)>, DescriptorStoreError> {
    let mut rows = Vec::new();
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let landmark_id = tokens
            .next()
            .ok_or_else(|| DescriptorStoreError::InvalidLine {
                line: line.to_owned(),
            })?
            .parse()?;
        let descriptor = tokens
            .map(str::parse::<f32>)
            .collect::<Result<Vec<_>, _>>()?;
        if descriptor.is_empty() {
            return Err(DescriptorStoreError::InvalidLine {
                line: line.to_owned(),
            });
        }
        rows.push((landmark_id, descriptor));
    }
    Ok(rows)
}

pub fn read_landmark_descriptor_rows(
    path: impl AsRef<Path>,
) -> Result<Vec<(LandmarkId, Vec<f32>)>, DescriptorStoreError> {
    parse_landmark_descriptor_rows(&fs::read_to_string(path)?)
}

/// Formats descriptor rows, in the order given, for
/// [`parse_landmark_descriptor_rows`].
pub fn format_landmark_descriptor_rows<D: AsRef<[f32]>>(rows: &[(LandmarkId, D)]) -> String {
    let mut output = String::from("# LANDMARK_ID DESCRIPTOR[] (several rows per landmark)\n");
    for (landmark_id, descriptor) in rows {
        output.push_str(&landmark_id.to_string());
        for value in descriptor.as_ref() {
            output.push(' ');
            output.push_str(&value.to_string());
        }
        output.push('\n');
    }
    output
}

pub fn write_landmark_descriptor_rows<D: AsRef<[f32]>>(
    rows: &[(LandmarkId, D)],
    path: impl AsRef<Path>,
) -> Result<(), DescriptorStoreError> {
    fs::write(path, format_landmark_descriptor_rows(rows))?;
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

    #[test]
    fn descriptor_rows_keep_repeated_ids_in_order() {
        let rows = vec![
            (7_u64, vec![1.0_f32, 0.0]),
            (3, vec![0.0, 0.0]),
            (7, vec![0.0, 1.0]),
        ];
        let text = format_landmark_descriptor_rows(&rows);
        assert_eq!(parse_landmark_descriptor_rows(&text).unwrap(), rows);
        assert!(parse_landmark_descriptor_rows("5\n").is_err());
    }
}
