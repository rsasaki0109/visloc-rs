//! Convert the bounded binary export from `export_colmap_verified_for_visloc.py`
//! into the repository's checksummed verified-pair snapshot format.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use visloc_rs::verified_pair_snapshot::{self, PairRecord};

const MAGIC: &[u8; 16] = b"VISLOC-COLMAP-1\0";

fn next_path(args: &[String], index: &mut usize, flag: &str) -> Result<PathBuf, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("{flag} requires PATH"))?;
    *index += 1;
    Ok(PathBuf::from(value))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("read u64: {error}"))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_string(reader: &mut impl Read) -> Result<String, String> {
    let length = usize::try_from(read_u64(reader)?).map_err(|_| "string is too long")?;
    let mut bytes = vec![0u8; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("read string: {error}"))?;
    String::from_utf8(bytes).map_err(|error| format!("image name is not UTF-8: {error}"))
}

fn image_index_remap(
    source_names: &[String],
    template_names: &[String],
) -> Result<Vec<usize>, String> {
    if source_names.len() != template_names.len() {
        return Err(format!(
            "export has {} images but template has {}",
            source_names.len(),
            template_names.len()
        ));
    }
    let mut template_indices = HashMap::with_capacity(template_names.len());
    for (index, name) in template_names.iter().enumerate() {
        if template_indices.insert(name.as_str(), index).is_some() {
            return Err(format!("template repeats image name {name:?}"));
        }
    }
    let mut seen = vec![false; template_names.len()];
    let mut remap = Vec::with_capacity(source_names.len());
    for name in source_names {
        let Some(&index) = template_indices.get(name.as_str()) else {
            return Err(format!("export image {name:?} is absent from template"));
        };
        if seen[index] {
            return Err(format!("export repeats image name {name:?}"));
        }
        seen[index] = true;
        remap.push(index);
    }
    if let Some(index) = seen.iter().position(|present| !present) {
        return Err(format!(
            "template image {:?} is absent from export",
            template_names[index]
        ));
    }
    Ok(remap)
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut template = None;
    let mut pairs_bin = None;
    let mut output = None;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--template-snapshot" => {
                template = Some(next_path(&args, &mut index, "--template-snapshot")?)
            }
            "--pairs-bin" => pairs_bin = Some(next_path(&args, &mut index, "--pairs-bin")?),
            "--output" => output = Some(next_path(&args, &mut index, "--output")?),
            "--help" | "-h" => {
                return Err("usage: import_colmap_verified_snapshot --template-snapshot PATH --pairs-bin PATH --output PATH".into())
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    let template = template.ok_or("--template-snapshot is required")?;
    let pairs_bin = pairs_bin.ok_or("--pairs-bin is required")?;
    let output = output.ok_or("--output is required")?;

    let mut snapshot = verified_pair_snapshot::read(&template)?;
    snapshot.pairs.clear();
    snapshot.pairs.shrink_to_fit();
    let mut reader = BufReader::new(
        File::open(&pairs_bin).map_err(|error| format!("open {}: {error}", pairs_bin.display()))?,
    );
    let mut magic = [0u8; 16];
    reader
        .read_exact(&mut magic)
        .map_err(|error| format!("read export magic: {error}"))?;
    if &magic != MAGIC {
        return Err("unsupported COLMAP frontend export".into());
    }
    let image_count = usize::try_from(read_u64(&mut reader)?).map_err(|_| "too many images")?;
    let mut source_names = Vec::with_capacity(image_count);
    let mut source_feature_counts = Vec::with_capacity(image_count);
    for _ in 0..image_count {
        source_names.push(read_string(&mut reader)?);
        source_feature_counts.push(read_u64(&mut reader)?);
    }
    let source_to_template = image_index_remap(&source_names, &snapshot.image_names)?;
    let mut feature_counts = vec![0; image_count];
    for (source, &template) in source_to_template.iter().enumerate() {
        feature_counts[template] = source_feature_counts[source];
    }
    let pair_count = usize::try_from(read_u64(&mut reader)?).map_err(|_| "too many pairs")?;
    let mut pairs = Vec::with_capacity(pair_count);
    let mut accepted = 0u64;
    for _ in 0..pair_count {
        let source_image_i =
            usize::try_from(read_u64(&mut reader)?).map_err(|_| "image_i does not fit usize")?;
        let source_image_j =
            usize::try_from(read_u64(&mut reader)?).map_err(|_| "image_j does not fit usize")?;
        let count = usize::try_from(read_u64(&mut reader)?).map_err(|_| "too many matches")?;
        let left_count = *source_feature_counts
            .get(source_image_i)
            .ok_or("image_i is outside the manifest")?;
        let right_count = *source_feature_counts
            .get(source_image_j)
            .ok_or("image_j is outside the manifest")?;
        let template_image_i = *source_to_template
            .get(source_image_i)
            .ok_or("image_i is outside the remap")?;
        let template_image_j = *source_to_template
            .get(source_image_j)
            .ok_or("image_j is outside the remap")?;
        if template_image_i == template_image_j {
            return Err("pair maps both endpoints to one image".into());
        }
        let mut matches = Vec::with_capacity(count);
        for _ in 0..count {
            let mut bytes = [0u8; 8];
            reader
                .read_exact(&mut bytes)
                .map_err(|error| format!("read correspondence: {error}"))?;
            let left = u64::from(u32::from_le_bytes(bytes[..4].try_into().unwrap()));
            let right = u64::from(u32::from_le_bytes(bytes[4..].try_into().unwrap()));
            if left >= left_count || right >= right_count {
                return Err(format!(
                    "pair ({source_image_i},{source_image_j}) has out-of-range match ({left},{right})"
                ));
            }
            matches.push((left, right));
        }
        let (image_i, image_j) = if template_image_i < template_image_j {
            (template_image_i as u64, template_image_j as u64)
        } else {
            for pair in &mut matches {
                *pair = (pair.1, pair.0);
            }
            (template_image_j as u64, template_image_i as u64)
        };
        let accepted_inlier_indices = (0..count as u64).collect::<Vec<_>>();
        accepted += count as u64;
        pairs.push(PairRecord {
            image_i,
            image_j,
            raw_match_count: count as u64,
            raw_matches: matches.clone(),
            accepted_inlier_indices,
            essential_inlier_indices: Vec::new(),
            matches,
            essential_matches: None,
            config: 0,
            calibrated: true,
            e_inlier_count: count as u64,
            f_inlier_count: 0,
            h_inlier_count: 0,
            essential_matrix_bits: None,
            fundamental_matrix_bits: None,
            homography_matrix_bits: None,
            relative_rotation_bits: None,
            relative_translation_bits: None,
        });
    }
    pairs.sort_unstable_by_key(|pair| (pair.image_i, pair.image_j));
    if pairs.windows(2).any(|window| {
        (window[0].image_i, window[0].image_j) == (window[1].image_i, window[1].image_j)
    }) {
        return Err("export contains duplicate unordered image pairs".into());
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| format!("check trailing data: {error}"))?
        != 0
    {
        return Err("COLMAP frontend export has trailing data".into());
    }
    snapshot.feature_counts = feature_counts;
    snapshot.feature_manifest_hash = 0;
    snapshot.pairs = pairs;
    snapshot.accepted_match_count = accepted;
    snapshot.effective_config = "colmap-two-view-geometries-import-v1".into();
    snapshot.verifier_config = "colmap-two-view-geometries-import-v1".into();
    // The writer checks stream integrity and the mapper independently checks
    // image names, feature counts, and every correspondence bound.
    snapshot.effective_config_hash = 0;
    snapshot.verifier_config_hash = 0;
    snapshot.pair_order_hash = 0;
    snapshot.unordered_edge_hash = 0;
    verified_pair_snapshot::write_atomic(&output, &snapshot)?;
    println!(
        "imported {image_count} images / {pair_count} pairs / {accepted} correspondences -> {}",
        output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::image_index_remap;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn image_remap_accepts_a_permutation() {
        assert_eq!(
            image_index_remap(&names(&["b", "c", "a"]), &names(&["a", "b", "c"])),
            Ok(vec![1, 2, 0])
        );
    }

    #[test]
    fn image_remap_rejects_missing_and_duplicate_names() {
        assert!(image_index_remap(&names(&["a", "x"]), &names(&["a", "b"])).is_err());
        assert!(image_index_remap(&names(&["a", "a"]), &names(&["a", "b"])).is_err());
        assert!(image_index_remap(&names(&["a", "b"]), &names(&["a", "a"])).is_err());
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
