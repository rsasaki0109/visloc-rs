//! Overlay verified pairs from a disjoint per-image feature space.
//!
//! The base feature indices remain an unchanged prefix. Addition indices are
//! shifted by the base count for each image, and only image pairs absent from
//! the complete base snapshot are admitted. Selected pairs are inserted after
//! the trusted structure prefix and before its deferred tail. This keeps the
//! two track graphs disjoint while allowing a bounded second detector to add
//! independently verified local structure without rematching the base bank.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use visloc_rs::verified_pair_snapshot::{merge_owned, read, write_atomic, PairRecord, Snapshot};

fn usage() -> ! {
    eprintln!(
        "usage: overlay_disjoint_verified_snapshot \
         --base-snapshot PATH --addition-snapshot PATH --rig-manifest PATH \
         --structure-pair-prefix COUNT [--min-frame-gap COUNT] \
         --max-frame-gap COUNT [--include-same-frame-supplements] --output PATH"
    );
    std::process::exit(2);
}

fn next_path(args: &[String], index: &mut usize, flag: &str) -> Result<PathBuf, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("{flag} requires PATH"))?;
    *index += 1;
    Ok(PathBuf::from(value))
}

fn next_usize(args: &[String], index: &mut usize, flag: &str) -> Result<usize, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("{flag} requires COUNT"))?;
    *index += 1;
    value
        .parse::<usize>()
        .map_err(|error| format!("{flag} requires COUNT: {error}"))
}

fn validate_envelopes(base: &Snapshot, addition: &Snapshot) -> Result<(), String> {
    if base.image_names != addition.image_names
        || base.image_manifest_hash != addition.image_manifest_hash
    {
        return Err("base and addition image manifests differ".into());
    }
    if base.width != addition.width
        || base.height != addition.height
        || base.intrinsics_bits != addition.intrinsics_bits
    {
        return Err("base and addition camera envelopes differ".into());
    }
    if base.verifier_config_hash != addition.verifier_config_hash
        || base.verifier_config != addition.verifier_config
    {
        let parse = |config: &str| -> Result<(usize, HashMap<String, String>), String> {
            let mut fields = HashMap::new();
            for field in config.split(';') {
                let (key, value) = field.split_once('=').ok_or_else(|| {
                    "verifier configuration contains a malformed field".to_owned()
                })?;
                if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                    return Err(format!("verifier configuration repeats field {key:?}"));
                }
            }
            let minimum = fields
                .remove("min_matches")
                .ok_or_else(|| "verifier configuration omits min_matches".to_owned())?
                .parse::<usize>()
                .map_err(|error| format!("invalid verifier min_matches: {error}"))?;
            Ok((minimum, fields))
        };
        let (base_minimum, base_fields) = parse(&base.verifier_config)?;
        let (addition_minimum, addition_fields) = parse(&addition.verifier_config)?;
        if base_fields != addition_fields || addition_minimum < base_minimum {
            return Err(
                "addition verifier must differ only by an equal or stricter min_matches".into(),
            );
        }
    }
    if base.feature_counts.len() != addition.feature_counts.len() {
        return Err("base and addition feature-count vectors differ in length".into());
    }
    Ok(())
}

fn parse_image_frames(path: &Path, image_names: &[String]) -> Result<Vec<usize>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let image_index = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut frames = vec![None; image_names.len()];
    for (zero_line, raw) in text.lines().enumerate() {
        let fields = raw.split_whitespace().collect::<Vec<_>>();
        if fields.first().copied() != Some("F") {
            continue;
        }
        if fields.len() != 4 {
            return Err(format!(
                "rig manifest {}:{} has malformed frame row",
                path.display(),
                zero_line + 1
            ));
        }
        let frame = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "rig manifest {}:{} has invalid frame id: {error}",
                path.display(),
                zero_line + 1
            )
        })?;
        let image = *image_index.get(fields[2]).ok_or_else(|| {
            format!(
                "rig manifest {}:{} names unknown image {:?}",
                path.display(),
                zero_line + 1,
                fields[2]
            )
        })?;
        if frames[image].replace(frame).is_some() {
            return Err(format!("rig manifest assigns image {:?} twice", fields[2]));
        }
    }
    frames
        .into_iter()
        .enumerate()
        .map(|(image, frame)| frame.ok_or_else(|| format!("rig manifest omits image {image}")))
        .collect()
}

fn shift_match(
    matched: (u64, u64),
    image_i: usize,
    image_j: usize,
    base_counts: &[u64],
) -> Result<(u64, u64), String> {
    Ok((
        matched
            .0
            .checked_add(base_counts[image_i])
            .ok_or_else(|| "shifted left keypoint overflows u64".to_owned())?,
        matched
            .1
            .checked_add(base_counts[image_j])
            .ok_or_else(|| "shifted right keypoint overflows u64".to_owned())?,
    ))
}

fn shift_pair(pair: &PairRecord, base_counts: &[u64]) -> Result<PairRecord, String> {
    let image_i = usize::try_from(pair.image_i)
        .map_err(|_| "addition image_i does not fit usize".to_owned())?;
    let image_j = usize::try_from(pair.image_j)
        .map_err(|_| "addition image_j does not fit usize".to_owned())?;
    if image_i >= base_counts.len() || image_j >= base_counts.len() {
        return Err("addition pair image index is outside the base manifest".into());
    }
    let mut shifted = pair.clone();
    shifted.raw_matches = pair
        .raw_matches
        .iter()
        .copied()
        .map(|matched| shift_match(matched, image_i, image_j, base_counts))
        .collect::<Result<_, _>>()?;
    shifted.matches = pair
        .matches
        .iter()
        .copied()
        .map(|matched| shift_match(matched, image_i, image_j, base_counts))
        .collect::<Result<_, _>>()?;
    shifted.essential_matches = pair
        .essential_matches
        .as_ref()
        .map(|matches| {
            matches
                .iter()
                .copied()
                .map(|matched| shift_match(matched, image_i, image_j, base_counts))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    Ok(shifted)
}

fn append_disjoint_matches(base: &mut PairRecord, shifted: &PairRecord) -> Result<(), String> {
    if (base.image_i, base.image_j) != (shifted.image_i, shifted.image_j) {
        return Err("duplicate pair orientations differ".into());
    }
    let raw_offset = base.raw_matches.len() as u64;
    base.raw_matches.extend(shifted.raw_matches.iter().copied());
    base.accepted_inlier_indices.extend(
        shifted
            .accepted_inlier_indices
            .iter()
            .map(|index| index + raw_offset),
    );
    base.matches.extend(shifted.matches.iter().copied());
    if let Some(matches) = shifted.essential_matches.as_ref() {
        base.essential_inlier_indices.extend(
            shifted
                .essential_inlier_indices
                .iter()
                .map(|index| index + raw_offset),
        );
        base.essential_matches
            .get_or_insert_with(Vec::new)
            .extend(matches.iter().copied());
    }
    base.raw_match_count = base.raw_matches.len() as u64;
    base.e_inlier_count = base.e_inlier_count.saturating_add(shifted.e_inlier_count);
    base.f_inlier_count = base.f_inlier_count.saturating_add(shifted.f_inlier_count);
    base.h_inlier_count = base.h_inlier_count.saturating_add(shifted.h_inlier_count);
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OverlayStats {
    structure_pairs: usize,
    deferred_pairs: usize,
    admitted_pairs: usize,
    admitted_matches: u64,
    supplemented_pairs: usize,
    supplemented_matches: u64,
    duplicate_pair_rejections: usize,
    frame_gap_rejections: usize,
}

fn overlay_disjoint(
    mut base: Snapshot,
    addition: &Snapshot,
    image_frames: &[usize],
    structure_pair_prefix: usize,
    min_frame_gap: usize,
    max_frame_gap: usize,
    include_same_frame_supplements: bool,
) -> Result<(Snapshot, OverlayStats), String> {
    validate_envelopes(&base, addition)?;
    if structure_pair_prefix > base.pairs.len() {
        return Err(format!(
            "structure pair prefix {structure_pair_prefix} exceeds {} base pairs",
            base.pairs.len()
        ));
    }
    let base_keys = base
        .pairs
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            (
                (
                    pair.image_i.min(pair.image_j),
                    pair.image_i.max(pair.image_j),
                ),
                index,
            )
        })
        .collect::<HashMap<_, _>>();
    let mut admitted = Vec::new();
    let mut admitted_matches = 0u64;
    let mut duplicate_pair_rejections = 0usize;
    let mut frame_gap_rejections = 0usize;
    let mut supplemented_pairs = 0usize;
    let mut supplemented_matches = 0u64;
    for pair in &addition.pairs {
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        let image_i = usize::try_from(pair.image_i)
            .map_err(|_| "addition image_i does not fit usize".to_owned())?;
        let image_j = usize::try_from(pair.image_j)
            .map_err(|_| "addition image_j does not fit usize".to_owned())?;
        let frame_i = *image_frames
            .get(image_i)
            .ok_or_else(|| "addition image_i is outside rig frames".to_owned())?;
        let frame_j = *image_frames
            .get(image_j)
            .ok_or_else(|| "addition image_j is outside rig frames".to_owned())?;
        let frame_gap = frame_i.abs_diff(frame_j);
        if let Some(&base_index) = base_keys.get(&key) {
            if include_same_frame_supplements && frame_gap == 0 {
                let shifted = shift_pair(pair, &base.feature_counts)?;
                append_disjoint_matches(&mut base.pairs[base_index], &shifted)?;
                supplemented_pairs += 1;
                supplemented_matches = supplemented_matches
                    .checked_add(shifted.matches.len() as u64)
                    .ok_or_else(|| "supplemented match count overflows u64".to_owned())?;
            } else {
                duplicate_pair_rejections += 1;
            }
            continue;
        }
        if frame_gap < min_frame_gap || frame_gap > max_frame_gap {
            frame_gap_rejections += 1;
            continue;
        }
        admitted_matches = admitted_matches
            .checked_add(pair.matches.len() as u64)
            .ok_or_else(|| "admitted match count overflows u64".to_owned())?;
        admitted.push(shift_pair(pair, &base.feature_counts)?);
    }
    let admitted_pairs = admitted.len();
    let deferred = base.pairs.split_off(structure_pair_prefix);
    base.pairs.extend(admitted);
    base.pairs.extend(deferred);
    for (base_count, addition_count) in base.feature_counts.iter_mut().zip(&addition.feature_counts)
    {
        *base_count = base_count
            .checked_add(*addition_count)
            .ok_or_else(|| "combined feature count overflows u64".to_owned())?;
    }
    let provenance = format!(
        "disjoint-overlay-v1;base_feature_manifest={:016x};addition_feature_manifest={:016x};structure_prefix={structure_pair_prefix};min_frame_gap={min_frame_gap};max_frame_gap={max_frame_gap};same_frame_supplements={include_same_frame_supplements}",
        base.feature_manifest_hash, addition.feature_manifest_hash
    );
    base.feature_manifest_hash = fnv1a64(provenance.as_bytes());
    base.effective_config = format!("{};{provenance}", base.effective_config);
    let output = merge_owned(vec![base])?;
    let deferred_pairs = output.pairs.len() - structure_pair_prefix - admitted_pairs;
    Ok((
        output,
        OverlayStats {
            structure_pairs: structure_pair_prefix + admitted_pairs,
            deferred_pairs,
            admitted_pairs,
            admitted_matches,
            supplemented_pairs,
            supplemented_matches,
            duplicate_pair_rejections,
            frame_gap_rejections,
        },
    ))
}

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
    }
    let mut base_path = None;
    let mut addition_path = None;
    let mut rig_manifest = None;
    let mut output_path = None;
    let mut structure_pair_prefix = None;
    let mut min_frame_gap = 0usize;
    let mut max_frame_gap = None;
    let mut include_same_frame_supplements = false;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--base-snapshot" => base_path = Some(next_path(&args, &mut index, "--base-snapshot")?),
            "--addition-snapshot" => {
                addition_path = Some(next_path(&args, &mut index, "--addition-snapshot")?)
            }
            "--rig-manifest" => {
                rig_manifest = Some(next_path(&args, &mut index, "--rig-manifest")?)
            }
            "--structure-pair-prefix" => {
                structure_pair_prefix =
                    Some(next_usize(&args, &mut index, "--structure-pair-prefix")?)
            }
            "--max-frame-gap" => {
                max_frame_gap = Some(next_usize(&args, &mut index, "--max-frame-gap")?)
            }
            "--min-frame-gap" => min_frame_gap = next_usize(&args, &mut index, "--min-frame-gap")?,
            "--include-same-frame-supplements" => include_same_frame_supplements = true,
            "--output" => output_path = Some(next_path(&args, &mut index, "--output")?),
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    let base_path = base_path.ok_or_else(|| "--base-snapshot is required".to_owned())?;
    let addition_path =
        addition_path.ok_or_else(|| "--addition-snapshot is required".to_owned())?;
    let rig_manifest = rig_manifest.ok_or_else(|| "--rig-manifest is required".to_owned())?;
    let output_path = output_path.ok_or_else(|| "--output is required".to_owned())?;
    let structure_pair_prefix =
        structure_pair_prefix.ok_or_else(|| "--structure-pair-prefix is required".to_owned())?;
    let max_frame_gap = max_frame_gap.ok_or_else(|| "--max-frame-gap is required".to_owned())?;
    if min_frame_gap > max_frame_gap {
        return Err("--min-frame-gap cannot exceed --max-frame-gap".into());
    }
    let base = read(&base_path)
        .map_err(|error| format!("invalid base snapshot {}: {error}", base_path.display()))?;
    let addition = read(&addition_path).map_err(|error| {
        format!(
            "invalid addition snapshot {}: {error}",
            addition_path.display()
        )
    })?;
    let image_frames = parse_image_frames(&rig_manifest, &base.image_names)?;
    let (output, stats) = overlay_disjoint(
        base,
        &addition,
        &image_frames,
        structure_pair_prefix,
        min_frame_gap,
        max_frame_gap,
        include_same_frame_supplements,
    )?;
    write_atomic(&output_path, &output)?;
    println!(
        "disjoint verified overlay: structure_pairs={} deferred_pairs={} admitted_pairs={} \
         admitted_matches={} supplemented_pairs={} supplemented_matches={} \
         duplicate_pair_rejections={} frame_gap_rejections={} \
         output_pairs={} output_matches={} output_features={} -> {}",
        stats.structure_pairs,
        stats.deferred_pairs,
        stats.admitted_pairs,
        stats.admitted_matches,
        stats.supplemented_pairs,
        stats.supplemented_matches,
        stats.duplicate_pair_rejections,
        stats.frame_gap_rejections,
        output.pairs.len(),
        output.accepted_match_count,
        output.feature_counts.iter().sum::<u64>(),
        output_path.display(),
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(left: u64, right: u64, count: usize) -> PairRecord {
        PairRecord {
            image_i: left,
            image_j: right,
            raw_match_count: count as u64,
            raw_matches: (0..count)
                .map(|index| (index as u64, index as u64))
                .collect(),
            accepted_inlier_indices: (0..count as u64).collect(),
            essential_inlier_indices: Vec::new(),
            matches: (0..count)
                .map(|index| (index as u64, index as u64))
                .collect(),
            essential_matches: None,
            config: 0,
            calibrated: true,
            e_inlier_count: count as u64,
            f_inlier_count: count as u64,
            h_inlier_count: 0,
            essential_matrix_bits: None,
            fundamental_matrix_bits: None,
            homography_matrix_bits: None,
            relative_rotation_bits: None,
            relative_translation_bits: None,
        }
    }

    fn snapshot(pairs: Vec<PairRecord>) -> Snapshot {
        Snapshot {
            schema_version: 1,
            image_names: (0..4).map(|index| format!("image-{index}.png")).collect(),
            image_manifest_hash: 1,
            feature_manifest_hash: 2,
            feature_counts: vec![8; 4],
            width: 10,
            height: 10,
            intrinsics_bits: [0; 4],
            effective_config_hash: 3,
            effective_config: "config".into(),
            verifier_config_hash: 4,
            verifier_config: "verifier".into(),
            pair_order_hash: 0,
            unordered_edge_hash: 0,
            accepted_match_count: pairs.iter().map(|pair| pair.matches.len() as u64).sum(),
            pairs,
        }
    }

    #[test]
    fn shifts_new_local_pairs_between_structure_and_deferred_streams() {
        let base = snapshot(vec![pair(0, 1, 2), pair(2, 3, 2)]);
        let addition = snapshot(vec![pair(0, 1, 3), pair(0, 2, 2), pair(0, 3, 4)]);
        let (output, stats) =
            overlay_disjoint(base, &addition, &[0, 1, 2, 3], 1, 0, 2, false).unwrap();
        assert_eq!(stats.structure_pairs, 2);
        assert_eq!(stats.deferred_pairs, 1);
        assert_eq!(stats.admitted_pairs, 1);
        assert_eq!(stats.admitted_matches, 2);
        assert_eq!(stats.supplemented_pairs, 0);
        assert_eq!(stats.duplicate_pair_rejections, 1);
        assert_eq!(stats.frame_gap_rejections, 1);
        assert_eq!(output.feature_counts, vec![16; 4]);
        assert_eq!(output.pairs[1].matches, vec![(8, 8), (9, 9)]);
        assert_eq!(
            output
                .pairs
                .iter()
                .map(|pair| (pair.image_i, pair.image_j))
                .collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (2, 3)]
        );
    }

    #[test]
    fn appends_shifted_same_frame_matches_to_the_existing_pair() {
        let base = snapshot(vec![pair(0, 1, 2)]);
        let addition = snapshot(vec![pair(0, 1, 3)]);
        let (output, stats) =
            overlay_disjoint(base, &addition, &[0, 0, 1, 1], 1, 32, 128, true).unwrap();
        assert_eq!(stats.admitted_pairs, 0);
        assert_eq!(stats.supplemented_pairs, 1);
        assert_eq!(stats.supplemented_matches, 3);
        assert_eq!(stats.duplicate_pair_rejections, 0);
        assert_eq!(output.feature_counts, vec![16; 4]);
        assert_eq!(
            output.pairs[0].matches,
            vec![(0, 0), (1, 1), (8, 8), (9, 9), (10, 10)]
        );
        assert_eq!(output.pairs[0].accepted_inlier_indices, vec![0, 1, 2, 3, 4]);
    }
}
