//! Inspect a verified-pair snapshot without rerunning matching or mapping.
//!
//! The TSV output is intentionally simple so benchmark scripts can compute
//! verified-edge recall against a diagnostic control while the binary codec
//! remains implemented and validated in Rust.

use std::fmt::Write as _;
use std::path::PathBuf;

use visloc_rs::verified_pair_snapshot::read;

fn usage() -> ! {
    eprintln!(
        "usage: inspect_verified_pair_snapshot --snapshot PATH [--images-tsv PATH] [--pairs-tsv PATH] [--relative-poses-tsv PATH] [--pair-matches IMAGE_I,IMAGE_J]"
    );
    std::process::exit(2);
}

fn next_path(args: &[String], index: &mut usize, flag: &str) -> Result<PathBuf, String> {
    let Some(value) = args.get(*index + 1) else {
        return Err(format!("{flag} requires PATH"));
    };
    *index += 1;
    if value.trim().is_empty() {
        return Err(format!("{flag} requires a non-empty PATH"));
    }
    Ok(PathBuf::from(value))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut snapshot_path = None;
    let mut pairs_path = None;
    let mut images_path = None;
    let mut relative_poses_path = None;
    let mut pair_matches = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--snapshot" => snapshot_path = Some(next_path(&args, &mut index, "--snapshot")?),
            "--images-tsv" => images_path = Some(next_path(&args, &mut index, "--images-tsv")?),
            "--pairs-tsv" => pairs_path = Some(next_path(&args, &mut index, "--pairs-tsv")?),
            "--relative-poses-tsv" => {
                relative_poses_path = Some(next_path(&args, &mut index, "--relative-poses-tsv")?)
            }
            "--pair-matches" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--pair-matches requires IMAGE_I,IMAGE_J".to_owned())?;
                index += 1;
                let (left, right) = value
                    .split_once(',')
                    .ok_or_else(|| "--pair-matches requires IMAGE_I,IMAGE_J".to_owned())?;
                pair_matches = Some((
                    left.parse::<u64>().map_err(|error| error.to_string())?,
                    right.parse::<u64>().map_err(|error| error.to_string())?,
                ));
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    let snapshot_path = snapshot_path.ok_or_else(|| "--snapshot is required".to_owned())?;
    let snapshot = read(&snapshot_path)
        .map_err(|error| format!("invalid snapshot {}: {error}", snapshot_path.display()))?;
    if let Some(path) = images_path {
        let mut tsv = String::from("image_index\timage_name\tfeature_count\n");
        for (image_index, (image_name, feature_count)) in snapshot
            .image_names
            .iter()
            .zip(&snapshot.feature_counts)
            .enumerate()
        {
            writeln!(tsv, "{image_index}\t{image_name}\t{feature_count}")
                .map_err(|error| error.to_string())?;
        }
        std::fs::write(&path, tsv)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    if let Some(path) = pairs_path {
        let mut tsv = String::from(
            "image_i\timage_j\traw_matches\taccepted_matches\te_inliers\tf_inliers\th_inliers\n",
        );
        for pair in &snapshot.pairs {
            writeln!(
                tsv,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                pair.image_i,
                pair.image_j,
                pair.raw_match_count,
                pair.matches.len(),
                pair.e_inlier_count,
                pair.f_inlier_count,
                pair.h_inlier_count,
            )
            .map_err(|error| error.to_string())?;
        }
        std::fs::write(&path, tsv)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    if let Some(path) = relative_poses_path {
        let mut tsv = String::from(
            "image_i\timage_j\te_inliers\tr00\tr01\tr02\tr10\tr11\tr12\tr20\tr21\tr22\ttx\tty\ttz\n",
        );
        for pair in &snapshot.pairs {
            let (Some(rotation), Some(translation)) =
                (pair.relative_rotation_bits, pair.relative_translation_bits)
            else {
                continue;
            };
            write!(
                tsv,
                "{}\t{}\t{}",
                pair.image_i, pair.image_j, pair.e_inlier_count
            )
            .map_err(|error| error.to_string())?;
            for value in rotation {
                write!(tsv, "\t{:.17}", f64::from_bits(value))
                    .map_err(|error| error.to_string())?;
            }
            for value in translation {
                write!(tsv, "\t{:.17}", f64::from_bits(value))
                    .map_err(|error| error.to_string())?;
            }
            writeln!(tsv).map_err(|error| error.to_string())?;
        }
        std::fs::write(&path, tsv)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    }
    if let Some((image_i, image_j)) = pair_matches {
        let pair = snapshot
            .pairs
            .iter()
            .find(|pair| pair.image_i == image_i && pair.image_j == image_j)
            .ok_or_else(|| format!("snapshot has no ordered pair ({image_i}, {image_j})"))?;
        println!("pair_match_index_i\tpair_match_index_j");
        for &(left, right) in &pair.matches {
            println!("{left}\t{right}");
        }
    }
    println!(
        "images={} verified_pairs={} accepted_correspondences={} pair_order_hash={:016x} unordered_edge_hash={:016x}",
        snapshot.image_names.len(),
        snapshot.pairs.len(),
        snapshot.accepted_match_count,
        snapshot.pair_order_hash,
        snapshot.unordered_edge_hash,
    );
    println!(
        "verifier_config_hash={:016x} verifier_config={:?}",
        snapshot.verifier_config_hash, snapshot.verifier_config,
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
