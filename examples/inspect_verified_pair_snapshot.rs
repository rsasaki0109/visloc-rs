//! Inspect a verified-pair snapshot without rerunning matching or mapping.
//!
//! The TSV output is intentionally simple so benchmark scripts can compute
//! verified-edge recall against a diagnostic control while the binary codec
//! remains implemented and validated in Rust.

use std::fmt::Write as _;
use std::path::PathBuf;

use visloc_rs::verified_pair_snapshot::read;

fn usage() -> ! {
    eprintln!("usage: inspect_verified_pair_snapshot --snapshot PATH [--pairs-tsv PATH]");
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
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--snapshot" => snapshot_path = Some(next_path(&args, &mut index, "--snapshot")?),
            "--pairs-tsv" => pairs_path = Some(next_path(&args, &mut index, "--pairs-tsv")?),
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    let snapshot_path = snapshot_path.ok_or_else(|| "--snapshot is required".to_owned())?;
    let snapshot = read(&snapshot_path)
        .map_err(|error| format!("invalid snapshot {}: {error}", snapshot_path.display()))?;
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
    println!(
        "images={} verified_pairs={} accepted_correspondences={} pair_order_hash={:016x} unordered_edge_hash={:016x}",
        snapshot.image_names.len(),
        snapshot.pairs.len(),
        snapshot.accepted_match_count,
        snapshot.pair_order_hash,
        snapshot.unordered_edge_hash,
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
