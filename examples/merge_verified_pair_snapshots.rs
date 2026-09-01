//! Merge hash-checked verified-pair snapshot shards.
//!
//! Each input snapshot contains the complete image/feature/camera/config
//! envelope and a disjoint subset of pairs.  The library codec validates the
//! per-file checksum while reading; the merge additionally rejects envelope
//! mismatches and overlapping pairs, then recomputes the merged stream hashes.
//! This small utility keeps the large-scale runner independent of a Python
//! implementation of the binary snapshot format.

use std::path::{Path, PathBuf};

use visloc_rs::verified_pair_snapshot::{merge_owned, read, write_atomic};

fn usage() -> ! {
    eprintln!(
        "usage: merge_verified_pair_snapshots --output PATH [--snapshot PATH ...]\n\
         or:    merge_verified_pair_snapshots --output PATH --snapshot-list PATH"
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

fn read_snapshot_list(path: &Path) -> Result<Vec<PathBuf>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read snapshot list {}: {error}", path.display()))?;
    let mut paths = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.split_whitespace().count() != 1 {
            return Err(format!(
                "snapshot list {}:{} must contain one path per line",
                path.display(),
                line_number + 1
            ));
        }
        paths.push(PathBuf::from(line));
    }
    Ok(paths)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut output = None;
    let mut snapshots = Vec::new();
    let mut snapshot_list = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => output = Some(next_path(&args, &mut index, "--output")?),
            "--snapshot" => snapshots.push(next_path(&args, &mut index, "--snapshot")?),
            "--snapshot-list" => {
                snapshot_list = Some(next_path(&args, &mut index, "--snapshot-list")?)
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    let output = output.ok_or_else(|| "--output is required".to_owned())?;
    if snapshot_list.is_some() && !snapshots.is_empty() {
        return Err("--snapshot and --snapshot-list are mutually exclusive".into());
    }
    if let Some(list) = snapshot_list {
        snapshots = read_snapshot_list(&list)?;
    }
    if snapshots.is_empty() {
        return Err("at least one --snapshot is required".into());
    }
    let decoded = snapshots
        .iter()
        .map(|path| {
            read(path).map_err(|error| format!("invalid snapshot {}: {error}", path.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let merged = merge_owned(decoded)?;
    write_atomic(&output, &merged)?;
    println!(
        "merged {} snapshot shard(s): {} pairs, {} accepted correspondences -> {}",
        snapshots.len(),
        merged.pairs.len(),
        merged.accepted_match_count,
        output.display(),
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
