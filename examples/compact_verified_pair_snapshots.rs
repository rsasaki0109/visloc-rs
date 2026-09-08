//! Convert a v1 shard directory to shared-envelope chunks and verify every record.
use std::path::PathBuf;
use visloc_rs::verified_pair_snapshot::{read, write_shared_atomic};

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let resume = args.len() == 3 && args[2] == "--resume";
    if args.len() != 2 && !resume {
        return Err(
            "usage: compact_verified_pair_snapshots INPUT_DIRECTORY OUTPUT_DIRECTORY [--resume]"
                .into(),
        );
    }
    let source = PathBuf::from(&args[0]);
    let destination = PathBuf::from(&args[1]);
    let mut paths = std::fs::read_dir(&source)
        .map_err(|error| error.to_string())?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "vps"));
    paths.sort();
    if paths.is_empty() {
        return Err("no snapshot shards".into());
    }
    if resume {
        if !destination.is_dir() {
            return Err("resume requires an existing output directory".into());
        }
        let expected = paths
            .iter()
            .filter_map(|path| path.file_name())
            .collect::<std::collections::HashSet<_>>();
        for entry in std::fs::read_dir(&destination).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.extension().is_some_and(|extension| extension == "vps")
                && !path.file_name().is_some_and(|name| expected.contains(name))
            {
                return Err("resume output contains an unexpected snapshot".into());
            }
        }
    } else {
        std::fs::create_dir(&destination).map_err(|error| error.to_string())?;
    }
    let mut pairs = 0usize;
    let mut reused = 0usize;
    for (index, path) in paths.iter().enumerate() {
        let snapshot = read(path)?;
        let output = destination.join(path.file_name().ok_or("missing filename")?);
        if resume && output.exists() {
            reused += 1;
        } else {
            write_shared_atomic(&output, &snapshot)?;
        }
        if read(&output)? != snapshot {
            return Err(format!("round-trip mismatch: {}", path.display()));
        }
        pairs += snapshot.pairs.len();
        if (index + 1) % 250 == 0 {
            eprintln!("verified {}/{} shards", index + 1, paths.len());
        }
    }
    println!(
        "verified {} shards, {pairs} pairs, {reused} reused; shared-envelope record parity passed",
        paths.len()
    );
    Ok(())
}
