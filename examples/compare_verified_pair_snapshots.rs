//! Compare every snapshot record in a candidate directory with its reference.
use std::path::PathBuf;
use visloc_rs::verified_pair_snapshot::read;

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err(
            "usage: compare_verified_pair_snapshots CANDIDATE_DIRECTORY REFERENCE_DIRECTORY".into(),
        );
    }
    let candidate = PathBuf::from(&args[0]);
    let reference = PathBuf::from(&args[1]);
    let mut paths = std::fs::read_dir(candidate)
        .map_err(|error| error.to_string())?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "vps"));
    paths.sort();
    if paths.is_empty() {
        return Err("no candidate snapshots".into());
    }
    for path in &paths {
        let reference = reference.join(path.file_name().ok_or("missing filename")?);
        if read(path)? != read(&reference)? {
            return Err(format!("snapshot record mismatch: {}", path.display()));
        }
    }
    println!("PASS: {} candidate snapshots match all reference records (reference may contain additional shards)", paths.len());
    Ok(())
}
