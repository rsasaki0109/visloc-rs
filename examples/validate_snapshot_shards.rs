//! Structural readback only; does not validate external banks or SfM quality.
use std::path::PathBuf;
use visloc_rs::verified_pair_snapshot::validate_files;

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 1 {
        return Err("usage: validate_snapshot_shards DIRECTORY".into());
    }
    let mut paths = std::fs::read_dir(PathBuf::from(&args[0]))
        .map_err(|error| error.to_string())?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    paths.retain(|path| path.extension().is_some_and(|extension| extension == "vps"));
    paths.sort();
    println!("PASS: {:?}", validate_files(&paths)?);
    Ok(())
}
