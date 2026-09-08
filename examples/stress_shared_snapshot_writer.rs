//! Synthetic image-envelope/pair-chunk I/O scaling; not a reconstruction benchmark.
use std::path::PathBuf;
use visloc_rs::verified_pair_snapshot::{read, SharedPairChunk, SharedSnapshotWriter};

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(
            "usage: stress_shared_snapshot_writer SEED_SNAPSHOT IMAGE_COUNT NEW_OUTPUT_DIRECTORY"
                .into(),
        );
    }
    let count = args[1]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    if !(2..=100_000).contains(&count) {
        return Err("image count must be 2..100000".into());
    }
    let mut snapshot = read(&PathBuf::from(&args[0]))?;
    let seed = snapshot
        .pairs
        .first()
        .ok_or("seed needs a nonempty pair")?
        .clone();
    let features = *snapshot
        .feature_counts
        .iter()
        .max()
        .ok_or("seed needs images")?;
    snapshot.image_names = (0..count)
        .map(|index| format!("synthetic_{index:06}.png"))
        .collect();
    snapshot.feature_counts = vec![features; count];
    // Synthetic metadata: these are not claimed to bind any real feature bank.
    snapshot.image_manifest_hash = 0;
    snapshot.feature_manifest_hash = 0;
    snapshot.pairs.clear();
    let output = PathBuf::from(&args[2]);
    std::fs::create_dir(&output).map_err(|error| error.to_string())?;
    let writer = SharedSnapshotWriter::new(&output, &snapshot)?;
    let mut chunks = 0;
    for start in (0..count - 1).step_by(32) {
        let pairs = (start..(start + 32).min(count - 1))
            .map(|index| {
                let mut pair = seed.clone();
                pair.image_i = index as u64;
                pair.image_j = index as u64 + 1;
                pair
            })
            .collect::<Vec<_>>();
        let path = output.join(format!("chunk-{chunks:06}.vps"));
        writer.write_chunk(
            &path,
            SharedPairChunk {
                pair_order_hash: 0,
                unordered_edge_hash: 0,
                accepted_match_count: pairs.iter().map(|pair| pair.matches.len() as u64).sum(),
                pairs: &pairs,
            },
        )?;
        chunks += 1;
    }
    writer.finish()?;
    let bytes = std::fs::read_dir(&output)
        .map_err(|error| error.to_string())?
        .map(|entry| {
            entry
                .and_then(|entry| entry.metadata())
                .map(|metadata| metadata.len())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?
        .into_iter()
        .sum::<u64>();
    println!("synthetic images={count} pairs={} chunks={chunks} directory_bytes={bytes}; writer I/O only, no SfM or quality claim", count - 1);
    Ok(())
}
