# Shared snapshot envelopes

`unordered_sfm_demo --shared-snapshot-envelope` opts snapshot exports (including
persistent worker shards) into a compact, lossless representation. Default
exports remain legacy v1. Snapshot readers and the streaming merger accept both.

Each directory stores immutable `envelope-<sha256>.vpe` files and pair chunks.
The envelope is the exact v1 payload prefix through verifier configuration;
the chunk contains the four shard-specific counters/hashes and encoded pairs.
The chunk header binds the envelope SHA-256, and the existing FNV checksum covers
that binding and the chunk payload. FNV is accidental-corruption detection, not
an adversarial authentication mechanism. No matches or floating-point bits are
changed. Readers concatenate the validated envelope and chunk payload logically
and reuse the existing decoder, including compact/capped mapper validation.

Publication writes and syncs owned temporary files. A hard link publishes an
envelope only if absent; existing content must match exactly. A same-directory
rename publishes the completed chunk after its envelope. A retry may leave an
unreferenced envelope or interrupted temporary file; neither is a completed
chunk. This is process-interruption safety, not a tested power-loss guarantee.
Copy the whole directory, including `.vpe` files. A chunk alone is incomplete.

To convert retained shards into a fresh directory with full record equality
checked for every output:

```sh
cargo run --locked --release --example compact_verified_pair_snapshots -- INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY
```

The converter retains one input/output shard at a time. It does not replace or
delete inputs and refuses an existing destination. The writer supports retry;
the batch converter itself does not yet resume partially completed conversions.

## Remaining work

This format removes repeated envelope storage, but currently reads and validates
the shared envelope for every chunk. Add a bounded, explicit reader/writer
envelope cache before claiming linear metadata I/O/CPU. Verify concurrent
publication, forced-interruption restart, real persistent-worker output parity,
and 100k scaling before switching the native runner default. Full E2E, trajectory
quality and whole-pipeline memory gates remain separate.
