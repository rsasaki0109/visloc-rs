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

The streaming merger now uses a one-entry envelope cache per pass. It validates
on first load, before eviction, and at pass completion; caches are not retained
between passes. A change fails the merge before output publication. Standalone
reads remain uncached. The cache assumes immutable inputs within each pass and
uses the validated bytes consistently; it is not an instantaneous file watcher.
Memory is bounded to one serialized envelope, not one per shard. Alternating
different envelopes can still cause misses; normal same-bank shards share one.

All 2,500 real dense shards merge to the exact retained legacy output SHA with
both shared and legacy input formats. Single warm-cache measurements were
10.54 s / 16,644 KiB for shared input; these are not repeated speed evidence or
an uncached-shared A/B. See `m8-dense-shared-merge-cache-v1.json`.
Eight concurrent writers also pass the unit test with exactly one envelope and
eight valid chunks. Missing/changed envelopes and cache eviction are tested.

This does not eliminate repeated envelope decoding/owned Snapshot construction,
nor writer-side envelope serialization/validation. Add prepared writer/borrowed
envelope interfaces before claiming linear metadata CPU. Verify
forced-interruption restart, real persistent-worker output parity,
and 100k scaling before switching the native runner default. Full E2E, trajectory
quality and whole-pipeline memory gates remain separate.
