# Shared snapshot envelopes

`unordered_sfm_demo --shared-snapshot-envelope` opts snapshot exports (including
persistent worker shards) into a compact, lossless representation. Default
exports remain legacy v1. Snapshot readers and the streaming merger accept both.

The Python native runner (`scripts/benchmark_electro.py`) also accepts
`--shared-snapshot-envelope` with `--persistent-matcher`. It forwards the flag
and records each completed chunk's `snapshot_envelope` filename/SHA-256 alongside
the chunk hash. Completion recovery uses the same binding path. Resume/index
validation and merge preflight reject missing, changed, or unrecorded envelope
dependencies. Within one validation call, each immutable envelope is hashed
once. The Rust reader separately validates payload checksums and bindings.
Defaults remain legacy; mixed legacy/shared completed shards can be read.
`stress_native_shared_restart.py` now tests the actual CLI runner with 128 real
feature files and 988 local candidate pairs (124 shards). The runner/worker
process group was SIGKILLed after eight completed shards; resume ran only the
remaining 116. All 124 chunks and the merged snapshot match uninterrupted control
bytes, prior chunks are unchanged, and all complete index entries bind their
envelopes. Evidence: `m8-native-shared-restart-v2.json`. This covers matching and
merge restart, not extraction, mapper restart, full10k or continuous E2E.

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
delete inputs and refuses an existing destination unless `--resume` is explicit.
Resume requires that directory to exist, rejects unexpected `.vps` files, and
validates every existing output against its source before accepting it. Missing
chunks are written atomically; existing corrupt chunks fail instead of being
silently replaced. Interrupted temporary files are not treated as complete chunks.

`stress_shared_snapshot_restart.py` confirmed SIGKILL during a real partial
conversion with 253 published chunks. Resume completed all 2,500 shards with full
decoded-record equality and all previously published chunk bytes unchanged.
This tests partial-batch process interruption, not a guaranteed mid-file kill,
power-loss durability, matching-worker resume, or mapper restart. Evidence:
`m8-shared-snapshot-restart-v1.json`.

`validate_snapshot_shards DIRECTORY` performs bounded structural readback using
the shared reader. It checks payload checksums, raw/inlier relationships,
compatible envelopes, image IDs, disjoint pair membership, and accepted counts.
It does not recompute declared edge/order hashes or bind external feature banks.
All 100k synthetic output chunks passed (3,125 shards, 99,999 pairs), with one
envelope load, 8.79 s and peak 20,524 KiB. 1k/10k runs also passed. This is not
full100k SfM or quality evidence; see `m8-shared-readback-scaling-v1.json`.

## Real worker parity probe

`scripts/probe_shared_match_worker.py --binary SFM_BINARY --compare-binary
COMPARE_BINARY --output NEW_DIRECTORY --shards 8` uses the retained dense
candidate plan and feature bank, selecting evenly spaced shard IDs. It preserves
the full image order and upstream candidate-index binding, but generates a subset
plan with the selected pair/shard counts. Input candidate hashes and the full
feature manifest are checked before running the worker with
`--shared-snapshot-envelope`. Output membership must match exactly, and
`compare_verified_pair_snapshots` checks full decoded record equality against
each corresponding legacy shard. Reference-only shards are intentionally allowed
by that comparison utility; the probe enforces candidate membership separately.
Use `--shards 2500` to cover the entire retained dense schedule. This is matching
parity, not extraction or continuous native E2E.

The first eight-shard probe passed: IDs 0, 357, 714, 1071, 1428, 1785, 2142,
2499, covering 256 candidate pairs with the full 10k feature bank. Exact output
membership and every decoded snapshot record match the retained legacy shards.
Worker peak RSS was 451,184 KiB. Evidence: `m8-shared-worker-probe-v1.json`.

The complete 2,500-shard/80,000-candidate worker replay also passed exact output
membership and full decoded record equality. The plan itself matches the frozen
reference byte-for-byte. Worker-only time was 594.68 s, peak 451,060 KiB.
This single run does not show a matching speedup (the earlier legacy replay was
557.35 s under a different run history). The demonstrated outcomes are parity
and shared storage reduction, not E2E or COLMAP quality. Evidence:
`m8-shared-worker-full-v1.json`.

## Remaining work

The streaming merger now uses a one-entry envelope cache per pass. It validates
on first load, before eviction, and at pass completion; caches are not retained
between passes. A change fails the merge before output publication. Standalone
reads remain uncached. The cache assumes immutable inputs within each pass and
uses the validated bytes consistently; it is not an instantaneous file watcher.
The cache now retains a parsed immutable envelope, not just serialized bytes.
An internal merge view shares that envelope by `Arc` and owns only shard-local
pairs. The existing public owned `Snapshot` API remains unchanged. The pair
decoder is shared with legacy and compact/capped reads, preserving validation.
Repeated shards with the same cached envelope skip redundant envelope comparisons
using pointer identity; a new envelope still receives full compatibility checks.
Memory is bounded to a constant number of envelopes, not one per shard. Alternating
different envelopes can still cause misses; normal same-bank shards share one.

All 2,500 real dense shards merge to the exact retained legacy output SHA with
both shared and legacy input formats. Single warm-cache measurements were
10.54 s / 16,644 KiB for shared input; these are not repeated speed evidence or
an uncached-shared A/B. See `m8-dense-shared-merge-cache-v1.json`.
Eight concurrent writers also pass the unit test with exactly one envelope and
eight valid chunks. Missing/changed envelopes and cache eviction are tested.

`SharedSnapshotWriter` now prepares an envelope once and accepts only shard-local
`SharedPairChunk` records afterward. The opt-in persistent worker uses this API
and requires a single output directory. It validates the immutable envelope at
preparation and at batch completion; image-index bounds and pair encoding are
checked on each write. Standalone `write_shared_atomic` remains a convenience
wrapper that prepares and validates for each call.

The synthetic writer stress spans 1k/10k/100k image envelopes with adjacent-image
pairs copied from one seed payload. Output sizes are 20,012,094 / 200,292,475 /
2,003,096,487 bytes. The 100k write completed in 8.86 s, peak 19,324 KiB, with
3,125 chunks. This is writer-only scaling, not matching, mapping, or full output
readback. The synthetic manifest/pair hashes do not represent real input banks.
See `m8-shared-writer-scaling-v1.json` and `stress_shared_snapshot_writer`.

The shared-input merger now avoids repeated envelope decoding/owned Snapshot
construction. Standalone owned reads and legacy v1 shards still pay their own
envelope costs. This does not establish that the entire native pipeline is
N²-free: candidate planning, restart orchestration, and other consumers need
their own scaling evidence. Verify
forced-interruption restart, real persistent-worker output parity,
and 100k scaling before switching the native runner default. Full E2E, trajectory
quality and whole-pipeline memory gates remain separate.
