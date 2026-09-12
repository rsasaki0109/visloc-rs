# Replay storage cleanup

## Synthetic shared-writer archive, 2026-09-09

Archived the completed synthetic writer/readback fixture
`/home/sasaki/datasets/openloris/corridor1-1-m8-shared-writer-stress-v1` to
`/home/sasaki/datasets/openloris/synthetic-stress-archive.ZTe3sv/shared-writer-stress-v1.tar.gz`
(approximately 27 MiB). Archive SHA-256:
`343f64faf884dace5d691a56fe91ed03cd8b3b25eee6ca4068a2f89b59a4a8a4`.
GNU tar comparison exited zero, comparing archived contents and metadata with
the original tree. An independent membership check found exactly 3,481 files,
no duplicate archive file entries, and only regular-file/directory members.
After these checks, removed only the expanded synthetic fixture directory.
Its binaries, time logs and synthetic chunks are retained in the archive.
No real image, feature bank or reconstruction was removed.

To restore, verify the archive SHA, sufficient free space, and absence of the
original directory, then extract with `tar -xzf ARCHIVE -C /home/sasaki/datasets/openloris`.
The archive includes its top-level directory. This is storage recovery, not
an SfM memory or speed improvement.

User authorized external-disk relocation and disk cleanup. The destination is
`/media/sasaki/aiueo1/visloc-replay-retired-20260909` (directory identifier).

- `corridor1-1-m8-adaptive-bank-publication-v1`: moved from the OpenLORIS
  dataset directory; original path now a symlink. All 10,004 files verified
  against the pre-move aggregate SHA-256 inventory:
  `7eab135e2d47d8a9fbf72199b13435db234af1f5dbc918d2ecbd6dcdf239459d`.
- `corridor1-1-m8-extraction-shard0-replay-v1`: same relocation and symlink
  policy. All 2,502 files verified; inventory:
  `66de1b1714377cb1796523dd67199ab814b0f9676534aab8f5e127452b72d972`.
- `synthetic-sift-bank-100k-v1`: archived as
  `synthetic-sift-bank-100k-v1.tar.gz` at the destination (about 2.8 MiB).
  All 200,000 file contents and exact membership were compared with the
  archive before removing the original expanded synthetic fixture.

The first two inventories hash sorted relative paths, a tab, the SHA-256 of
each file, and a newline. Logs and evidence remain available. Source images,
original feature banks and original reconstruction outputs were not removed.

Root filesystem free space increased from about 308 MiB at operation start
to 3.5 GiB. External disk free space is about 1.9 GiB. This is storage cleanup,
not pipeline memory optimization. Future benchmarks must record the relocated
input/output filesystem; do not compare new I/O timings as unchanged conditions.

To restore the synthetic fixture, first verify sufficient free space and that
the original directory is absent, then extract the archive under
`/home/sasaki/datasets/openloris`. The archive contains its top-level directory.

## Broader cleanup, 2026-09-09

Removed these explicitly inspected, regenerable caches:

- `/home/sasaki/.cache/pip`: approximately 2.8 GiB of download cache.
- `/home/sasaki/.npm/_cacache`: approximately 2.5 GiB of package cache.
- `/tmp/visloc-rs-target-camerarig/release/deps`: approximately 332 MiB
  of Cargo build dependencies; no Cargo/rustc process was observed running.

These caches have no backup; they can be downloaded or built again. Installed
packages, source code, saved example binaries, datasets and experiment results
were retained. Root free space increased from 3.5 to 9.0 GiB (rounded `df -h`
values); external free space remains 1.9 GiB. This does not measure RAM savings.

Larger candidates require a separate retention decision: the approximately
58 GiB EuRoC experiment directory `/home/sasaki/euroc_mh03_official_20260830`
and other projects' evidence/output directories. They were inspected for size
only and were not deleted. `/tmp` contains experiment evidence, not just caches,
so it must not be cleared wholesale.

### Authorized EuRoC duplicate cleanup

After the user authorized cleanup of the EuRoC directory, removed 187
`ranges/*.part` download fragments totaling 12,537,235,986 bytes (11.68 GiB).
Before any removal, `scripts/cleanup_euroc_download_parts.py` verified the
retained `machine_hall.zip` against its recorded SHA-256
`5ed7d07903f8d19b6c8808e2ae8a0872b281f6e34ef5497023b8ac58c3de0f6f`
and compared every fragment byte with its named inclusive byte range in that
archive. The cleanup exited successfully. All `.part.headers`, download logs,
archives, original images, rectified images, manifests and run results remain.
Fragments can be reconstructed by reading the corresponding inclusive byte
ranges from the retained archive; the header filenames retain the range names.

The approximately 30 GiB `features_sp2048_cpu_2700` directory was retained:
`docs/nonregression_20260830.md` identifies it as the authoritative frozen
feature bank used for baseline/current comparison. Its exact contents should
not be discarded merely because feature extraction can be run again.

Four focused tests pass: audit-only preservation, verified removal, corrupt
fragment rejection and corrupt archive rejection. The script defaults to
audit-only and requires `--remove-verified-parts` for removal.
