# Replay storage cleanup

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
