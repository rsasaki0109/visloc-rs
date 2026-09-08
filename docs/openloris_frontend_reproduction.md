# OpenLORIS frontend reproduction boundary

The admission chain is now reproducible **from retained matching inputs**.
This is not a continuous native E2E result, and it does not fix the remaining
COLMAP trajectory-quality gate.

The native base branch has now also been regenerated from retained base features:
candidate manifest, all matching shards and the merged snapshot are byte-identical
to their references. Adaptive, targeted and dense-overlay frontend reproduction
remain open; this is not full input-image-to-model execution.

```text
base features → native-rig-runner verified snapshot ─────┐
base + supplemental features → adaptive matching ───────┤
                                                       ↓
                                             prefix supplement admission
                                                       ↓
                                             intermediate rig mapping
                                                       ↓ registration list
adaptive matching + prefix snapshot ─────────→ repair admission
                                                       ↓
targeted7 matching ──────────────────────────→ novel-pair admission
                                                       ↓
                                             atlas source mapping
```

The verified stages are:

| Stage | Complete reproduced output | Evidence JSON under benchmarks/electro |
| --- | --- | --- |
| Prefix supplements | 59,961-pair snapshot, SHA-256 `fbcd111c…` | m8-openloris-native-prefix-supplement-replay-v1.json |
| Intermediate registration | Both models' six files and retrieval-components.txt | m8-openloris-repair-registration-replay-v1.json |
| Repair admission | 173 new pairs; 60,134-pair snapshot, SHA-256 `ac93b28d…` | m8-openloris-repair19-admission-replay-v1.json |
| Targeted admission | 213 new pairs; 60,347-pair snapshot, SHA-256 `c9b30f7f…` | m8-openloris-targeted7-admission-replay-v1.json |

These stages were run separately with retained upstream inputs. File equality
supports assembling this dependency chain; it does not establish that a single
run executed it or that summed phase times equal E2E time. Intermediate mapping
is a required cost because repair selection consumes its registration result.

The raw M5 sparse7n snapshot is **not** the prefix-admission input: that hypothesis
produced 60,310 pairs and failed complete-file equality. The actual reproduced
input is `corridor1-1-m8-native-rig-runner-10k-v1/mapping/verified-merged.vps`.
Likewise, the `long128` directory's `verified-goodbase-repair19.vps` is byte-identical
to the adaptive control-prefix repair snapshot. Its directory name alone does
not require long128 matching in this chain.

Next, reproduce candidate generation and matching for adaptive, targeted7 and
the separate dense deferred overlay. Preserve their exact feature
indices and pair order. Freeze the complete executable recipe, run it as one
process tree with wall/RSS accounting, and evaluate the unchanged COLMAP gates.
The earlier 1,250-image extraction and synthetic bank-only 100k tests do not
close full10k extraction, whole-pipeline restart or full SfM100k I/O gates.

Disk cleanup on 2026-09-09 restored about 21 GiB of root free space before the
next replay. Do not silently move measured outputs to memory-backed storage or
discard retained evidence to obtain a pass.

## Native frontend replay commands

The machine-local diagnostic recipes are:

```sh
python3 scripts/replay_native_candidates.py \
  --binary /home/sasaki/datasets/openloris/corridor1-1-m8-native-frontend-binaries-v1/candidates-a7ff5ff \
  --binary-sha256 8eeee5c2f8b39de6575c2308e89cba11b96d38d66d1ae3d51ab6ce0d61d43e79 \
  --source-commit a7ff5ff47b561556ba20a0afdb40417ec9413120 \
  --output-name corridor1-1-m8-native-candidates-legacy-replay-v1
python3 scripts/replay_native_matching.py \
  --candidate-root /home/sasaki/datasets/openloris/corridor1-1-m8-native-candidates-legacy-replay-v1 \
  --binary /home/sasaki/datasets/openloris/corridor1-1-m8-native-frontend-binaries-v1/candidates-a7ff5ff \
  --binary-sha256 8eeee5c2f8b39de6575c2308e89cba11b96d38d66d1ae3d51ab6ce0d61d43e79
python3 scripts/replay_native_merge.py \
  --binary /home/sasaki/datasets/openloris/corridor1-1-m8-native-frontend-binaries-v1/merge-a7ff5ff \
  --binary-sha256 34840984753c39c23fb30506cc20631b48d69ab4319afdee719f73283b4faa16
```

They refuse existing replay directories. Candidate generation uses the retained
command-line recipe, a SHA-256-checked executable, CPU8, and
a 30-minute timeout. Matching requires a successful candidate replay, regenerates
all 2,188 legacy 32-pair candidate shards and checks their full hashes against the
recorded schedule, validates the retained feature bank, and uses CPU4. It compares
every resulting snapshot with the corresponding reference snapshot. These are
diagnostic stage replays, not a continuous E2E runner or portable demo commands.
Matching deliberately excludes snapshot merging and downstream reconstruction.

The external replay directories are
`corridor1-1-m8-native-candidates-legacy-replay-v1` and
`corridor1-1-m8-native-matching-replay-v1` under the OpenLORIS dataset root.
Each executed phase writes `report.json`, `run.log` and `time.txt`. A launched
process or a `running` report is not a pass; require terminal exit zero and
complete-file equality. Historical candidate thread settings were not recovered
from the timing file, so new candidate wall time alone is not a speedup claim.

The initial default-argument replay with `extract-3ae253a` completed but failed
candidate equality: 66,239 shared pairs and 3,761 different pairs on each side.
It remains in `corridor1-1-m8-native-candidates-replay-v1`; do not feed that run
into a claimed historical matching reproduction. Commit `67ead5e` introduced
score-descending retrieval fill, whereas `a7ff5ff` retains producer pair order.
The historical executable above was rebuilt in a clean detached worktree; its
build provenance is `m8-openloris-native-frontend-legacy-build-v1.json`.

The historical candidate replay completed successfully: 10,000 images, 70,000
pairs, full manifest SHA-256 `fca2fce1056e08966ed4d3c0a48781cf546ad5c70a6f6f86c39c365205da5716`
equal to the retained reference, wall 583.20 s, peak RSS 354,312 KiB.
Evidence: `m8-openloris-native-candidates-legacy-replay-v1.json`. Thus native
candidate generation no longer requires a retained candidate output, but still
requires the retained base feature bank.

Matching then completed in 209.94 s with peak RSS 869,812 KiB. All 2,188
regenerated candidate shards matched the frozen plan hashes; all 2,188 output
snapshots were byte-identical, with no missing or extra snapshots. The legacy
merger rebuilt from the same clean source produced 58,530 pairs / 3,177,460
accepted correspondences in 4.59 s, peak RSS 1,843,928 KiB. Full output SHA-256:
`ada96b24a29bea87cfe95853959b8915d3265b940d438b849a587dffd1401276`.
The output is in `corridor1-1-m8-native-merge-replay-v1`. Evidence is in
`m8-openloris-native-matching-replay-v1.json` and
`m8-openloris-native-merge-replay-v1.json`.

All timings cover their respective subprocess, not preparation or validation.
Do not sum them into a continuous E2E claim. Full extraction, the other frontend
branches, continuous E2E and the COLMAP quality gate remain open.
