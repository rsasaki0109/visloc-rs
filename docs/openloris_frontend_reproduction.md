# OpenLORIS frontend reproduction boundary

The admission chain is now reproducible **from retained matching inputs**.
This is not a continuous native E2E result, and it does not fix the remaining
COLMAP trajectory-quality gate.

The native base branch has now also been regenerated from retained base features:
candidate manifest, all matching shards and the merged snapshot are byte-identical
to their references. Adaptive matching/merge and targeted7 candidate/matching/merge
have also been reproduced. Dense candidate/matching/merge and the registration
manifest consumed by target selection are now reproduced too. This is still not
full input-image-to-model execution or a continuous E2E measurement.

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
repair admission → deferred registration → missing-frame targets
missing-frame targets → targeted7 candidates → matching → novel-pair admission
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

Next, reproduce both full feature extractions and assemble the complete executable
recipe. Preserve exact feature indices and pair order, run it as one
process tree with wall/RSS accounting, and evaluate the unchanged COLMAP gates.
The earlier 1,250-image extraction and synthetic bank-only 100k tests do not
close full10k extraction, whole-pipeline restart or full SfM100k I/O gates.

The source-mapping portion can now be compiled without executing shell snippets:

```sh
python3 scripts/build_native_source_recipe.py
```

This emits 21 argv templates and 23 atlas node bindings from the successful
source reproduction evidence. Binary/source identities, evidence hashes,
timeouts and expected model hashes are retained. Node model hashes must agree
with their producing source evidence. The source4200 debug environment is
preserved explicitly; apply environment removals before overrides. All five
input path flags and the output directory become named placeholders; an
unconverted absolute argv path is rejected. This is a recipe compiler, not a
runner: it does not validate a live binary, extract features, execute mapping,
enforce RSS, or establish input provenance. The future executor must bind inputs
to outputs of the same cold run and verify them before launch. Expected model
hashes are post-execution comparison data, never reconstruction inputs.

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

Candidate replay now also accepts `--features-dir EXPLICIT_BANK --output
NEW_CANDIDATE_DIRECTORY` for connection to an enclosing run. `--output` and
`--output-name` are mutually exclusive. Both the default and explicit feature
bank must validate against the frozen variant's manifest before output creation;
the resolved bank and manifest hash are recorded. The final candidate comparison
is unchanged. Binding a path is not proof that extraction ran: the enclosing
executor must separately establish that provenance. This adapter does not yet
connect extraction, all matching variants, admission and atlas execution.

Matching replay accepts `--features-dir BANK --output NEW_MATCHING_DIRECTORY`
alongside `--candidate-root`; merge replay accepts `--matching-root DIRECTORY
--output NEW_MERGE_DIRECTORY`. Existing destinations remain rejected. Matching
checks exact feature membership as well as manifest bytes, and merge retains
its shard membership/content and final-byte checks. These options allow the
diagnostic stages to exchange fresh outputs, but do not add extraction, an
orchestrator, restart, or shared-envelope exports to the historical replay path.
The final bounded-I/O runner must use the separately validated shared worker
path, not promote legacy repeated-envelope artifacts as N²-free evidence.

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

## Adaptive and targeted matching replay

Adaptive and native worker plans differ only in the feature-manifest hash.
Adaptive therefore consumes the independently reproduced native candidates;
there is no additional adaptive ANN candidate run in this frozen recipe.
The replay uses the newly published adaptive bank (full manifest validated),
now on external ext4 storage after authorized disk cleanup.

```sh
python3 scripts/replay_native_matching.py --variant adaptive \
  --candidate-root /home/sasaki/datasets/openloris/corridor1-1-m8-native-candidates-legacy-replay-v1 \
  --binary /home/sasaki/datasets/openloris/corridor1-1-m8-native-frontend-binaries-v1/candidates-a7ff5ff \
  --binary-sha256 8eeee5c2f8b39de6575c2308e89cba11b96d38d66d1ae3d51ab6ce0d61d43e79
python3 scripts/replay_native_merge.py --variant adaptive \
  --binary /home/sasaki/datasets/openloris/corridor1-1-m8-native-frontend-binaries-v1/merge-a7ff5ff \
  --binary-sha256 34840984753c39c23fb30506cc20631b48d69ab4319afdee719f73283b4faa16
```

All 2,188 candidate shards were regenerated and hash-checked. Original adaptive
match shards are no longer present: the matching result remains explicitly
`complete-awaiting-merge-comparison`, not a per-shard reproduction pass. The
merger checks new-shard membership and the post-matching content inventory before
consuming them. Its complete output matches the retained 61,286-pair snapshot
exactly (SHA-256 `852c43c3df905ca150e44fbc65c35ed5a451507399019649b3900d06503ccf19`).
See `m8-openloris-adaptive-frontend-replay-v1.json`.

Targeted7 candidates were reproduced with:

```sh
python3 scripts/build_targeted_rig_candidates.py \
  --rig-manifest /home/sasaki/datasets/openloris/corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt \
  --target-frames 1999,3264,3266,3267,4493,4494,4495 --max-frame-gap 256 \
  --output-directory /home/sasaki/datasets/openloris/corridor1-1-m8-targeted7-candidates-replay-v1
```

Use the matching command above with `--variant targeted7` and
`--candidate-root /home/sasaki/datasets/openloris/corridor1-1-m8-targeted7-candidates-replay-v1`,
then the merge command with `--variant targeted7`. All 14,319 candidate pairs,
448 match shards and the complete 265-pair merged snapshot match the references.
See `m8-openloris-targeted7-frontend-replay-v1.json`.

The initial targeted replay used frozen target IDs. Its preceding mapping command
was subsequently recovered from a completed same-thread execution event dated
2026-09-02T23:34:18.633Z. It uses the repair19 snapshot and
`--deferred-registration-pair-prefix 59961`, not a dense-overlay input. See
`m8-openloris-targeted-selection-recovered-command-v1.json`.

```sh
python3 scripts/replay_targeted_selection.py
python3 scripts/build_targeted_rig_candidates.py \
  --rig-manifest /home/sasaki/datasets/openloris/corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt \
  --selection-json /home/sasaki/datasets/openloris/corridor1-1-m8-targeted-selection-replay-v1/selection.json \
  --max-frame-gap 256 \
  --output-directory /home/sasaki/datasets/openloris/corridor1-1-m8-targeted7-derived-candidates-replay-v1
```

The new mapper run exited zero in 160.02 s with peak RSS 976,540 KiB. Its full
registration manifest is byte-identical, and the seven unregistered frames are
derived without GT. The resulting candidate file is also byte-identical. Three
other intermediate model files differ (main images, main points, components.tsv);
this is **selection-boundary reproduction, not whole-model byte reproduction**.
The downstream selector consumes only registration membership, not these poses
or points. Do not omit the required mapping cost. Evidence:
`m8-openloris-targeted-selection-replay-v1.json`.

`replay_targeted_selection.py` now accepts `--features-dir`, `--snapshot` and
`--output` so this required mapper can consume an enclosing run's repair output.
It retains the frozen repair snapshot hash and exact adaptive feature validation;
existing output directories are rejected. These options have not yet been used
in a continuous cold run. The three admission commands can separately be emitted
as non-executing templates with `python3 scripts/build_native_admission_recipe.py`.
The repair template explicitly requires the intermediate mapper's generated
registration components. Binary/evidence/output hashes are retained, but live
input provenance and resource enforcement are responsibilities of the executor.

## Dense overlay frontend

```sh
python3 scripts/replay_native_candidates.py --variant dense
python3 scripts/replay_native_matching.py --variant dense \
  --candidate-root /home/sasaki/datasets/openloris/corridor1-1-m8-dense-candidates-replay-v1
python3 scripts/replay_native_merge.py --variant dense \
  --binary /home/sasaki/datasets/openloris/corridor1-1-m8-native-frontend-binaries-v1/merge-e136ae6 \
  --binary-sha256 f726c044973a21725296011bf08a49daed2b510813bce449e5f1c6fc8b95cd36
```

The dense branch uses score-descending fill, topK128, minimum frame gap128,
budget80,000, and LSH8/auto9/6. Its candidate manifest is byte-identical.
All 2,500 v2 candidate shards were regenerated and hash-verified. The recorded
streaming matcher recipe (min30) produced byte-identical snapshots, and the
current `merge_files_atomic` implementation produced a byte-identical final
snapshot with peak RSS18,188 KiB. This last number is merge-only, not pipeline
RSS. See `m8-openloris-dense-candidates-replay-v1.json` and
`m8-openloris-dense-frontend-replay-v1.json`.

## Avoid duplicate supplemental extraction

Every one of the 692 selected supplemental feature files is byte-identical to
the corresponding file in the dense bank (231,393,842 bytes compared). Passing
the dense directory directly as `merge_sift_supplements.py --supplement` and
validating against the completed adaptive bank reproduced all 10,000 expected
file hashes; zero files were written. Thus a continuous recipe can compute the
dense bank once and use its selected subset for adaptive construction. Evidence:
`m8-openloris-dense-supplement-reuse-audit-v1.json`. This is content/recipe
equivalence, not a measured E2E speedup or whole-pipeline memory reduction.

The base bank is a different extraction recipe (SIFT256, one orientation,
contrast0.02, no compatible-detector flags), so it cannot be replaced by the
dense bank. Both full extractions still need fresh input-image execution.

All timings cover their respective subprocess, not preparation or validation.
Do not sum them into a continuous E2E claim. External-bank I/O is not unchanged
from historical runs. Full extraction, continuous E2E and the COLMAP quality gate
remain open. Legacy v1 diagnostic shard reproduction is not proof of N²-free
artifact growth: audit shared-envelope storage in the final runner before 100k
claims.

## Extraction and storage preflights

For the future continuous executor, `scripts/measure_native_scope.py` wraps a
command in an **already configured dedicated** cgroup v2 scope:

```sh
systemd-run --user --scope --property=MemoryMax=2G --property=MemorySwapMax=0 \
  python3 scripts/measure_native_scope.py --report NEW_REPORT.json \
  --timeout 3600 -- COMMAND ARGUMENTS
```

It rejects different limits or initial unrelated scope members, samples aggregate
RSS every 50 ms (including the monitor), and records cgroup `memory.peak` and
before/after `memory.events`. Charged cache/kernel memory makes the cgroup value
different from RSS; shared mappings can be counted repeatedly in summed RSS.
Short peaks between samples may be missed. The cgroup is the hard limit, not the
sampler. A timeout kills the owned command process group; leftover descendants
or new OOM events prevent success. Commands must not escape the scope/session.
A small 32 MiB allocation probe exited zero; a sleeping child hit its deadline
and returned failure. These are launcher checks, not SfM resource evidence.
The runner still needs the complete extraction-to-atlas DAG and restart protocol.

For long work, use the detached service launcher instead of keeping the launch
terminal alive:

```sh
python3 scripts/launch_native_measurement.py --unit visloc-UNIQUE-LOWERCASE-NAME \
  --output NEW_MEASUREMENT_DIRECTORY --timeout 7200 -- COMMAND ARGUMENTS
```

Replace the example unit with an all-lowercase `visloc-` name. The launcher
copies and hashes the monitor, configures the same memory limits, disables
systemd argument environment expansion, persists stdout/stderr in `service.log`,
and returns after service creation. It does not freeze the measured command's
scripts/binaries/inputs; the executor must do that separately. `launch.json`
with `launched-not-completed` is not a pass. Check `systemctl --user show UNIT`
for `Result=success`, `ExecMainStatus=0`, and `SubState=exited`, plus a terminal
passing `measurement.json`. The service has a runtime deadline and control-group
cleanup; its retained service result is not a guarantee that cgroup counters
remain available after exit. Persisted measurement data is still required.

A short detached 32 MiB allocation probe saved its terminal ledger after the
launch command returned. A 3,700-second low-load probe is in progress as
`visloc-durable-long-probe-v1.service`, with artifacts under
`/home/sasaki/datasets/openloris/m8-durable-long-probe-v1`. Long-duration success
is not established until that service and its report are terminal. This addresses
measurement durability, not extraction performance or the still-failing quality
gate; the cause of the earlier missing final ledger remains unproven.

The 2026-09-09 post-full-runner preflight found 8,914,128,896 bytes free on
the dataset filesystem. Fresh base, dense (including loci) and adaptive banks
alone require 8,332,098,526 logical bytes, leaving only 582,030,370 bytes before
matching outputs, models, allocation overhead and temporary files. Do not launch
an all-banks-retained cold run under that observation. See
`m8-native-e2e-storage-preflight-v1.json`. This is not a complete peak-storage
estimate: the executable DAG must establish last consumers and resume dependencies
before releasing any newly generated intermediate. Retained evidence stays
protected; using retained feature banks would change a cold extraction-inclusive
measurement into a resumed/replay measurement. Recheck free space at launch.

`merge_sift_supplements.py --hardlink-unselected` is an opt-in storage primitive
for newly generated immutable banks on the same filesystem. Unselected files
share their base inode; selected files still use independent atomic publication.
Default behavior remains independent copies. Cross-device linking fails without
silently copying, and source symlinks are rejected for new links. Never edit either
bank in place: this mode is not independent backup storage. Resume compares bytes
against current inputs, so an externally pinned base manifest must be validated
to detect mutations that would affect both links. No input file is deleted.
Unit tests cover copy/link inventory equality, selected-file independence, resume
and cross-device failure. The real10k probe now passes: 9,308 shared files and
692 independent selected files exactly match the retained adaptive bank; base
bytes are unchanged and resume reuses all10k. Newly allocated regular file blocks
total 257,925,120 bytes, excluding directories/metadata/logs. Evidence:
`m8-linked-adaptive-bank-v2.json`. The initial v1 correctly rejected source
symlinks; v2 checks the tier's exact membership and resolves every entry to the
common real base directory before using it. Peak-DAG disk accounting remains
required before a measured continuous run; this is not extraction or E2E proof.

### Snapshot storage audit

`scripts/audit_snapshot_envelopes.py` measured the 2,500 dense matching snapshots:
1,294,392,088 total bytes, of which 777,135,000 are shared envelope bytes.
All envelopes have the same SHA-256; 776,824,146 bytes are duplicate copies.
The image-dependent portion alone is 775,000,000 bytes. Fixed-pair-count shards
therefore retain quadratic image metadata when shard count grows with image
count. Streaming merge bounds merge memory, not this storage/I/O cost.
Evidence: `m8-dense-snapshot-envelope-audit-v1.json`. This structural audit does
not validate pair payload checksums; full dense replay parity is separate.
Next: shared immutable envelope plus identity-bound pair chunks, legacy v1 read
compatibility, and restart/round-trip tests before runner adoption.

### Raw image probe

The base extraction preflight also passed on four raw images spread across both
cameras and the sequence (`cam1_000000`, `cam1_006666`, `cam2_003333`,
`cam2_009999`). `scripts/probe_openloris_base_extraction.py` replays the recorded
base SIFT recipe with the frozen image-capable extractor; all four feature files
(584 keypoints) match the retained base bank byte-for-byte. Evidence:
`m8-openloris-base-extraction-probe-v1.json`. This does not establish full10k
extraction parity or E2E performance.

The same script now accepts `--all-images --workers 6`: the full frozen 10k
image set is split into disjoint balanced partitions, with one Rayon thread per
worker. Each worker writes distinct feature filenames and its own log/timing;
the harness compares the complete combined feature membership and hashes after
all workers exit. Four-image/two-worker parity passed. Full10k six-worker replay
now passes every feature hash and exact membership, with all six workers exiting
zero, in `corridor1-1-m8-full-base-extraction-v2`. However, the outer scope report
remained `running` after the processes/cgroup disappeared: final aggregate RSS
and cgroup counters were not recovered. This closes full base feature parity,
not the resource ledger. See `m8-full-base-extraction-v2.json`; validate durable
detached service measurement before the next long measured run.
The earlier single-worker v1 was deliberately interrupted, not completed or
reused as a speed baseline. See `m8-full-base-extraction-transition-v1.json`.
Use the dedicated 2 GiB scope wrapper for aggregate measurement/enforcement;
the extraction script alone does not impose that limit. Worker RSS peaks are
not summed. GNU timeout uses `--foreground` so workers remain in the command
process group for wrapper timeout cleanup. This remains base extraction only,
not dense extraction or continuous E2E.

## Empty vocabulary safety

Streamed candidate export now rejects a missing/empty appearance vocabulary
before candidate construction. It no longer calls `all_pairs(N)` in that case:
the candidate budget would only have been applied after the quadratic allocation.
The error explicitly says `refusing exhaustive fallback`; no candidate manifest
is published. Nonempty descriptor data keeps the same retrieval functions and
borrowed global descriptors, with the batch/streamed equality test preserved.

`scripts/stress_empty_streamed_candidates.py` exercised the real CLI with 100,000
empty feature files and calibrated image entries under a 2 GiB address-space cap
and a timeout. It rejected normally (exit 1 and the required diagnostic), emitted
no candidate manifest, took 1.00 s and peaked at 102,892 KiB RSS. Evidence:
`m8-empty-streamed-candidates-100k-v1.json`. This is an invalid-input safety gate,
not a 100k reconstruction, nonempty 100k retrieval benchmark, whole-pipeline
memory bound or quality result.
