# OpenLORIS frontend reproduction boundary

The admission chain is now reproducible **from retained matching inputs**.
This is not a continuous native E2E result, and it does not fix the remaining
COLMAP trajectory-quality gate.

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

Next, reproduce candidate generation and matching for native-runner, adaptive,
targeted7 and the separate dense deferred overlay. Preserve their exact feature
indices and pair order. Freeze the complete executable recipe, run it as one
process tree with wall/RSS accounting, and evaluate the unchanged COLMAP gates.
The earlier 1,250-image extraction and synthetic bank-only 100k tests do not
close full10k extraction, whole-pipeline restart or full SfM100k I/O gates.

Storage currently constrains additional large runs. Do not silently move measured
outputs to memory-backed storage or discard retained evidence to obtain a pass.
