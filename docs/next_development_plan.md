# Next Development Plan (forward-looking, as of 2026-07-02)

This is a focused, prioritized plan for continuing `visloc-rs`. It complements the
long historical log in [`PLAN.md`](../PLAN.md) and the staged
[`docs/roadmap.md`](roadmap.md); read those for background. This file is the
*short* answer to "what next".

## Where we are

- Mature pure-Rust visual-localization / VO-SLAM / SfM foundation. All three
  pillars (map-reuse localization, VO/SLAM building blocks, SfM) run with
  registry-backed public-data measurements.
- Latest committed milestone: **Sequential SfM self-calibration** — SfM from raw
  images with joint-intrinsics BA and self-calibrated radial distortion
  (PRs #30–#34).
- Large **uncommitted branch** (~52 files, +7.8k lines, 2026-06-18→20):
  public-release boundary cleanup, **benchmark registry v1 + claim matrix**,
  **covisibility local BA**, **tracked-landmark-drop keyframe policy**,
  **adaptive stereo depth gate**, seq02 loop-verifier A/B tooling, CI release
  gates. This work is essentially complete and stabilizing — it is not yet on
  `main`.

## Known open threads (measured, not speculative)

| Id | Thread | Status |
|----|--------|--------|
| A | Covisibility local BA regresses MH_05 (tracking 0.565→0.220) while winning MH_01/MH_03 | opt-in only; cannot be defaulted |
| B | Tracked-landmark-drop keyframe policy | A/B evidence-gathering; default off |
| C | seq02 true loops are never proposed by VLAD | unsolved (needs offline vocab / learned global descriptor) |
| D | Tight VI coupling for V103 / V203 / V2_03 | Phase 2 not started; vision-only ceiling confirmed |

## Plan (priority order)

### Phase 0 — Land the uncommitted branch (highest EV, ~hours)
The +7.8k-line branch is finished, valuable work sitting off `main`. Getting it
committed with green CI is the single highest-EV move.
- Pass `scripts/check.sh` (fmt + clippy + test + doc + registry + feature-matrix).
- Commit in the reviewable P0/P1 groupings from `docs/release_change_sets.md`.
- Push and confirm GitHub CI is green.
- **Gate:** `git status --short` clean; CI green.

### Phase 1 — Kill the MH_05 covisibility-BA regression (top technical item)
This decides whether online local BA can become a default path. Root cause is
already measured: BA firing too early / too often on MH_05 plus no-local-landmark
selection failures.
- Levers: scene-scale-adaptive cadence (same shape as the min-depth fix),
  window-selection quality gate, always-on write-back quality gate.
- **Gate:** covisibility local BA beats the disabled baseline on MH_01/MH_03/MH_05
  simultaneously before it is defaulted. Until then it stays honest opt-in.

### Phase 2 — Close the SfM-vs-COLMAP head-to-head (high public value)
With self-calibration landed, finish SfM as the README's fourth pillar.
- Metric-video regime (EuRoC/KITTI, GT available): a registry-backed 5-metric
  table — wall-clock / registration rate / ATE-vs-GT / reprojection / downstream
  3DGS quality.
- One-command reproduction: raw images → self-calibrated SfM → COLMAP export →
  3DGS. Promote to README once registry-backed.
- **Honest caveat to keep:** SfM has no standardized published per-sequence table,
  so the claim form is same-tool head-to-head, not published-number-beating.

### Phase 3 — Real-time deep frontend polish (medium)
In-process SP/LG via ONNX CUDA already runs. Harden the "single-binary,
real-time deep stereo SLAM, pure Rust" claim into a registry-backed end-to-end
wall-clock number that feeds both the SfM and VI comparisons.

### Phase 4 — v1.0 API stabilization (long-term)
The `docs/api_stability.md` allowlist exists. Freeze the prelude / trait
boundaries, write a migration guide, and move 0.1 → 0.x → 1.0.

### Deferred (only on explicit request)
- **Tight VI coupling (Phase 2 of the ORB-SLAM3 battle plan):** the only path to
  the V-room / V2_03 cells, but OpenVINS-class frontend surgery (blackout
  bridging + reacquisition + velocity/bias co-estimation). Do not start until the
  user asks to push V2_03. Keep the honest vision-only-ceiling framing in
  `docs/euroc_loop_closure_benchmark.md`.
- **seq02 loop closure:** blocked on an offline vocabulary or a learned global
  descriptor. Prove candidate-stage recall (`scripts/eval_loop_retrieval_recall.py`)
  before touching verifier/optimizer settings.

## Guardrails (unchanged)
- No mandatory OpenCV / ONNX / PyTorch / CUDA in default crates; learned/GPU paths
  stay opt-in behind features or file-backed adapters.
- Do not claim full SLAM, full loop closure, or leaderboard results; keep public
  wording scoped to the registry and claim matrix.
- Every behavior change updates README / progress / roadmap / interfaces /
  decisions / CHANGELOG as applicable, then `scripts/check.sh`.
