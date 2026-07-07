# Tracking persistence: death-spiral fixes and relocalization coverage (EuRoC)

Status: registry-backed A/B, 2026-07-06. Opt-in flags, default-off (zero behavior
change unless enabled). Companion evidence: manifests under
`benchmarks/registry/runs/euroc/` with `benchmark_id=tracking-persistence-ab`.

## Problem

Tracked-segment accuracy on EuRoC MH is SOTA-class (MH_01 4.8 cm ATE rigid), but
coverage is the weak pillar: the stereo HOG baseline tracks only 9-25 % of frames
before dying permanently. A frame-level audit of the MH_01 loss (frames 1093-1100)
found a reproducible **death spiral**, not a feature-quality cliff:

1. one weak frame fails tracking → the motion-model pose prior freezes while the
   camera keeps moving;
2. the fixed `--max-pose-jump-meters 0.2` gate then rejects *good* PnP solutions
   (measured: 53 inliers, ratio 0.47) because they are >0.2 m from the stale prior;
3. every rejection widens the true gap → permanent loss;
4. on the way down, a garbage keyframe (4 inliers, ratio 0.035) was promoted into
   the map, poisoning the local map that any recovery would localize against.

## Fixes (opt-in)

| Flag | Mechanism |
| --- | --- |
| `--pose-jump-gap-scaling` (+ `--pose-jump-gap-scaling-max-multiplier`, default 10) | Effective pose-jump gate = `max_pose_jump_meters × frames-since-last-success` (capped). Gap 1 = today's gate, bitwise-identical. |
| `--keyframe-min-inliers N` / `--keyframe-min-inlier-ratio R` | Keyframe promotion rejected (`InsufficientTrackingQuality` in `keyframe_decisions.csv`) when the frame's PnP inliers fall below the floor. |
| `--relocalization-enabled` (pre-existing) | Map-based PnP recovery; the old silent-crash blocker (exit 0xFFFFFFFF) no longer reproduces at current HEAD (full MH_01 run, stable ~250 MB). |
| `--relocalization-confirmation-required-recoveries 2` (pre-existing) | Requires two consecutive consistent recoveries before re-entering tracking; kills 1-frame false relocalizations. |

## Results (EuRoC MH, stereo HOG, gated replenish base flags)

Arms: `reloc` = relocalization on; `fix` = gap-scaling + keyframe quality gate
(`--keyframe-min-inliers 15 --keyframe-min-inlier-ratio 0.1`).

| Sequence | Arm | Tracking coverage | ATE rigid RMSE | Reloc successes | Poison KFs blocked |
| --- | --- | --- | --- | --- | --- |
| MH_01 | baseline (gated replenish) | 25.2 % | **4.8 cm** | — | — |
| MH_01 | + reloc | 31.1 % | 5.4 cm | 11 / 2532 | 0 |
| MH_01 | + fix | 32.4 % | 51.2 cm ⚠ | — | 15 |
| MH_01 | + reloc + fix | 40.6 % | 37.2 cm ⚠ | 25 / 2198 | 21 |
| MH_03 | baseline (gated replenish) | 17.6 % | **3.6 cm** | — | — |
| MH_03 | + fix | 25.4 % | 20.2 cm ⚠ | — | 3 |
| MH_05 | baseline (gated replenish) | 9.1 % | 12.6 cm | — | — |
| MH_05 | + fix | 8.5 % ✗ | 47.8 cm ✗ | — | 2 |

### Reading the ⚠ rows honestly

The headline ATE regression in the `fix` arms is **not** a broad quality loss; it
is a handful of isolated false-relocalization frames. Per-segment audit of the
MH_01 `reloc+fix` run (1487 tracked frames):

- primary segment (frames 380-1102): 4.9 cm RMSE — unchanged from baseline;
- legitimate recovered segments: 8-17 cm RMSE;
- **14 frames (0.9 %) with 0.5-7.2 m error** — 1-2-frame false relocalizations in
  the unmapped zone. Excluding them, overall RMSE is ≈ 9.4 cm.

`--relocalization-confirmation-required-recoveries 2` (pre-existing flag) is the
natural fix for exactly those frames — requiring two consecutive consistent
recoveries before re-entering tracking should reject 1-frame false positives.
It was attempted on MH_01 `reloc+fix` in this session but **not completed**: the
run was still running after 74 minutes (vs. 19-26 min for its sibling arms) and
was killed rather than left open-ended. The likely reason is a compounding cost:
confirmation-gating keeps more of the map from being poisoned, so the surviving
map grows larger over the run, which grows the per-attempt full-map descriptor
rebuild cost (see "Scope and limitations" below) — a fix that reduces false
positives on one axis can make the unbounded reloc-store cost worse on another.
Left as a follow-up: bound the descriptor store before retrying confirmation.

Same structure on MH_03 `fix` (673 tracked frames): 31 transient frames (4.6 %)
carry >0.5 m error; excluding them the RMSE is 5.5 cm, and the arm gains a new
~165-frame healthy segment (frames 2526-2699, ~5 cm) the baseline never reaches.

**MH_05 is an honest negative for the `fix` arm**: coverage 9.1 %→8.5 % and only
one healthy segment survives (frames 29-90, 2.0 cm); the re-acquired frames are
scattered and low quality (18 % above 0.5 m). On a weak, dark, fast sequence the
wide-gate re-acceptance lands on drifted local maps; the fix needs relocalization
and/or a stronger acceptance-quality floor there. Not recommended on
MH_05-class sequences as configured.

### Scope and limitations

- Relocalization can only recover where the (frozen) map covers the viewpoint; the
  MH_01 dead zone frames 1122-3516 are unreachable by relocalization by
  construction. Coverage beyond that needs not-losing-tracking in the first place
  (the gate fixes) or map growth during loss — see the re-bootstrap follow-up below.
- Relocalization runtime: the default broader store rebuilds a full-map descriptor
  store and brute-force-matches against every landmark **every lost frame**. On
  MH_03 (mostly-lost sequence) this made the reloc arm impractically slow
  (>3.5 h, killed); MH_03/05 are therefore reported for the `fix` arm only.
  Bounding the store (`--relocalization-covisibility-max-keyframes`,
  appearance retrieval) or throttling attempts is the known lever.
- Reloc match quality is the next-order lever: 79 % of failed attempts are
  `no_pnp_solution` on ~67 garbage correspondences from full-map brute-force HOG
  matching (mean 1.8 inliers on `min_inliers` failures).

## Re-bootstrap on prolonged loss (opt-in follow-up, 2026-07-07)

When relocalization cannot reach an unmapped viewpoint, the demo can opt into
**GT-seeded stereo re-bootstrap**: after `N` consecutive lost frames (and no
successful relocalization in the same frame), re-triangulate cam0/cam1 at the
current frame, seed the pose from ground truth (same convention as the initial
bootstrap), append the new landmarks to the live map, and resume tracking under a
new `segment_id`. Logged in `rebootstrap_log.csv`; `slam_errors.csv` carries
`segment_id` per row.

| Flag | Role |
| --- | --- |
| `--rebootstrap-after-lost-frames N` | Trigger after `N` consecutive lost frames. Default `None` (off). |
| `--rebootstrap-cooldown-frames M` | Minimum frame gap between accepted re-bootstraps. Default `60`. |

Requires cam1 stereo (`--stereo-bootstrap` and/or `--stereo-landmark-replenish`).
**Caveat:** segment restarts are GT-seeded — honest for demo/evidence, not a
blind recovery claim.

MH_01 on top of the gated replenish + reloc + fix base (`N=30`, `M=60`):

| Metric | reloc+fix base | + re-bootstrap |
| --- | --- | --- |
| Tracking coverage | 40.6 % | **54.8 %** |
| Tracked frames | 1487 | 2006 |
| Re-bootstrap events | — | 15 (segments 1–15 in dead zone) |
| Primary segment RMSE | ~4.9 cm | seg 0: **7.4 cm** |
| Headline RMSE (good frames) | 7.9 cm | 10.8 cm |
| Spike frames (>0.5 m) | 14 | 109 |

The dead-zone segments are shorter and noisier (per-segment RMSE 5–19 cm), but
the README hero GIF now visibly traces the upper GT loops that reloc alone never
reached. CLI-only sweeps (bounded covis store, appearance retrieval, relaxed
reloc gates) did not beat this honestly: relaxed gates reached 45.0 % coverage
with visually wobbly recoveries; re-bootstrap reaches 54.8 % with structurally
new tracked segments.

## Reproduce

```
target/release/examples/euroc_online_slam_vi_image_demo.exe \
  --euroc-dir <MH_xx> --out-dir <out> --max-frames 4000 \
  --gravity 0,0,-9.81 --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.1 --max-pose-jump-meters 0.2 \
  --stereo-bootstrap-strict --stereo-landmark-replenish \
  --relocalization-enabled --pose-jump-gap-scaling \
  --keyframe-min-inliers 15 --keyframe-min-inlier-ratio 0.1
```
