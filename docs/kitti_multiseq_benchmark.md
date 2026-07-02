# KITTI Multi-Sequence Benchmark — vs Published Stereo-SLAM ATE

The [seq00 loop-closure benchmark](kitti_loop_closure_benchmark.md) measures
one sequence. This benchmark runs the **same full stack with one uniform
configuration** over every loop-closure sequence of the KITTI odometry
training split that has published stereo-SLAM ATE to compare against
(00/02/05/06/07/09), and puts the numbers next to the two C++ systems that
actually publish per-sequence KITTI ATE: **ORB-SLAM2** (the canonical
non-real-time numbers from its paper, Table I) and **OV2SLAM** (real-time,
Table V). Almost nobody else publishes KITTI ATE — Stereo DSO and SOFT2
report only relative metrics, DROID-SLAM never evaluated KITTI, and deep
monocular systems are an order of magnitude off in metric terms
(DPV-SLAM++ seq00: 8.30 m Sim(3)).

## Result (stride 1, SuperPoint 2048 + LightGlue, Umeyama ATE RMSE)

| seq | length | visloc-rs SE(3) | visloc-rs Sim(3) | loops | ORB-SLAM2 | OV2SLAM (RT) |
| --- | -----: | --------------: | ---------------: | ----: | --------: | -----------: |
| 00  | 3724 m | **1.23 m**      | 0.97 m           |    34 | 1.3 m     | 1.17 m |
| 02  | 5067 m | 12.66 m †       | 12.66 m          |     1 | 5.7 m     | 6.24 m |
| 05  | 2206 m | 1.62 m          | 1.38 m           |   165 | 0.8 m     | 1.44 m |
| 06  | 1233 m | **1.42 m**      | 1.25 m           |    77 | 0.8 m     | 1.27 m |
| 07  | 695 m  | 2.33 m          | 2.14 m           |    66 | 0.5 m     | 0.37 m |
| 09  | 1705 m | **2.07 m**      | 1.65 m           |     8 | 3.2 m     | 1.59 m |

- **Beats ORB-SLAM2's published ATE on 00 and 09**, and is within 5–15 % of
  OV2SLAM's real-time numbers on 00/05/06 — with a pure-Rust stack and no
  offline-trained vocabulary.
- ORB-SLAM2's column is its *non-real-time* configuration; in OV2SLAM's
  Table V the real-time ORB-SLAM2 reads 10.74 m on seq00 with frequent
  failures. The comparison column here is the strongest published number.
- † seq02 is the honest failure in the default policy: raw appearance
  candidates include true revisits, but the default loop verifier accepts only
  one weakly useful edge, so the trajectory remains near open-drift accuracy.
  External LightGlue loop matches plus guarded confidence-weighted PnP improve
  this sequence in an opt-in run, but seq00 non-regression currently blocks
  promoting that policy to the uniform benchmark configuration.

Reproduce one sequence (downloads + exports + runs + evaluates):

```sh
python3 scripts/fetch_kitti_seq00_images.py --sequence 05 --stride 1 \
    --max-frames 99999 --cameras image_0,image_1 --also-fetch-poses \
    --out-dir ~/datasets/kitti_seq05_full
scripts/run_kitti_multiseq_benchmark.sh --sequence 05 \
    --data-root ~/datasets/kitti_seq05_full \
    --capture-run-registry
```

`--capture-run-registry` writes `full/evaluation.json` plus a
`kitti-multiseq` run manifest with ATE, loop count, command/config, trajectory,
VO log, and diagnostic CSV artifacts. Retrieval-recall manifests remain a
separate opt-in (`--capture-retrieval-recall`) because they measure the
candidate stage rather than the final trajectory.

## What this benchmark caught: three real-world failure modes

Extending from one sequence to six immediately surfaced three distinct
robustness failures, each now measured, diagnosed to a root cause, and fixed
(or documented). This is the benchmark's real value: seq00 alone exercises
none of them.

### 1. seq07 — a crossing truck captures the PnP consensus

At frames 634–641 a truck crosses directly in front of the camera and fills
most of the frame. The PnP RANSAC consensus locks onto the *moving* truck
(inlier ratio collapses to 0.10–0.25) and injects ~12° of false rotation
over five pairs — the dominant error of the whole sequence.

Two mechanisms have to cooperate to fix it, and each alone is **neutral**:

- The frontend already has weak-consensus rescue clamps
  (rotation-spike / rotation-vector / translation-direction), but their
  sustained-motion arming gate defaults to a recent median translation of
  1.5 m/frame — highway speed. The truck event happens at 0.7 m/frame.
  `--rescue-min-median-translation 0.5` arms them for urban driving.
- Armed rescues alone do not help (19.75 → 20.35 m open ATE): the online BA
  window re-imposes the rejected motion through the *same contaminated
  temporal matches*. `--ba-exclude-rescued-pairs`
  (`OnlineStereoVoBaConfig::exclude_rescued_pair_matches`) blanks a rescued
  pair's matches during BA track building, so tracks split at the
  contaminated pair and the clamped odometry carries the trajectory through
  the event.

Together: open VO 19.75 → 7.52 m; full stack 6.77 → 2.33 m.

### 2. seq09 — the motion-scale rescue freezes the translation

The motion-scale rescue (built for genuine PnP scale collapse) had a
weak-consensus bound of `max_pnp_inlier_ratio: 1.05` — a ratio is always
< 1.05, so **every** pose counted as weak — plus a lower translation-ratio
bound of 0.97 that flags any ≥3 %/frame deceleration as "collapsed". Rescued
magnitudes feed the same history the rescue compares against, so one
sustained fast-then-decelerating stretch locks the translation magnitude at
the stale history median *forever*: on seq09 the estimate froze at exactly
1.619 m/pair for ~1300 pairs while the raw (discarded) PnP translation
tracked ground truth the whole time. The urban sequences never trip it only
because the rescue arms above a 1.5 m/frame median.

`--motion-scale-rescue-max-inlier-ratio 0.45` restricts the rescue to
genuinely weak consensus, like every other rescue clamp. Full stack:
40.6 → **2.07 m** (~20×), now ahead of ORB-SLAM2's published 3.2 m.

Both fixes are measured **bit-identical** on KITTI seq00 and EuRoC MH_03
(the gates never arm there) — they only ever engage on the failure they
target.

### 3. seq02 — candidate recall exists; acceptance policy is the open problem

seq02 has genuine loops (the final ~470 frames re-traverse two earlier zones).
The first diagnosis blamed the sequence-trained VLAD vocabulary, but the
registry-backed recall evaluator changed that conclusion: on the file-backed
CPU/512 run, raw appearance candidates reach recall@1/5/20 = 0.7515 over
eligible true-revisit queries. Retrieval is therefore not the only bottleneck.

The current default still accepts only one loop edge and leaves the sequence at
12.66 m ATE. Replaying external Python LightGlue matches for 400 loop
candidates showed a sharp split: 383/400 pass a 2D-2D essential-matrix
diagnostic, while PnP accepts only 1/400 without confidence-weighted sampling.
Filtering PnP down to essential inliers stayed at 1/400 and slightly regressed
ATE (12.6603 m -> 12.6906 m).

Guarded confidence-weighted PnP is the useful diagnostic so far. With external
LightGlue loop matches and `--loop-pnp-confidence-weights`, seq02 improves to
5.8106 m SE(3) / 5.6977 m Sim(3), accepting two loop edges
(`1015->4261`, `1944->4528`). The guard only enables weights when enough
finite, non-uniform confidences are present and otherwise falls back to uniform
sampling.

That policy is **not** the default. On seq00 at 2000 frames, where accepted
loops are already present, the built-in brute-force loop matcher verifies 117
loops at 1.2064 m SE(3) / 1.1299 m Sim(3), while external LightGlue loop
matches regress to 1.2416 m / 1.1747 m unweighted and 1.2419 m / 1.1711 m
with guarded confidence weights. The external loop-match path is therefore an
opt-in per-sequence diagnostic until it passes a broader non-regression matrix.
The representative registry manifests are:
`kitti-multiseq-seq02-external-lightglue-confweights-guarded-20260619`
(positive exploratory seq02),
`kitti-multiseq-seq00-2000-baseline-bf-20260619` (seq00 baseline),
`kitti-multiseq-seq00-2000-external-lightglue-unweighted-20260619`, and
`kitti-multiseq-seq00-2000-external-lightglue-confweights-guarded-20260619`
(negative seq00 non-regression evidence).

Use `scripts/eval_loop_retrieval_recall.py` to keep future seq02 work honest.
It labels true revisits from KITTI poses, applies the same temporal/path-length
gates as the loop stage, and reports recall@K/MRR over the candidate CSV. The
multiseq runner can capture this evidence alongside the trajectory artifacts
and write a benchmark-registry manifest:

```sh
scripts/run_kitti_multiseq_benchmark.sh --sequence 02 \
  --data-root ~/datasets/kitti_seq02_full \
  --capture-retrieval-recall \
  --retrieval-dnf-if-recall-at 20=0.01
```

This writes `full/loop_candidates.csv`, `full/retrieval_recall/`, and a
`kitti_loop_retrieval_recall_v1` manifest under
`benchmarks/registry/runs/kitti/`. For standalone experiments, run the
evaluator directly on raw appearance candidates when available; running it on
post-verification candidate files is still useful, but should be labelled as
post-geometry recall:

```sh
python scripts/eval_loop_retrieval_recall.py \
  --candidates target/kitti_seq02_full/loop_candidates.csv \
  --poses ~/datasets/kitti_seq02_full/poses_02.txt \
  --input-kind raw_appearance_candidates \
  --distance-threshold-m 10 \
  --min-temporal-gap 50 \
  --min-path-length-m 5 \
  --ks 1 5 20 \
  --out-json target/kitti_seq02_retrieval/retrieval_recall.json
```

This keeps the next seq02 iteration honest: a retriever change should improve
true-revisit recall@K and false-top1 rate, while a verifier change should be
judged by accepted-loop precision, ATE, and seq00/05/09 non-regression before
it is allowed into the uniform benchmark configuration.

## The BA init-residual gate: a second dynamic-object channel

seq05 exposed a *fourth* finding, fixed by an existing knob: while stopped
at a junction (ground-truth motion 2–3 mm/frame), the *final* trajectory
accelerated to a 5.8 m jump. The frontend PnP was fine (strong static
consensus, no rescue needed) — the corruption entered through **BA track
building, which has no RANSAC**: a vehicle crossing the stopped car's view
produces high-confidence LightGlue tracks that bundle adjustment must
reconcile, and the optimizer slides the poses instead.

`--ba-max-init-residual 10` gates each track by its initial reprojection
residual against the (RANSAC-validated) frontend poses — a slow crossing
vehicle projects ~14 px/frame of parallax and gets filtered; static tracks
survive. Measured per sequence (full stack, SE(3)):

| run | 00 | 05 | 06 | 07 | 09 | MH_03 | MH_05 |
| --- | -- | -- | -- | -- | -- | ----- | ----- |
| no gate | 1.23 | 1.62 | **1.42** | 2.33 | **2.07** | 0.0582 | 0.0799 |
| gate 10 px | **1.03** | **1.39** | 2.32 | 2.16 | 2.53 | **0.0569** | **0.0720** |
| gate 20 px | 1.04 | **1.39** | 1.77 | **2.15** | 2.44 | — | — |

The gate removes the seq05 jump and helps five of seven runs — including
**both EuRoC flights** (MH_03 0.0582 → 0.0569, MH_05 0.0799 → 0.0720,
SE(3) vs ORB-SLAM3's 0.024/0.052) — but hurts 06/09 at every strength
tried: the same long tracks it trims are what anchor the scale there (the
Sim(3) column barely moves while SE(3) degrades). The KITTI table above
therefore reports the un-gated uniform configuration; the gate is an
explicit, documented trade-off, not a default.

## Configuration

One configuration for all six sequences (the same stack as the
[EuRoC benchmark](euroc_loop_closure_benchmark.md), with KITTI's loop gap):

```
--relative-pose-mode pnp
--min-stereo-confidence 0.5 --min-temporal-confidence 0.5
--online-ba --online-ba-window 30 --online-ba-trigger-every 10
--online-ba-history 20
--rescue-min-median-translation 0.5 --ba-exclude-rescued-pairs
--motion-scale-rescue-max-inlier-ratio 0.45
--loop-closure --loop-two-view-ba --loop-edge-information
--loop-min-frame-gap 50 --loop-min-path-length 5 --loop-min-similarity 0.2
--loop-vocab-k 64 --loop-max-candidates-per-frame 3 --loop-max-verifications 400
```

The external loop-match A/B flags (`--loop-matches-dir`,
`--loop-pnp-essential-inliers`, `--loop-pnp-confidence-weights`) are excluded
from this default configuration. Results using them must be labelled
exploratory or per-sequence opt-in unless they pass the same multi-sequence
non-regression bar as the table above.

Published comparison numbers: ORB-SLAM2 (arXiv:1610.06475, Table I) and
OV2SLAM (arXiv:2102.04060, Table V); both papers evaluate translation ATE
RMSE against the official KITTI poses, metric (SE(3)-style) alignment.
