# Sequential SfM vs COLMAP (metric video, head-to-head)

The other SfM benchmarks measure visloc-rs against *itself*
([euroc_sfm_benchmark](euroc_sfm_benchmark.md)) or use COLMAP's own
reconstruction as the reference to match
([the unordered SfM benchmark](../scripts/run_colmap_sfm_benchmark.sh) reaches
~1 cm of COLMAP's poses on its example photo sets). This one is an honest
**competitor** head-to-head on the turf the project is built for — ordered
**metric video**: both engines reconstruct the *same* rectified EuRoC frames,
and both are scored against the timestamped Vicon/Leica ground truth with the
same `evo_ape` tooling (the DROID/DPVO same-tool battle pattern).

- **visloc-rs**: linear-time stereo VO + online windowed BA + loop-closure
  pose-graph optimization → merged multi-view tracks → COLMAP model
  (`stereo_vo_external_deep_files --sfm-colmap-out`). **Metric scale by
  construction** (rectified stereo baseline).
- **COLMAP**: monocular `sequential_matcher` + incremental `mapper` (its SIFT
  frontend). Scale-free, so its trajectory needs a Sim(3) (scale-fitted)
  alignment to the metric ground truth.

## Result — EuRoC MH_03_medium, full 2700 frames

ATE rmse via `evo_ape`, SE(3) (rigid, the metric-frame number) and Sim(3)
(scale-absorbed). Single CPU machine, COLMAP 4.0.3 (no CUDA).

| engine | wall-clock | registered | mean reproj | ATE Sim(3) | metric scale |
|---|---|---|---|---|---|
| COLMAP mono incremental | **11.7 h** (mapper 11.5 h) | 2700 / 2700 | 0.58 px | 2.18 m | ✗ (scale-free) |
| **visloc stereo VO + loop SfM** | **6 min** | 2700 / 2700 | 2.60 px¹ | **0.13 m** (model) | ✓ |
| └ loop-closed VO trajectory | — | — | — | **0.066 m** (SE(3), metric) | ✓ |

![MH_03 trajectories](assets/sfm_vs_colmap_mh03.png)

visloc-rs wins all three axes — **speed ≈ 117×, accuracy ≈ 17–33×, and metric
scale COLMAP's monocular path cannot recover**.

Why COLMAP loses on its own metric:

- **Speed.** COLMAP's incremental mapper interleaves a *global* bundle
  adjustment that grows with the registered-image count, so its cost is
  super-linear in frame count — 31 min reached only ~310 / 2700 frames, and the
  full run took 11.7 hours. visloc's VO frontend is linear-time and the loop
  closure is a one-shot pose-graph solve, so the whole reconstruction is
  minutes.
- **Accuracy.** COLMAP's reconstruction is locally tight (0.58 px reprojection)
  but **monocular SfM drifts in scale** over a long, low-parallax forward
  flight; a single global Sim(3) cannot absorb local scale drift, leaving a
  2.18 m residual. The figure shows it (red) collapsing inward against ground
  truth (black) mid-sequence, while visloc (blue) tracks it.

¹ The 2.60 px reprojection is a deliberate knob, not a ceiling: the COLMAP-export
bundle adjustment is capped at 5 iterations so it polishes structure without
re-deforming the loop-closed trajectory (a full 30-iteration export BA reaches
0.94 px but pulls trajectory ATE out to 0.21 m — the reprojection-vs-ATE
tension). For a downstream 3DGS/NeRF model you raise the iteration count; for the
trajectory benchmark you keep it low.

## Honest framing: stereo vs monocular

This is visloc's **stereo** pipeline against COLMAP's **monocular** one — and
that asymmetry *is* the thesis, not a thumb on the scale. The claim is narrow
and turf-specific: **on metric video SfM, an architecture built around a stereo
VO frontend + windowed BA + loop closure dominates a from-scratch monocular
incremental mapper on speed, accuracy, and scale recovery.** It is *not* a claim
that visloc beats COLMAP on COLMAP's home turf — unordered internet photo
collections, where retrieval + multi-hypothesis incremental mapping is COLMAP's
strength. On a small-scene monocular subset (300 frames, same images, both
monocular), COLMAP's full global BA wins accuracy (3.8 cm vs visloc's monocular
`sequential_sfm_demo` at 11 cm); the win reported here is specifically the
long-metric-video regime.

## Reproduce

```sh
# Needs: COLMAP (>=4.x), evo, python3, the rectified EuRoC stereo images and
# exported SuperPoint/LightGlue stereo+temporal features (reuse the
# loop-closure / euroc-sfm benchmark artifacts).
scripts/run_sfm_vs_colmap_battle.sh \
    --rect-dir /tmp/MH_03_rect --feat-dir /tmp/sp_MH_03 \
    --gt /tmp/MH_03_gt.tum --frames 2700 --out-dir target/sfm_vs_colmap
```

The runner builds both reconstructions, converts each to TUM
(`scripts/colmap_images_to_tum.py` for the COLMAP-format models — note it reads
the frame index from the image-name *basename*, since COLMAP stores resolved
symlink paths), and reports the table above.
