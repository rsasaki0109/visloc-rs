# Sequential SfM vs COLMAP Head-to-Head (Registry-Backed)

Generated from benchmark-registry run manifests. Phase 2 formalization of the head-to-head documented in `docs/sfm_vs_colmap_benchmark.md`: the same rectified EuRoC `MH_03_medium` 2700-frame stereo stream reconstructed by both engines and scored against the same timestamped Vicon/Leica ground truth with the same `evo_ape` tooling. visloc-rs is stereo VO + online windowed BA + loop-closure pose-graph optimization -> merged multi-view tracks -> COLMAP model export, metric scale by construction from the rectified stereo baseline. COLMAP is monocular `sequential_matcher` + incremental `mapper` (its SIFT frontend), scale-free, Sim(3)-aligned to ground truth.

| engine | wall-clock | registration rate | ATE vs GT | mean reprojection | downstream 3DGS | metric scale | run id |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| visloc stereo VO + loop SfM | 6 min | 1.000 | 0.130 m (Sim3, model) / 0.066 m (SE3, metric VO) | 2.60 px | blurry | yes | sfm-vs-colmap-visloc-MH_03_medium-20260703T120000Z |
| COLMAP mono incremental | 11.7 h | 1.000 | 2.180 m (Sim3) | 0.58 px | blurry | no | sfm-vs-colmap-colmap-MH_03_medium-20260703T120000Z |

## Headline

- **~117x faster**: visloc 6 min vs COLMAP 11.7 h (COLMAP's mapper stage alone is ~11.5 h; its incremental mapper interleaves a global bundle adjustment that grows with the registered-image count, so cost is super-linear in frame count).
- **~17-33x more accurate**: visloc 0.13 m (Sim3, COLMAP-model export) / 0.066 m (SE3, loop-closed VO trajectory, metric) vs COLMAP 2.18 m (Sim3, scale-absorbed).
- **Metric scale**: visloc recovers metric scale by construction (rectified stereo baseline); COLMAP's monocular reconstruction is scale-free and cannot recover it -- a single global Sim(3) cannot even absorb the local scale drift COLMAP accumulates over this long, low-parallax forward flight.

## Caveats

- **The stereo-vs-monocular asymmetry is the thesis, not a thumb on the scale.** The claim is narrow and turf-specific: on metric video SfM, an architecture built around a stereo VO frontend + windowed BA + loop closure dominates a from-scratch monocular incremental mapper on speed, accuracy, and scale recovery. This is **not** a claim that visloc-rs beats COLMAP on COLMAP's home turf -- unordered internet photo collections, where retrieval + multi-hypothesis incremental mapping is COLMAP's strength.
- **On COLMAP's home turf (monocular, small scene), COLMAP wins.** On the first 300 MH_03 frames reconstructed monocularly (left camera only, both engines scale-free, same Sim(3)-aligned scoring), COLMAP reaches **0.37 cm** Sim(3) ATE at 300/300 registered vs visloc's **1.64 cm** at 299/300 (`--colmap-style` incremental mapper). visloc does not yet match COLMAP on this turf.
- **The 3DGS blur is capture-geometry-limited, not a pose defect.** Both engines produce a blurry downstream 3DGS fly-through on this forward-flight sequence -- the same limited-parallax capture geometry blurs any pose source, including COLMAP's. Contrast the V2_03 orbit sequence (better capture geometry), which renders crisp (l1 ~= 0.006), against this MH-class forward flight (l1 ~= 0.24 on MH_05). The blur is a property of the capture trajectory, not of which engine estimated the poses.
- **The COLMAP arm is a prior-run reference, not reproduced this session.** COLMAP is not installed on this machine, and its documented 11.7 h wall-clock cost (single CPU, COLMAP 4.0.3, no CUDA) makes a local re-run impractical for this evidence-formalization pass. The COLMAP manifest captures the already-documented, previously-executed result from `docs/sfm_vs_colmap_benchmark.md` rather than re-executing it.

## Conclusion

On the metric-video turf visloc-rs is built for, the stereo VO + loop-closure SfM architecture wins all three axes against a from-scratch monocular COLMAP reconstruction -- speed, accuracy, and metric-scale recovery. That win is scoped honestly: it does not extend to COLMAP's unordered-photo home turf, where the small-scene monocular subset still favors COLMAP, and the downstream 3DGS blur on this sequence reflects capture geometry rather than either engine's pose quality.
