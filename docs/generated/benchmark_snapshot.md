# Benchmark Snapshot

Generated from `benchmarks/registry/readme_claims_v1.json`. This is the public headline table; registered run evidence, including exploratory and negative runs, is rendered separately in `docs/generated/registered_runs.md`.

| Benchmark | Result |
| --- | ---: |
| **KITTI multi-sequence published-baseline comparison** | one uniform full-stack config over 00/02/05/06/07/09; narrow published-baseline wins on seq00 (**1.23 m vs ORB-SLAM2 1.3 m**) and seq09 (**2.07 m vs ORB-SLAM2 3.2 m**), with seq00/05/06 in the OV2SLAM-RT accuracy band. This is not a leaderboard or ORB-SLAM3 claim; the run also records real-world frontend failure-mode fixes. |
| **EuRoC MH_03 / MH_05 full pipeline** | stereo visual loop-closure + BA on MH_03 / MH_05: **0.057 m / 0.072 m** ATE. The claim matrix marks ORB-SLAM3 comparisons as behind (**~2.4x / ~1.4x**), OV2SLAM as near, and VINS-Fusion stereo as a stereo-only win; this is not a tight-VIO claim. |
| **TUM RGB-D fr1_xyz / fr1_desk** | indoor handheld via **virtual stereo** (depth as a synthetic right image, zero backend changes): **0.014 m / 0.026 m** ATE, compared against published ORB-SLAM2 RGB-D ranges in the claim matrix; loop closure is a **6x** lever on the revisit-heavy desk. |
| **KITTI seq00 loop closure** | open VO 36.3 m -> **2.6 m** Sim(3) ATE (**14x**), 35 verified loops |
| **EuRoC MH_03 SfM reconstruction** | merged multi-view tracks + global BA: mean reprojection **4.08 px -> 1.04 px**, 179 k tracks, COLMAP export for 3DGS / MVS |
| **Sequential SfM vs COLMAP (metric video)** | same 2700-frame EuRoC flight, same evo scoring: visloc stereo VO + loop SfM **6 min, 0.13 m** (trajectory 0.066 m, metric) vs COLMAP mono incremental **11.7 h, 2.18 m** (scale-free) - **~117x faster, ~17-33x more accurate, metric scale**. (Stereo-vs-mono: the win is the metric-video regime, not COLMAP's unordered-photo home turf.) |
| **Unordered SfM (real photo collections)** | Orderless monocular photos -> VLAD view graph -> incremental reconstruction (robust multi-seed init, P3P register, scale-gauge-fixed BA, iterative track filter), vs **COLMAP's own model** with an independent SuperPoint frontend: **COLMAP South Building** (128 photos) **128/128 reg, 1.09 cm**; **Gerrard Hall** (100 photos, 5616x3744 OPENCV) **98/100, 0.68 cm** (3/100 single-seed) - both **0.1 % of extent**. EuRoC V2_03 orbit **31/31, 1.08 cm** |
| COLMAP South Building localization | deep frontend gives **+37% to +98%** more verified inliers as the viewpoint gap grows |
| **Multi-session lifelong mapping (7-Scenes)** | a map bootstrapped from one session is **grown across later visits by relocalization alone** (no GT poses): learned EigenPlaces retrieval integrates **126 vs 120** later-session keyframes loose / **106 vs 93** strict-gate vs bag-of-features, the gap widening as the gate tightens, and its merges are **~0.09 m vs ~0.14 m** accurate - a cleaner lifelong map (test median **0.059 m vs 0.144 m**) |
| Pose-graph optimization (SE-Sync `.g2o`) | **ties GTSAM 4.x LM** on `parking`/`sphere`/`cubicle`; **beats** it on `torus3D` (2.4e4 vs 6.0e4) and `rim` (8.3e4 vs 6.1e5) |
| Outlier-robust PGO (GNC) | `sphere2500` + 30 wrong loops: L2 **89x** baseline, GNC **1.0x** (30/30 rejected) |
