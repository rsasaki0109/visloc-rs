# visloc-rs Development Handoff

Short handoff for whoever picks up `visloc-rs` next, human or agent. It says
where the project stands and where to read further. The forward plan is
[`docs/next_development_plan.md`](docs/next_development_plan.md).

The long session-by-session log that used to live here (2026-05 to 2026-09:
KITTI drift tuning, the ORB-SLAM3 battle plan, the back-end workstream,
Phase-20..27 close-outs, and many recorded negative results) is archived
verbatim in
[`docs/archive/plan_history_2026-05_to_09.md`](docs/archive/plan_history_2026-05_to_09.md).
Read it before retrying a lever: many levers have already been measured and
rejected there.

## Current status (2026-10-08)

- Latest release: **0.2.1** (camera-distortion correctness fixes on top of
  0.2.0). See [`CHANGELOG.md`](CHANGELOG.md).
- MSRV 1.88, `unsafe` forbidden, no mandatory ML or GPU runtime in the
  default build.
- All 16 workspace crates package for crates.io (`cargo package --workspace`);
  nothing has been published yet. See [`docs/publishing.md`](docs/publishing.md).

| Pillar | State | Evidence |
| --- | --- | --- |
| Unordered photo SfM | Benchmarked. ETH3D 9,996/10,008 registered; Electro 1,200 3.46× faster than COLMAP CPU at 25% lower error | [`docs/sfm_benchmarks.md`](docs/sfm_benchmarks.md) |
| Video SfM (GPU) | Benchmarked. Faster than COLMAP CUDA on 8/8 EuRoC, more accurate on 4/8 | [`docs/euroc_gpu_sfm_vs_colmap.md`](docs/euroc_gpu_sfm_vs_colmap.md) |
| Stereo-inertial VI-SLAM (Basalt port + online mapper) | Benchmarked. Real time on 11/11 EuRoC, beats ORB-SLAM3 on 9/11 | [`docs/vi_slam_benchmarks.md`](docs/vi_slam_benchmarks.md) |
| Photos to 3DGS + mesh | Benchmarked. Faster than brush on 5/5 scenes | [`docs/rust_3dgs_plan.md`](docs/rust_3dgs_plan.md) |
| Map-based localization | Benchmarked. OpenLORIS rig 98.96% localized; RNE simulated house 38 ms/frame | [`docs/map_matching_localization_plan.md`](docs/map_matching_localization_plan.md) |
| Stereo / RGB-D VO | Benchmarked. KITTI 00 1.23 m, 09 2.07 m; TUM fr1_xyz 1.4 cm | [`docs/kitti_multiseq_benchmark.md`](docs/kitti_multiseq_benchmark.md) |
| Monocular + IMU VIO | Experimental, no measured result | [`docs/next_development_plan.md`](docs/next_development_plan.md) |
| Camera + GNSS | Example-level only | [`docs/gnss_demo.md`](docs/gnss_demo.md) |
| Multi-camera rig SfM | Experimental. OpenLORIS 10k registered; RMSE 0.3890 m vs COLMAP 0.3843 m | [`docs/sfm_benchmarks.md`](docs/sfm_benchmarks.md) |

## Threads that are closed (do not reopen without new evidence)

- **MH_04 / MH_05 vs ORB-SLAM3.** Recorded as negative results in
  2026-09: IMU-derived global-BA factors, joint VI global BA, the
  epipolar-error lever (flips MH_05 but loses V2_03), `optical_flow_levels=4`,
  IMU-seeded KLT and `--match-top-k`. The joint VI-BA code is kept opt-in.
  The remaining gap is VIO tracking robustness on fast, blurred motion.
- **EuRoC SfM accuracy gap vs COLMAP CUDA (V1_02, V1_03).** Bisected to
  frontend localization precision and marginal RANSAC, not a mapper bug. GPU
  SIFT subpixel refinement, SuperPoint-seeded refinement and the SP bridge did
  not close it.
- **KITTI seq02 loop closure.** True loops are never proposed by VLAD; this
  needs an offline vocabulary or a learned global descriptor first.

## Quality gate

`scripts/check.sh` runs everything CI runs (fmt, MSRV, feature matrix,
clippy `-D warnings`, tests, Python tests, benchmark-registry checks, docs
links, release metadata, examples, demo output checks, rustdoc, package
check). Every behavior change updates README / CHANGELOG / the relevant
`docs/` page.

## Guardrails

- Learned, GPU and accelerator paths stay opt-in behind features or
  file-backed adapters.
- New behavior is opt-in until it is measured. Default outputs stay
  bit-identical, and the existing parity and hash tests enforce this.
- Public claims are scoped to the benchmark registry
  (`benchmarks/registry/`). Negative results are recorded, not deleted.

## Key files to read first

- [`README.md`](README.md): headline results and entry points.
- [`docs/next_development_plan.md`](docs/next_development_plan.md): what to
  do next.
- [`docs/interfaces.md`](docs/interfaces.md) and
  [`docs/api_stability.md`](docs/api_stability.md): public API surface.
- [`docs/basalt_online_mapper_design.md`](docs/basalt_online_mapper_design.md)
  and
  [`docs/vi_slam_global_consistency_plan.md`](docs/vi_slam_global_consistency_plan.md):
  VI-SLAM design.
- [`docs/decisions.md`](docs/decisions.md): why things are the way they are.
