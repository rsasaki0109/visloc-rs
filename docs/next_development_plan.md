# Next Development Plan (as of 2026-10-08)

This is the short answer to "what next". Current state and the list of
closed threads are in [`PLAN.md`](../PLAN.md). The previous version of this
plan (2026-07-02) is archived at
[`archive/next_development_plan_2026-07.md`](archive/next_development_plan_2026-07.md).
All of its phases have since landed or been closed.

Scope rule for this round: **no new GPU work.** GPU paths (wgpu SfM, 3DGS,
DPVO CUDA) stay as they are; nothing here depends on them.

## Track 1: Correctness and release hygiene

| Item | Status | Gate |
| --- | --- | --- |
| 0.2.1 patch release (distortion fixes) | Done: version bumped, CHANGELOG cut | Tag `v0.2.1` after main CI is green |
| crates.io readiness | Done: all 16 crates package and verify-build from their packaged sources; root package trimmed from ~83 MB to 0.87 MiB compressed with an `include` allowlist | `cargo publish --workspace` (needs a crates.io token) |
| BA distortion models | Done: pose/structure BA linearises every lens model (OPENCV, FULL_OPENCV, the fisheye models, FOV, Double Sphere); opt-in p1/p2 self-calibration. Open: rig observations still project with a pinhole; matrix-free / GPU backends fall back for distorted cameras | Follow-up: lens model in rig BA |

## Track 2: Sensor coverage (the empty cells in the README sensor table)

| Item | Status | Gate |
| --- | --- | --- |
| Monocular + IMU VIO on the Basalt port | Done (experimental): `basalt_euroc_vio_demo --mono`; synthetic scale error < 0.3%, 5–9 mm RMS; stereo path unchanged ([notes](mono_inertial_vio.md)) | EuRoC ATE (needs dataset access); static-start drift; mono in the online mapper |
| GNSS fusion | Done (synthetic): joint VO + GNSS pose graph with estimated alignment, lever arm, GNC outlier rejection, dropout bridging; ATE 5.25 m → 0.90 m ([notes](gnss_fusion.md)) | Real-data run (e.g. KITTI raw OXTS); geodetic-to-ENU conversion; IMU in the graph |

## Track 3: Usability

| Item | Status | Gate |
| --- | --- | --- |
| Python bindings (pyo3 + maturin) | Done: `bindings/python` (Camera, Pose, COLMAP I/O, PnP localization, ATE/RPE), 42 pytest tests, CI job | Publish wheels to PyPI |
| ROS 2 nodes | Done: `ros2/visloc-ros2` VIO and localization nodes over pure-Rust DDS, tested end to end over RTPS, CI job | Test against a real ROS 2 install (Fast DDS / Cyclone); attach the online mapper to the VIO node |

## Track 4: Maintainability

| Item | Status | Gate |
| --- | --- | --- |
| Planning docs | Done: `PLAN.md` (233 KB) and the itemized 0.2.0 CHANGELOG (~4,800 lines) moved to [`archive/`](archive/); short handoff and changelog in their place | — |
| Split very large source files | `examples/unordered_sfm_demo.rs` (22.5k lines), `pipelines/basalt/src/vio/aom.rs` (21.5k), `pipelines/slam/src/incremental_sfm.rs` (18.5k), `pipelines/slam/src/bundle.rs` (17k) | Pure moves into submodules; no behavior change; full test suite and parity/hash tests unchanged |

## Needs real data (not runnable from a dataset-less environment)

These stay open, but each needs EuRoC / OpenLORIS access to measure:

- **Real-time margin.** The slowest EuRoC sequence runs at RTF 1.06×. LM
  inner iterations are about 80% of VIO wall time; any speed-up must be
  checked against the VIO trajectory hash.
- **OpenLORIS 10k rig SfM RMSE parity.** 0.3890 m vs COLMAP's 0.3843 m.
- **Mono-inertial VIO ATE on EuRoC** (the code is in; only the measurement is missing).
- **GNSS fusion on real data** (for example KITTI raw OXTS as the GNSS source).

## Guardrails (unchanged)

- No mandatory OpenCV / ONNX / PyTorch / CUDA in default crates.
- New behavior is opt-in until measured; default outputs stay bit-identical.
- Claims are scoped to the benchmark registry and claim matrix; negative
  results are recorded, not deleted.
- Every behavior change updates README / CHANGELOG / the relevant `docs/`
  page, then passes `scripts/check.sh`.
