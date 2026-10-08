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
| 0.2.1 patch release (distortion fixes) | Version bumped, CHANGELOG cut | Tag after main CI is green |
| crates.io readiness | All 16 crates package; root package trimmed from ~83 MB to 0.87 MiB compressed with an `include` allowlist | `VISLOC_PACKAGE_ALL=1 scripts/package_check.sh` passes; then `cargo publish --workspace` (needs a crates.io token) |
| BA distortion models | Pose/structure BA handles radial k1, k2 only. Extend to OPENCV p1/p2 and OPENCV_FISHEYE so BA optimizes the same model `Camera::project` measures | Finite-difference Jacobian tests and a synthetic BA round-trip per model; distortion-free and radial-only outputs stay bit-identical |

## Track 2: Sensor coverage (the empty cells in the README sensor table)

| Item | Status | Gate |
| --- | --- | --- |
| Monocular + IMU VIO on the Basalt port | Not started on the Basalt path | Synthetic mono-inertial test tracks with metric scale; stereo path bit-identical; then an EuRoC measurement (needs dataset access) |
| GNSS fusion | Example-level prior only | Joint optimization of VO constraints and GNSS position factors with online ENU alignment, lever arm and outlier gating; synthetic ATE well below VO-only ATE |

## Track 3: Usability

| Item | Status | Gate |
| --- | --- | --- |
| Python bindings (pyo3 + maturin) | Not started | Camera, Pose, COLMAP I/O, ATE evaluation from Python, with pytest coverage and a CI job |
| ROS 2 node | Not started | Needs a ROS 2 toolchain to build and test. Design first, then implement where it can be verified |

## Track 4: Maintainability

| Item | Status | Gate |
| --- | --- | --- |
| Planning docs | `PLAN.md` (233 KB) and the itemized 0.2.0 CHANGELOG (~4,800 lines) moved to [`archive/`](archive/); short handoff and changelog in their place | Docs link check passes |
| Split very large source files | `examples/unordered_sfm_demo.rs` (22.5k lines), `pipelines/basalt/src/vio/aom.rs` (21.5k), `pipelines/slam/src/incremental_sfm.rs` (18.5k), `pipelines/slam/src/bundle.rs` (17k) | Pure moves into submodules; no behavior change; full test suite and parity/hash tests unchanged |

## Needs real data (not runnable from a dataset-less environment)

These stay open, but each needs EuRoC / OpenLORIS access to measure:

- **Real-time margin.** The slowest EuRoC sequence runs at RTF 1.06×. LM
  inner iterations are about 80% of VIO wall time; any speed-up must be
  checked against the VIO trajectory hash.
- **OpenLORIS 10k rig SfM RMSE parity.** 0.3890 m vs COLMAP's 0.3843 m.
- **Mono-inertial VIO ATE on EuRoC** once Track 2 lands.

## Guardrails (unchanged)

- No mandatory OpenCV / ONNX / PyTorch / CUDA in default crates.
- New behavior is opt-in until measured; default outputs stay bit-identical.
- Claims are scoped to the benchmark registry and claim matrix; negative
  results are recorded, not deleted.
- Every behavior change updates README / CHANGELOG / the relevant `docs/`
  page, then passes `scripts/check.sh`.
