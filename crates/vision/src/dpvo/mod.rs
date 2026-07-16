//! DPVO (Teed, Lipson, Deng, NeurIPS 2023 — princeton-vl/DPVO, MIT licensed)
//! learned front-end, ported per Milestone M2 of
//! `docs/dpvo_droid_port_plan.md`. Read that plan doc — in particular its
//! "M1 results (2026-07-16)" section — before touching this module; it
//! records the *why* behind every shape/contract below, not just the *what*.
//!
//! # Scope (M2 only)
//!
//! This module is the **CPU inference/host-math layer** for one DPVO update
//! iteration: native patch extraction, native correlation lookup, an ONNX
//! Runtime session wrapper around the four graphs M1 exported
//! (`fnet.onnx`, `inet.onnx`, `dpvo_update_pre_agg.onnx`,
//! `dpvo_update_post_agg.onnx`), and a from-scratch Rust port of the two
//! pieces that resisted ONNX export (`SoftAgg`'s grouped softmax-aggregation
//! and `fastba::neighbors`'s edge-neighbour bookkeeping — see M1 results,
//! "What stayed host-side, and why").
//!
//! Explicitly **out of scope** (M3+, per the plan doc): the patch
//! inverse-depth bundle-adjustment solver, the sliding-window visual
//! odometry loop, and any wiring into `OnlineSlamPipeline` /
//! `pipelines/slam/src/sparse_factor_graph.rs`. Nothing in this module is
//! called from any pipeline yet.
//!
//! # Module layout
//!
//! * [`onnx_session`] — [`onnx_session::DpvoOnnxSession`], the `ort`-backed
//!   wrapper around the four graphs. Mirrors
//!   `crate::features::superpoint_onnx`'s session-loading pattern
//!   (`load-dynamic` / `ORT_DYLIB_PATH`, `CudaThenCpu`/`Cuda`/`Cpu` backend
//!   choice via the shared [`crate::features::superpoint_onnx::OnnxBackend`]
//!   enum) — see that module's doc comment for the runtime-library setup
//!   this crate already documents.
//! * [`patchify`] — [`patchify::patchify_cpu`], native P×P patch extraction
//!   from an `fnet`/`inet` feature map at arbitrary (possibly sub-pixel)
//!   centroids, ported from `scripts/export_dpvo_onnx.py`'s `patchify_cpu`.
//! * [`correlation`] — [`correlation::corr_cpu`], the normalized
//!   dot-product correlation lookup against a bilinearly-sampled
//!   neighbourhood of a target feature map, ported from that same script's
//!   `corr_cpu`.
//! * [`softagg`] — [`softagg::SoftAgg`] (the grouped softmax-aggregation
//!   block) and [`softagg::neighbors_cpu`] (the edge-neighbour bookkeeping),
//!   both pure Rust, no `ort`/`ndarray` tensor-graph dependency beyond the
//!   tensor *shapes* themselves.
//! * [`npz`] — a minimal in-crate `.npz` (uncompressed-ZIP-of-`.npy`)
//!   reader. Written because no `ndarray-npy` (or equivalent) dependency
//!   exists anywhere in this workspace's `Cargo.lock` (confirmed by
//!   inspection before writing this — see the M2 write-up in the plan doc),
//!   and the task's own constraints forbid adding one. It is *not*
//!   test-only: [`softagg::SoftAgg::load_from_npz`] uses it to load the
//!   SoftAgg block's trained `f`/`g`/`h` `Linear(384,384)` weights, which
//!   (see below) never made it into any ONNX graph.
//!
//! # Why does `SoftAgg` need to *load weights from an npz at all*?
//!
//! This is a gap M1 left implicit and M2 had to close. `SoftAgg` (grouped
//! softmax + weighted sum, `torch_scatter`-based upstream) could not be
//! exported to ONNX — a static graph cannot allocate a `(num_groups, DIM)`
//! scratch tensor whose `num_groups` is a traced *value* — so it was scoped
//! to run host-side from the start (M1 results, "What stayed host-side").
//! But `SoftAgg` is **not** parameter-free: it owns three trained
//! `Linear(384, 384)` layers (`f`, `g`, `h`) per instance (`agg_kk`,
//! `agg_ij`), and M1's exported artifact set (`fnet.onnx`, `inet.onnx`, the
//! two update-cell graphs, `manifest.json`) never captured those 12
//! tensors anywhere — they live entirely outside the ONNX boundary. Without
//! them, this module could implement SoftAgg's *math* but never reproduce
//! `update_cell_fixture.npz`'s real-weight `net_post_agg` (its
//! `have_real_weights: true` manifest flag means that fixture's own
//! `net_post_agg` was computed with the real checkpoint, not random init).
//!
//! M2 closed this by extending `scripts/export_dpvo_onnx.py` (not by
//! writing new hardcoded values into this crate) with a
//! `dump_softagg_weights_fixture` step that dumps `update.agg_kk.*` /
//! `update.agg_ij.*` straight from the same checkpoint M1 already used, as
//! `fixtures/softagg_weights_fixture.npz`. Regenerating fixtures from
//! scratch (`--checkpoint ... --fixtures-dir ...`) is therefore
//! self-sufficient again; there is no separate one-off script this crate
//! depends on. See that function's doc comment in the export script for the
//! full rationale, and the "M2 results" section of
//! `docs/dpvo_droid_port_plan.md` for the numeric self-check that this
//! extraction round-trips exactly (max abs diff `0.0`) against the
//! already-existing `update_cell_fixture.npz`.
//!
//! # Honesty caveats carried forward from M1
//!
//! * **Patchify and correlation are validated against this repo's own
//!   from-scratch reference reimplementation, not upstream's CUDA kernel.**
//!   DPVO ships no pure-Python reference for `altcorr.patchify`/`corr` (both
//!   are CUDA-only); `patchify_cpu`/`corr_cpu` in `export_dpvo_onnx.py` are
//!   that script's *own* reimplementation (integer-window gather +
//!   upstream's own bilinear-blend arithmetic for patchify; a
//!   `grid_sample`-equivalent bilinear neighbourhood sample + normalized dot
//!   product for correlation). Passing this module's parity tests against
//!   `patchify_fixture.npz`/`correlation_fixture.npz` therefore demonstrates
//!   **consistency with that documented reference**, not bit-parity with
//!   the real CUDA kernel — border/edge-clamp handling in particular may
//!   differ from upstream. Treat downstream ATE/tracking metrics (M4+), not
//!   this fixture, as the real arbiter of whether the reimplementation is
//!   *good enough*, per the plan doc's risk register.
//! * **`fnet.onnx`/`inet.onnx`/the two update-cell graphs *are* verified
//!   against upstream** — M1's PyTorch-vs-ONNXRuntime parity check
//!   (`scripts/check_dpvo_onnx_parity.py`) used the real `dpvo.pth`
//!   checkpoint and passed all six checked outputs at 2-4 orders of
//!   magnitude of headroom under the 1e-4 max-abs-diff threshold. What this
//!   module adds on top is: does `ort`'s CPU execution provider,called from
//!   Rust, reproduce the same graphs' outputs — a well-trodden path already
//!   proven by `superpoint_onnx.rs`/`lightglue_onnx.rs` in this same crate,
//!   not new territory.
//!
//! # Design choice: whole-module feature gate, no always-visible stub
//!
//! `crate::features::superpoint_onnx`/`lightglue_onnx` keep a
//! `cfg(not(feature = "onnx-inference"))` stub so existing call sites
//! (`--superpoint-onnx-model` CLI plumbing, etc.) still compile without the
//! feature and fail loudly at runtime. Nothing in this workspace calls into
//! `dpvo` yet (M3+ wires it up), so there is no such call site to keep
//! compiling — this whole module is gated behind `onnx-inference` instead
//! of carrying a currently-unused stub surface. Revisit this choice once
//! M3/M4 add a real caller.
#![cfg(feature = "onnx-inference")]

pub mod correlation;
pub mod npz;
pub mod onnx_session;
pub mod patchify;
pub mod softagg;

/// Matching-feature channel count (`fnet` output channels). Matches
/// `manifest.json`'s `fnet_dim` and the plan doc's M1 results.
pub const FNET_DIM: usize = 128;

/// Context/hidden channel count (`inet` output channels, and the GRU update
/// cell's hidden dimension `DIM`). Matches `manifest.json`'s `inet_dim`.
pub const DIM: usize = 384;

/// Correlation feature dimension fed into the update cell: 2 pyramid
/// levels × `(2·3+1)² = 49` taps × 3×3 patch pixels = `2*49*9 = 882`.
/// Matches `manifest.json`'s `corr_dim`.
pub const CORR_DIM: usize = 882;

/// `fnet`/`inet` output stride relative to the input image (both encoders
/// are `BasicEncoder4`, net stride 4). Matches `manifest.json`'s `res`.
pub const RES: usize = 4;

/// Patch side length (3×3 pixels). Matches `dpvo/config.py`'s patch size.
pub const PATCH: usize = 3;

/// Correlation lookup radius: `(2·3+1)² = 49` taps per pyramid level.
pub const CORR_RADIUS: usize = 3;

/// Number of correlation pyramid levels DPVO's update cell consumes.
pub const CORR_LEVELS: usize = 2;
