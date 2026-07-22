//! DPVO visual-odometry EuRoC runner — Milestone M4 of
//! `docs/dpvo_droid_port_plan.md`.
//!
//! Wires `visloc_rs::slam::dpvo_vo::DpvoOdometry` (the ported DPVO frame
//! loop) over a real EuRoC `mav0/cam0` sequence: reads images, undistorts
//! them with the sensor.yaml radial-tangential model at full resolution
//! (`dpvo/stream.py::image_stream` does the same — see `dpvo_vo.rs`'s
//! module doc, "no resolution downscaling happens"), runs
//! [`DpvoOdometry::process_frame`] per frame, writes a trajectory CSV, and
//! computes ATE (rigid + similarity, Umeyama-aligned) against ground truth
//! when available — the same evaluation pattern
//! `examples/euroc_imu_dead_reckon_demo.rs` already uses.
//!
//! # Model artifacts
//!
//! Reads the four ONNX graphs + the `SoftAgg` weight `.npz` from
//! `--model-dir` (default `E:/visloc_archive/dpvo_onnx_m1`, the M1/M2
//! artifact directory — read-only, never written here). Regenerate them
//! with `scripts/export_dpvo_onnx.py` if missing (see that script's own
//! `--help`).
//!
//! # Config: `config/default.yaml`, not `config.py`'s bare defaults
//!
//! `E:/tools/DPVO/config/default.yaml` is the config that actually produced
//! DPVO's published EuRoC number (`PATCHES_PER_FRAME=96`,
//! `REMOVAL_WINDOW=22`, `OPTIMIZATION_WINDOW=10`, `PATCH_LIFETIME=13`,
//! `KEYFRAME_THRESH=15.0`) — see `dpvo_vo.rs`'s module doc for the full
//! citation. This demo defaults to those values (overridable via CLI) rather
//! than `crate::dpvo_patch_graph::DpvoVoConfig::default`'s bare `config.py`
//! numbers.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features "image-io onnx-inference" \
//!     --example euroc_dpvo_vo_demo -- \
//!     --euroc-dir /path/to/MH_01_easy \
//!     --model-dir E:/visloc_archive/dpvo_onnx_m1 \
//!     --out-dir E:/visloc_archive/dpvo_m4_20260717 \
//!     --max-frames 400
//! ```
//!
//! Set `ORT_DYLIB_PATH` to the onnxruntime shared library, as with every
//! other ONNX-backed example in this repo.
//!
//! # `--imu` (Milestone M5, `docs/dpvo_droid_port_plan.md`)
//!
//! Feeds `mav0/imu0/data.csv` into `DpvoOdometry::push_imu`, interleaved by
//! timestamp with the camera frames (see the main loop's `imu_cursor`), and
//! builds `DpvoImuConfig::body_to_camera` directly from `cam0/sensor.yaml`'s
//! own `T_BS` via [`se3_from_t_bs`] (a verbatim copy of
//! `examples/euroc_online_slam_vi_demo.rs`'s own helper of the same name —
//! see `pipelines/slam/src/dpvo_vi_ba.rs`'s module doc for exactly which
//! direction this extrinsic must map). Off by default — omitting `--imu`
//! reproduces M4/M4-perf's visual-only behavior exactly. The summary echoes
//! the bootstrap chain's own diagnostics (`imu_bootstrapped`,
//! `imu_gravity_world_*`, `imu_bias_*`) alongside the usual ATE/scale
//! numbers, so a run's recovered `ate_similarity_scale` can be compared
//! directly against the M4-perf baseline (`1.266`) to see whether IMU
//! coupling actually pulled scale back toward `1.0`.
//!
//! # `--loop-closure` (Milestone M6, `docs/dpvo_droid_port_plan.md`)
//!
//! Enables `crate::dpvo_loop_closure`'s DPV-SLAM-style mid-term proximity
//! backend (`DpvoOdometryConfig::loop_closure`). Off by default — omitting
//! the flag reproduces M4/M4-perf/M5/M5b's visual-only-graph behavior
//! exactly (`loop_closure: None`). The summary echoes
//! `loop_closure_enabled`/`loop_batches_attempted`/`loop_candidates_evaluated`/
//! `loop_accepted`/`loop_patch_edges_added`/`loop_correction_*` alongside the
//! usual ATE numbers, and every periodic progress line reports a running
//! `loop_accepted`/`loop_candidates` count so a long run's console log shows
//! exactly when (if ever) a revisit was found.
//!
//! # `--global-ba` (Milestone M8, `docs/dpvo_droid_port_plan.md`)
//!
//! Enables the periodic full-graph "global" bundle adjustment over retained
//! active + inactive edges (`DpvoOdometryConfig::global_ba`) — the CPU-bounded
//! stand-in for upstream's `__run_global_BA`. Off by default — omitting the
//! flag reproduces M4-M7's exact behavior (`global_ba: None`, no
//! inactive-edge retention enabled on the patch graph at all). `--gba-frequency`/
//! `--gba-iterations`/`--gba-ep`/`--gba-lmbda`/`--gba-inactive-edge-cap`
//! mirror `DpvoGlobalBaConfig`'s own fields 1:1 (see that struct's doc for
//! defaults). The summary echoes `global_ba_enabled`/`global_ba_calls`/
//! `global_ba_inactive_edges_retained`/`global_ba_inactive_edges_evicted_total`/
//! `global_ba_last_*`/`global_ba_total_elapsed_ms` alongside the usual ATE
//! numbers, and every periodic progress line reports the same running
//! counters so a long run's console log shows exactly when (if ever) a global
//! pass ran and how expensive it was.
//!
//! # `--gba-widen-t0` / `--gba-max-free-poses` (Milestone M10, `docs/dpvo_droid_port_plan.md`)
//!
//! `--gba-widen-t0` is the actual M10 on/off switch layered on top of
//! `--global-ba` above (`DpvoGlobalBaConfig::widen_t0_with_loop_edges`) —
//! widens the global pass's own free-pose gauge (`t0`) using every accepted
//! proximity-loop edge's OLD endpoint, decoupled from
//! `--optimization-window`'s unrelated per-frame-BA sizing purpose,
//! materializing a folded-away endpoint (via
//! `crate::dpvo_patch_graph::DpvoPatchGraph::retained_poses`/
//! `retained_folded_frames`) as a real free pose with real patch geometry
//! when needed — see `pipelines/slam/src/dpvo_vo.rs`'s
//! `gather_widened_global_ba_problem` for the full mechanism this answers
//! (the M8/M9 results sections' own diagnosis: `free_pose_count` pinned at
//! `removal_window` on every M8 call). Off by default — omitting the flag
//! reproduces M8's exact behavior byte-for-byte even with `--global-ba`
//! itself on. `--gba-max-free-poses` bounds the resulting dense-solve cost
//! (`0` means unbounded; default mirrors `DpvoGlobalBaConfig::default()`'s
//! own `Some(256)`) — a capped call is always visible via the new
//! `global_ba_last_free_pose_count_capped` summary key, never silently
//! narrower than the loop evidence would otherwise justify. The summary
//! additionally echoes `global_ba_max_free_pose_count` (the largest
//! `free_pose_count` ever observed — the acceptance diagnostic for "did the
//! window ever actually widen"), `global_ba_last_t0_widened_by_loop_edge`,
//! and `global_ba_last_folded_poses_included`.
//!
//! # `--sim3-backend` (Milestone M9, `docs/dpvo_droid_port_plan.md`)
//!
//! Enables `crate::dpvo_sim3_backend`'s Sim(3) pose-graph scale-drift
//! correction over the full retained + live pose history
//! (`DpvoOdometryConfig::sim3_backend`) — see that module's own doc for the
//! full design and why this is a NEW mechanism layered on top of the ported
//! DPVO pipeline, not itself a straight port. Off by default — omitting the
//! flag reproduces M4-M8's exact behavior (`sim3_backend: None`). `--s3b-frequency`/
//! `--s3b-node-stride`/`--s3b-loop-edge-weight` mirror `DpvoSim3BackendConfig`'s
//! own fields 1:1. The summary echoes `sim3_backend_enabled`/`sim3_backend_calls`/
//! `sim3_backend_loop_edges_total`/`sim3_backend_last_*`/`sim3_backend_total_elapsed_ms`
//! alongside the usual ATE numbers. **Also note**: as of this milestone, the
//! trajectory CSV and ATE alignment vectors are built in a POST-HOC pass
//! after the whole run finishes (`final_pose_of`), reading each tracked
//! frame's FINAL pose rather than the live-at-commit-time pose
//! `process_frame` returns — required so a later correction (this
//! milestone's own Sim3 backend, or M8's global BA) actually reaches the
//! exported trajectory instead of being silently dropped; this changes
//! nothing for a run with neither `--global-ba` nor `--sim3-backend` enabled
//! (no mechanism ever corrects an already-committed frame's pose in that
//! configuration, so the post-hoc lookup returns the exact same value the
//! old incremental approach would have captured).
//!
//! # `--ll-sp-anchored-patches` (Milestone M12, `docs/dpvo_droid_port_plan.md`)
//!
//! Layered on top of `--long-loop`: enables
//! `DpvoLongLoopConfig::sp_anchored_patches`, attacking M11's own honest
//! negative (at `fast.yaml` patch density, a matched long-range appearance
//! keypoint essentially never lands near an existing RANDOMLY-placed DPVO
//! patch, so the 3D-3D bridge almost never finds enough correspondences) BY
//! CONSTRUCTION — patch centers are anchored at this frame's own SuperPoint
//! keypoints instead of pure uniform-random sampling, so a future revisit's
//! matched keypoint lands on (or very near) an existing patch. Off by
//! default — omitting the flag reproduces M11's exact fully-random
//! patch-sampling behavior byte-for-byte even with `--long-loop` itself on.
//! `--ll-sp-patch-min-separation` mirrors `DpvoLongLoopConfig::sp_patch_min_separation`.
//! See `crate::dpvo_long_loop::sp_anchored_patch_centers`'s own doc for the
//! coordinate mapping. The summary additionally echoes
//! `long_loop_bridge_sufficient_total` (the funnel step between "bridge
//! attempted" and "accepted") and writes every top-`K` retrieval candidate
//! (accepted or not) this run ever surfaced to
//! `<out-dir>/long_loop_candidates.csv` — the M11 open item 2 instrumentation
//! (was the tightest GT revisit ever even surfaced as a candidate?).
//!
//! # `--hover-freeze` (Milestone M14, `docs/dpvo_droid_port_plan.md`)
//!
//! Enables `DpvoOdometryConfig::low_parallax` — a causal, hysteresis-gated
//! detector for a sustained near-zero-parallax ("hover") regime that
//! freezes new-patch admission and patch/edge aging for its duration; see
//! `pipelines/slam/src/dpvo_vo.rs`'s module doc, "Low-parallax hover
//! freeze", for the full mechanism and the M13 finding it answers (MH_01's
//! own genuine ~24s near-total-stillness hover mid-sequence, whose
//! surviving ~9% of frames commit unconstrained-depth patches that BA later
//! has nothing old enough left to out-vote). Off by default — omitting the
//! flag reproduces M4-M13's exact behavior byte-for-byte (the detector is
//! never even evaluated). `--hover-window`/`--hover-enter-flow`/
//! `--hover-exit-flow` mirror `DpvoLowParallaxConfig`'s own fields 1:1 (a
//! rolling-window MEDIAN design, not a raw consecutive-frame streak — see
//! `dpvo_vo.rs`'s own module doc for why, and its own one-shot "disarms
//! after the first exit" limitation). The summary echoes
//! `hover_freeze_enabled`/`hover_regime_active`/`hover_times_entered`/
//! `hover_times_exited`/`hover_frames_suppressed_total`/`hover_disarmed`/
//! `hover_last_flow` alongside the usual ATE numbers, and every periodic
//! progress line reports the same running counters. `<out-dir>/hover_flow_trace.csv`
//! records every evaluated frame's own flow value + regime state — the
//! acceptance evidence for "did this fire at the right place, for the
//! right duration."
//!
//! # `--hover-response` / `--hover-depth-damp-factor` / `--hover-unflag-after-commits`
//! (Milestone M15, `docs/dpvo_droid_port_plan.md`)
//!
//! Only meaningful alongside `--hover-freeze` (which, despite its name, now
//! just means "enable `DpvoOdometryConfig::low_parallax`" — see
//! `dpvo_vo.rs`'s own module doc, "Milestone M15: depth-trust damping", for
//! why the flag name was kept for backward-compatible scripts).
//! `--hover-response freeze` (default) reproduces M14's exact mechanism;
//! `--hover-response depth_damp` switches to M15's Option B (commit
//! hover-span frames normally, but heavily damp their patches' depth
//! channel in every subsequent `dpvo_ba` call until the frame ages out —
//! `--hover-depth-damp-factor`/`--hover-unflag-after-commits` mirror
//! `DpvoLowParallaxConfig`'s own fields 1:1). The summary/progress-line
//! output gains `hover_response`/`hover_currently_damped_frames`/
//! `hover_frames_flagged_total`/`hover_patches_flagged_total`/
//! `hover_unflagged_total`/`hover_damped_solve_count` alongside M14's own
//! counters (all `0`/`freeze` whenever `depth_damp` is not selected).
//!
//! # A3 stage-1 densified query cadence + empty-query visibility
//! (`docs/visual_slam_sequential_sfm_plan.md`, "A3 -- Sound long-range loop
//! closure")
//!
//! `docs/visual_slam_sequential_sfm_plan.md`'s "Stage-1 baseline" measured
//! that at the COMMITTED `query_frequency = 40`, 98.3% of labelled GT
//! revisit query arrivals were never even queried — a cadence gap, not a
//! ranking gap (`scripts/eval_dpvo_long_loop_recall.py` found recall@K =
//! 1.0 CONDITIONED on a query being issued). `--ll-query-frequency` (already
//! a 1:1 mirror of `DpvoLongLoopConfig::query_frequency`, unchanged by this
//! slice) is the densification knob; its default (`40`) reproduces the
//! Stage-1 baseline's exact behavior byte-for-byte. Densifying alone would
//! be invisible in `long_loop_candidates.csv` wherever a query landed with
//! ZERO candidates (no row was ever written for it, indistinguishable from
//! "never issued") — this run now appends one
//! `rank=-1,candidate_arrival=-1,gap=-1,similarity=0.0,accepted=false`
//! marker row per such query (header unchanged; see
//! `crate::dpvo_long_loop::DpvoLongLoopIndex::empty_query_arrivals`'s own
//! doc), and the summary gains `long_loop_queries_issued_total`/
//! `long_loop_queries_with_zero_candidates` alongside the existing
//! `long_loop_queries_attempted`. `scripts/eval_dpvo_long_loop_recall.py`
//! skips `rank=-1` rows for recall-candidate purposes but still counts them
//! as issued queries, reporting `issued_query_count`/`empty_query_fraction`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use nalgebra::{Matrix4, Point2, Point3, UnitQuaternion, Vector3};
use ndarray::Array2;

use visloc_rs::core::geometry::SE3;
use visloc_rs::io::euroc::{read_euroc_dataset_dir, EurocGroundTruthSample, EurocImuSample};
use visloc_rs::io::images::read_common_image;
use visloc_rs::slam::dpvo_patch_graph::{DpvoPatchGraph, DpvoVoConfig};
use visloc_rs::slam::dpvo_vo::{
    DpvoGlobalBaConfig, DpvoImuConfig, DpvoLowParallaxConfig, DpvoOdometry, DpvoOdometryConfig,
    DpvoScaleCouplingConfig, LowParallaxResponse,
};
use visloc_rs::slam::{
    DpvoIntrinsics, DpvoLongLoopConfig, DpvoLoopClosureConfig, DpvoSim3BackendConfig,
    ImuNoiseModel, RetrievalScorer, ScaleCouplingConfig,
};
use visloc_rs::vision::distortion::RadialTangential;
use visloc_rs::vision::dpvo::npz::write_npy_f32;
use visloc_rs::vision::features::superpoint_onnx::OnnxBackend;
use visloc_rs::{umeyama_similarity_transform, TrajectorySimilarityTransform};

/// A3 ranking slice B: `--ll-min-similarity`'s default when `--ll-retrieval-scorer
/// vlad` is selected (or omitted, the overall default) — `DpvoLongLoopConfig::default().min_similarity`,
/// i.e. no behavior change from any prior milestone.
const DEFAULT_MIN_SIMILARITY_VLAD: f32 = 0.15;

/// A3 ranking slice B: `--ll-min-similarity`'s default when `--ll-retrieval-scorer
/// mean-pool` is selected. Calibrated from the offline ranking lab
/// (`scripts/eval_dpvo_retrieval_ranking_offline.py`'s own dump,
/// `E:/visloc_archive/dpvo_a3_20260721/frame_descriptors_800/`, the SAME
/// descriptors `.../ranking_offline/mh01_800_qf5_ranking.json` scores):
/// among the 355 MH_01 800-frame query arrivals whose top-1-ranked mean-pool
/// candidate is a true labelled revisit (radius 1.0 m, `min_gap=150`), the
/// MINIMUM observed similarity is `0.583` (median `0.915`); `0.5` sits
/// comfortably below that minimum (excludes 0/355 of those hits) while
/// still being `3.3x` `DEFAULT_MIN_SIMILARITY_VLAD` — i.e. still loose
/// enough that this floor is a proposal-only pre-filter, not a
/// correctness gate (the geometric gates are the actual backstop, per
/// `DpvoLongLoopConfig::min_similarity`'s own doc); a random sample of
/// eligible NON-hit pairs scores a similar range (median `~0.59`), so this
/// floor is not expected to meaningfully change WHICH candidates are
/// proposed, only to reject the appearance-degenerate low end mean-pool
/// cosine similarities can take that VLAD's own `0.15` floor was never
/// calibrated to filter.
const DEFAULT_MIN_SIMILARITY_MEAN_POOL: f32 = 0.5;

#[derive(Debug)]
struct CliArgs {
    euroc_dir: PathBuf,
    model_dir: PathBuf,
    out_dir: PathBuf,
    max_frames: usize,
    /// `evaluate_euroc.py`'s own `--stride` default (temporal subsampling,
    /// not spatial downscaling — see the module doc).
    stride: usize,
    seed: u64,
    patches_per_frame: usize,
    removal_window: usize,
    optimization_window: usize,
    patch_lifetime: usize,
    keyframe_index: usize,
    keyframe_thresh: f64,
    motion_damping: f64,
    onnx_cpu: bool,
    /// Require CUDA execution-provider registration. Unlike the default
    /// `CudaThenCpu` policy, this must fail instead of silently falling back
    /// to CPU, which makes V4 runtime evidence auditable.
    onnx_cuda: bool,
    onnx_correlation: bool,
    native_cuda_correlation_dll: Option<PathBuf>,
    /// Milestone M5 (`docs/dpvo_droid_port_plan.md`): feed `mav0/imu0/data.csv`
    /// into `DpvoOdometry::push_imu` and enable the IMU-coupled joint solve
    /// once its bootstrap chain succeeds. Default off — visual-only, exactly
    /// M4/M4-perf's own behavior.
    imu: bool,
    imu_gravity_norm_deviation_ratio: f64,
    imu_min_bootstrap_factors: usize,
    /// Multiplier on `mav0/imu0/sensor.yaml`'s own noise densities (default
    /// `1.0`, i.e. the real sensor numbers, unmodified). A diagnostic/tuning
    /// knob added while investigating real-data joint-solve behavior — see
    /// the "M5 results" section of `docs/dpvo_droid_port_plan.md`.
    imu_noise_scale: f64,
    /// Milestone M5b (`docs/dpvo_droid_port_plan.md`'s "M5b results"): the
    /// monocular-aware bootstrap's own gates, mirroring
    /// `DpvoImuConfig`'s new fields 1:1 — see that struct's own doc for what
    /// each guards against and its chosen default.
    imu_max_gyro_bias_magnitude_rad_s: f64,
    imu_gyro_bias_max_rms_after: f64,
    imu_gyro_bias_max_rms_fraction: f64,
    imu_min_mono_scale: f64,
    imu_max_mono_scale: f64,
    imu_max_mono_alignment_condition_number: f64,
    imu_rollback_mean_nis_bound: f64,
    imu_rollback_consecutive_frames: usize,
    /// Milestone M7 (`docs/dpvo_droid_port_plan.md`): enable
    /// `crate::dpvo_scale_coupling`'s continuous, uncertainty-weighted scale
    /// coupling — REPLACES the M5b one-shot bootstrap above when set
    /// (requires `--imu`). Default off — reproduces M5b's own behavior
    /// exactly (`DpvoImuConfig::scale_coupling: None`).
    scale_coupling: bool,
    sc_huber_delta: f64,
    sc_convergence_std: f64,
    sc_convergence_window: usize,
    sc_convergence_band: f64,
    sc_anneal_frames: f64,
    sc_decay_frames: f64,
    sc_max_log_step: f64,
    sc_min_window_factors: usize,
    /// Milestone M6 (`docs/dpvo_droid_port_plan.md`): enable
    /// `crate::dpvo_loop_closure`'s mid-term proximity backend. Default off
    /// — visual-graph behavior identical to every prior milestone.
    loop_closure: bool,
    /// Milestone M6: `DpvoLoopClosureConfig`'s own fields, mirrored 1:1 —
    /// see that struct's own doc for what each guards/bounds and why its
    /// default differs from `config.py` (`max_edges_per_batch` only).
    lc_backend_thresh: f64,
    lc_max_edge_age: usize,
    lc_global_opt_freq: usize,
    lc_min_loop_gap: usize,
    lc_max_edges_per_batch: usize,
    lc_nms_radius: usize,
    lc_min_valid_fraction: f64,
    /// Milestone M8 (`docs/dpvo_droid_port_plan.md`): enable periodic
    /// full-graph "global" bundle adjustment over retained active +
    /// inactive edges (`DpvoOdometryConfig::global_ba`). Default off —
    /// every prior milestone's behavior unaffected (no inactive-edge
    /// retention enabled on the patch graph at all).
    global_ba: bool,
    /// Milestone M8: `DpvoGlobalBaConfig`'s own fields, mirrored 1:1 — see
    /// that struct's own doc for what each bounds/costs.
    gba_frequency: usize,
    gba_iterations: usize,
    gba_ep: f64,
    gba_lmbda: f64,
    gba_inactive_edge_cap: usize,
    /// Milestone M10 (`docs/dpvo_droid_port_plan.md`): `DpvoGlobalBaConfig::widen_t0_with_loop_edges`
    /// — the actual "M10 on/off" switch, mirrored 1:1. Default off (M8
    /// legacy behavior).
    gba_widen_t0: bool,
    /// Milestone M10: `DpvoGlobalBaConfig::max_free_poses` — `0` here means
    /// `None` (unbounded); any other value is `Some(value)`. `0` is never a
    /// meaningful real cap (a widened pass always needs at least the live
    /// frame count), so this is an unambiguous sentinel, not a magic number
    /// collision.
    gba_max_free_poses: usize,
    /// Milestone M9 (`docs/dpvo_droid_port_plan.md`): enable the Sim(3)
    /// pose-graph scale-drift backend over the full retained + live pose
    /// history (`DpvoOdometryConfig::sim3_backend`). Default off — every
    /// prior milestone's behavior unaffected.
    sim3_backend: bool,
    /// Milestone M9: `DpvoSim3BackendConfig`'s own throttle/subsampling/
    /// loop-weight fields, mirrored 1:1 — see that struct's own doc. The
    /// nested `Sim3PoseGraphConfig` solver knobs are left at their own
    /// default (not exposed here) since M8's own `--gba-*` precedent already
    /// exposes every field of ITS solver-adjacent config and this milestone
    /// found no evidence yet that the reused `Sim3PoseGraph` solver's own
    /// iteration/damping defaults need tuning for this call site.
    s3b_frequency: usize,
    s3b_node_stride: usize,
    s3b_loop_edge_weight: f64,
    /// Transactional scale-cliff threshold, mirrored 1:1 from
    /// `DpvoSim3BackendConfig::max_abs_log_scale_correction` so frozen
    /// evaluation manifests can prove the threshold used by the run.
    s3b_max_abs_log_scale_correction: f64,
    /// Milestone M11 (`docs/dpvo_droid_port_plan.md`): enable the long-range,
    /// appearance-based loop-candidate source (`DpvoOdometryConfig::long_loop`)
    /// feeding M9's Sim3 backend and M10's widened global BA. Requires
    /// `--ll-superpoint-model`. Default off — every prior milestone's
    /// behavior unaffected.
    long_loop: bool,
    /// SuperPoint ONNX model path — required whenever `--long-loop` is set.
    ll_superpoint_model: PathBuf,
    /// Milestone M11: `DpvoLongLoopConfig`'s own fields, mirrored 1:1 — see
    /// that struct's own doc for what each bounds/gates.
    ll_vocab_bootstrap_frames: usize,
    ll_vocab_words: usize,
    /// A3 ranking slice B (`docs/visual_slam_sequential_sfm_plan.md`, "A3 —
    /// Sound long-range loop closure", "Decisive implication" paragraph):
    /// `DpvoLongLoopConfig::retrieval_scorer` — `vlad` (default, every prior
    /// milestone byte-for-byte unchanged) or `mean-pool` (vocabulary-free,
    /// queryable from the first eligible arrival — see `RetrievalScorer`'s
    /// own doc). Parsed from `--ll-retrieval-scorer {vlad,mean-pool}`.
    ll_retrieval_scorer: RetrievalScorer,
    ll_query_frequency: usize,
    ll_top_k: usize,
    /// A3 ranking slice B: `Some(..)` only when `--ll-min-similarity` was
    /// actually passed on the command line — `None` (the default) means
    /// "use the scorer-appropriate default"
    /// (`DEFAULT_MIN_SIMILARITY_VLAD`/`DEFAULT_MIN_SIMILARITY_MEAN_POOL`, see
    /// their own doc for why the two scorers need different defaults), so
    /// switching `--ll-retrieval-scorer` alone (with no explicit
    /// `--ll-min-similarity` override) picks up the right floor for the
    /// selected scorer automatically.
    ll_min_similarity: Option<f32>,
    ll_min_temporal_gap: usize,
    ll_max_indexed_frames: usize,
    ll_patch_pixel_radius: f64,
    ll_min_bridge_correspondences: usize,
    ll_ransac_iterations: usize,
    ll_min_ransac_inliers: usize,
    ll_max_mean_residual_ratio: f64,
    /// Milestone M12 (`docs/dpvo_droid_port_plan.md`): `DpvoLongLoopConfig::sp_anchored_patches` —
    /// anchor this frame's DPVO patch centers at its own SuperPoint
    /// keypoints instead of pure uniform-random sampling, attacking M11's
    /// own honest negative (the bridge from a matched appearance keypoint to
    /// a randomly-placed patch essentially never succeeds) "by construction"
    /// rather than by loosening `--ll-patch-pixel-radius`. Requires
    /// `--long-loop`. Default off — every prior milestone's behavior
    /// unaffected.
    ll_sp_anchored_patches: bool,
    /// Milestone M12: `DpvoLongLoopConfig::sp_patch_min_separation`, mirrored
    /// 1:1.
    ll_sp_patch_min_separation: f64,
    /// Milestone M12 (post-mortem addendum): `DpvoLongLoopConfig::max_rotation_inconsistency_deg`,
    /// mirrored 1:1 — the physical-consistency gate added after a real 800f
    /// corruption run (see `docs/dpvo_droid_port_plan.md`'s "M12 results").
    ll_max_rotation_inconsistency_deg: f64,
    /// A3 stage 2, first slice (`docs/visual_slam_sequential_sfm_plan.md`):
    /// `DpvoLongLoopConfig::stage2_2d2d_geometry` — require 2D-2D-first loop
    /// geometry (essential-matrix RANSAC over cross-checked SuperPoint
    /// keypoints) to pass BEFORE the existing 3D-3D bridge ever runs.
    /// Requires `--long-loop`. Default off — every prior milestone's
    /// behavior (M1-M12) unaffected.
    ll_2d2d_geometry: bool,
    /// A3 low-baseline/convention probe: run the default-off diagnostic E/F/H
    /// classifier and append its result plus recovered rotations to the
    /// candidate CSV. Requires `--ll-2d2d-geometry`; acceptance is unchanged.
    ll_2d2d_low_baseline_diagnostic: bool,
    /// A3 stage 2: `DpvoLongLoopConfig`'s new fields, mirrored 1:1.
    ll_2d2d_match_ratio: f32,
    ll_2d2d_min_inliers: usize,
    ll_2d2d_min_coverage_fraction: f64,
    ll_2d2d_max_mean_sampson_error: f64,
    ll_2d2d_umeyama_vs_e_rotation_max_deg: f64,
    /// A3 ranking-lab offline dump (`docs/visual_slam_sequential_sfm_plan.md`,
    /// "A3 — Sound long-range loop closure", ranking slice A): when `Some`,
    /// dumps EVERY frame `--long-loop` ingests (arrival index, keypoint
    /// count, raw SP keypoints + 256-d descriptors, exactly as
    /// `crate::dpvo_long_loop::DpvoLongLoopIndex::ingest_frame` receives
    /// them) as `.npy` files under this directory, for
    /// `scripts/eval_dpvo_retrieval_ranking_offline.py` to load with numpy —
    /// no Rust re-run needed to try a new ranking method. Requires
    /// `--long-loop`. `None` (default): zero extra I/O, zero extra clones
    /// (`DpvoOdometryConfig::long_loop_dump_enabled` stays `false`), and the
    /// trajectory/ATE this run produces is byte-for-byte identical to the
    /// same flags without this one.
    ll_dump_frame_descriptors: Option<PathBuf>,
    /// Milestone M14 (`docs/dpvo_droid_port_plan.md`): enable the
    /// low-parallax ("hover") freeze (`DpvoOdometryConfig::low_parallax`).
    /// Default off — every prior milestone's behavior unaffected (the
    /// detector is never even evaluated).
    hover_freeze: bool,
    /// Milestone M14: `DpvoLowParallaxConfig`'s own fields, mirrored 1:1 —
    /// see that struct's own doc for what each gates.
    hover_window: usize,
    hover_enter_flow: f64,
    hover_exit_flow: f64,
    /// Milestone M15: `"freeze"` (default, M14's mechanism) or `"depth_damp"`
    /// (M15's Option B) — parsed into `LowParallaxResponse` at config-build
    /// time (see `--hover-response`'s own arg-loop entry for the exact
    /// accepted spellings).
    hover_response: LowParallaxResponse,
    /// Milestone M15 (`DpvoLowParallaxConfig::depth_damp_factor`, mirrored
    /// 1:1).
    hover_depth_damp_factor: f64,
    /// Milestone M15 (`DpvoLowParallaxConfig::unflag_after_commits`,
    /// mirrored 1:1).
    hover_unflag_after_commits: usize,
    /// Milestone M16 gradual release duration; zero preserves M15.
    hover_release_duration_commits: usize,
    /// Milestone M16 maximum frame cohorts beginning release per commit.
    hover_release_start_cap_frames: usize,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            euroc_dir: PathBuf::new(),
            model_dir: PathBuf::from("E:/visloc_archive/dpvo_onnx_m1"),
            out_dir: PathBuf::from("target/euroc_dpvo_vo"),
            max_frames: 0,
            stride: 2,
            seed: 0,
            // config/default.yaml (see module doc).
            patches_per_frame: 96,
            removal_window: 22,
            optimization_window: 10,
            patch_lifetime: 13,
            keyframe_index: 4,
            keyframe_thresh: 15.0,
            motion_damping: 0.5,
            onnx_cpu: false,
            onnx_cuda: false,
            onnx_correlation: false,
            native_cuda_correlation_dll: None,
            imu: false,
            imu_gravity_norm_deviation_ratio: 0.3,
            imu_min_bootstrap_factors: 10,
            imu_noise_scale: 1.0,
            // Milestone M5b: mirror `DpvoImuConfig::default()` exactly so
            // omitting these flags reproduces that struct's own defaults.
            imu_max_gyro_bias_magnitude_rad_s: 0.05,
            imu_gyro_bias_max_rms_after: 0.03,
            imu_gyro_bias_max_rms_fraction: 0.5,
            imu_min_mono_scale: 0.05,
            imu_max_mono_scale: 20.0,
            imu_max_mono_alignment_condition_number: 1.0e8,
            imu_rollback_mean_nis_bound: 500.0,
            imu_rollback_consecutive_frames: 5,
            scale_coupling: false,
            // Mirror `ScaleCouplingConfig::default()`/`DpvoScaleCouplingConfig::default()`
            // exactly so omitting these flags reproduces those defaults.
            sc_huber_delta: ScaleCouplingConfig::default().huber_delta,
            sc_convergence_std: ScaleCouplingConfig::default().convergence_std,
            sc_convergence_window: ScaleCouplingConfig::default().convergence_window,
            sc_convergence_band: ScaleCouplingConfig::default().convergence_band,
            sc_anneal_frames: ScaleCouplingConfig::default().anneal_frames,
            sc_decay_frames: ScaleCouplingConfig::default().decay_frames,
            sc_max_log_step: ScaleCouplingConfig::default().max_log_step,
            sc_min_window_factors: DpvoScaleCouplingConfig::default().min_window_factors,
            loop_closure: false,
            // Mirror `DpvoLoopClosureConfig::default()` exactly so omitting
            // these flags reproduces that struct's own defaults.
            lc_backend_thresh: 64.0,
            lc_max_edge_age: 1000,
            lc_global_opt_freq: 15,
            lc_min_loop_gap: 30,
            lc_max_edges_per_batch: 8,
            lc_nms_radius: 1,
            lc_min_valid_fraction: 0.75,
            global_ba: false,
            // Mirror `DpvoGlobalBaConfig::default()` exactly so omitting
            // these flags reproduces that struct's own defaults.
            gba_frequency: DpvoGlobalBaConfig::default().frequency,
            gba_iterations: DpvoGlobalBaConfig::default().iterations,
            gba_ep: DpvoGlobalBaConfig::default().ep,
            gba_lmbda: DpvoGlobalBaConfig::default().lmbda,
            gba_inactive_edge_cap: DpvoGlobalBaConfig::default().inactive_edge_cap,
            gba_widen_t0: DpvoGlobalBaConfig::default().widen_t0_with_loop_edges,
            // `Default`'s own `Some(256)` mapped through the `0 == None`
            // sentinel above.
            gba_max_free_poses: DpvoGlobalBaConfig::default().max_free_poses.unwrap_or(0),
            sim3_backend: false,
            // Mirror `DpvoSim3BackendConfig::default()` exactly so omitting
            // these flags reproduces that struct's own defaults.
            s3b_frequency: DpvoSim3BackendConfig::default().frequency,
            s3b_node_stride: DpvoSim3BackendConfig::default().node_stride,
            s3b_loop_edge_weight: DpvoSim3BackendConfig::default().loop_edge_weight,
            s3b_max_abs_log_scale_correction: DpvoSim3BackendConfig::default()
                .max_abs_log_scale_correction,
            long_loop: false,
            ll_superpoint_model: PathBuf::new(),
            // Mirror `DpvoLongLoopConfig::default()` exactly so omitting
            // these flags reproduces that struct's own defaults.
            ll_vocab_bootstrap_frames: DpvoLongLoopConfig::default().vocab_bootstrap_frames,
            ll_vocab_words: DpvoLongLoopConfig::default().vocab_words,
            ll_retrieval_scorer: DpvoLongLoopConfig::default().retrieval_scorer,
            ll_query_frequency: DpvoLongLoopConfig::default().query_frequency,
            ll_top_k: DpvoLongLoopConfig::default().top_k,
            // `None` until `--ll-min-similarity` is actually passed — see
            // that field's own doc for why the resolved value depends on
            // `ll_retrieval_scorer`.
            ll_min_similarity: None,
            ll_min_temporal_gap: DpvoLongLoopConfig::default().min_temporal_gap,
            ll_max_indexed_frames: DpvoLongLoopConfig::default().max_indexed_frames,
            ll_patch_pixel_radius: DpvoLongLoopConfig::default().patch_pixel_radius,
            ll_min_bridge_correspondences: DpvoLongLoopConfig::default().min_bridge_correspondences,
            ll_ransac_iterations: DpvoLongLoopConfig::default().ransac_iterations,
            ll_min_ransac_inliers: DpvoLongLoopConfig::default().min_ransac_inliers,
            ll_max_mean_residual_ratio: DpvoLongLoopConfig::default().max_mean_residual_ratio,
            ll_sp_anchored_patches: DpvoLongLoopConfig::default().sp_anchored_patches,
            ll_sp_patch_min_separation: DpvoLongLoopConfig::default().sp_patch_min_separation,
            ll_max_rotation_inconsistency_deg: DpvoLongLoopConfig::default()
                .max_rotation_inconsistency_deg,
            ll_2d2d_geometry: DpvoLongLoopConfig::default().stage2_2d2d_geometry,
            ll_2d2d_low_baseline_diagnostic: DpvoLongLoopConfig::default()
                .stage2_low_baseline_diagnostic,
            ll_2d2d_match_ratio: DpvoLongLoopConfig::default()
                .stage2_match_ratio
                .unwrap_or(0.9),
            ll_2d2d_min_inliers: DpvoLongLoopConfig::default().stage2_min_inliers,
            ll_2d2d_min_coverage_fraction: DpvoLongLoopConfig::default()
                .stage2_min_coverage_fraction,
            ll_2d2d_max_mean_sampson_error: DpvoLongLoopConfig::default()
                .stage2_max_mean_sampson_error,
            ll_2d2d_umeyama_vs_e_rotation_max_deg: DpvoLongLoopConfig::default()
                .stage2_umeyama_vs_e_rotation_max_deg,
            ll_dump_frame_descriptors: None,
            hover_freeze: false,
            // Mirror `DpvoLowParallaxConfig::default()` exactly so omitting
            // these flags reproduces that struct's own defaults.
            hover_window: DpvoLowParallaxConfig::default().window,
            hover_enter_flow: DpvoLowParallaxConfig::default().enter_flow,
            hover_exit_flow: DpvoLowParallaxConfig::default().exit_flow,
            hover_response: DpvoLowParallaxConfig::default().response,
            hover_depth_damp_factor: DpvoLowParallaxConfig::default().depth_damp_factor,
            hover_unflag_after_commits: DpvoLowParallaxConfig::default().unflag_after_commits,
            hover_release_duration_commits: DpvoLowParallaxConfig::default()
                .gradual_release_duration_commits,
            hover_release_start_cap_frames: DpvoLowParallaxConfig::default()
                .gradual_release_start_cap_frames,
        }
    }
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut args = CliArgs::default();
    let mut euroc_dir: Option<PathBuf> = None;
    let mut raw: Vec<String> = env::args().skip(1).collect();
    let i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--euroc-dir" => euroc_dir = Some(PathBuf::from(raw.remove(i + 1))),
            "--model-dir" => args.model_dir = PathBuf::from(raw.remove(i + 1)),
            "--out-dir" => args.out_dir = PathBuf::from(raw.remove(i + 1)),
            "--max-frames" => args.max_frames = raw.remove(i + 1).parse()?,
            "--stride" => args.stride = raw.remove(i + 1).parse()?,
            "--seed" => args.seed = raw.remove(i + 1).parse()?,
            "--patches-per-frame" => args.patches_per_frame = raw.remove(i + 1).parse()?,
            "--removal-window" => args.removal_window = raw.remove(i + 1).parse()?,
            "--optimization-window" => args.optimization_window = raw.remove(i + 1).parse()?,
            "--patch-lifetime" => args.patch_lifetime = raw.remove(i + 1).parse()?,
            "--keyframe-index" => args.keyframe_index = raw.remove(i + 1).parse()?,
            "--keyframe-thresh" => args.keyframe_thresh = raw.remove(i + 1).parse()?,
            "--motion-damping" => args.motion_damping = raw.remove(i + 1).parse()?,
            "--onnx-cpu" => {
                args.onnx_cpu = true;
                raw.remove(i);
                continue;
            }
            "--onnx-cuda" => {
                args.onnx_cuda = true;
                raw.remove(i);
                continue;
            }
            "--onnx-correlation" => {
                args.onnx_correlation = true;
                raw.remove(i);
                continue;
            }
            "--native-cuda-correlation-dll" => {
                args.native_cuda_correlation_dll = Some(PathBuf::from(raw.remove(i + 1)))
            }
            "--imu" => {
                args.imu = true;
                raw.remove(i);
                continue;
            }
            "--imu-gravity-norm-deviation-ratio" => {
                args.imu_gravity_norm_deviation_ratio = raw.remove(i + 1).parse()?
            }
            "--imu-min-bootstrap-factors" => {
                args.imu_min_bootstrap_factors = raw.remove(i + 1).parse()?
            }
            "--imu-noise-scale" => args.imu_noise_scale = raw.remove(i + 1).parse()?,
            "--imu-max-gyro-bias-magnitude-rad-s" => {
                args.imu_max_gyro_bias_magnitude_rad_s = raw.remove(i + 1).parse()?
            }
            "--imu-gyro-bias-max-rms-after" => {
                args.imu_gyro_bias_max_rms_after = raw.remove(i + 1).parse()?
            }
            "--imu-gyro-bias-max-rms-fraction" => {
                args.imu_gyro_bias_max_rms_fraction = raw.remove(i + 1).parse()?
            }
            "--imu-min-mono-scale" => args.imu_min_mono_scale = raw.remove(i + 1).parse()?,
            "--imu-max-mono-scale" => args.imu_max_mono_scale = raw.remove(i + 1).parse()?,
            "--imu-max-mono-alignment-condition-number" => {
                args.imu_max_mono_alignment_condition_number = raw.remove(i + 1).parse()?
            }
            "--imu-rollback-mean-nis-bound" => {
                args.imu_rollback_mean_nis_bound = raw.remove(i + 1).parse()?
            }
            "--imu-rollback-consecutive-frames" => {
                args.imu_rollback_consecutive_frames = raw.remove(i + 1).parse()?
            }
            "--scale-coupling" => {
                args.scale_coupling = true;
                raw.remove(i);
                continue;
            }
            "--sc-huber-delta" => args.sc_huber_delta = raw.remove(i + 1).parse()?,
            "--sc-convergence-std" => args.sc_convergence_std = raw.remove(i + 1).parse()?,
            "--sc-convergence-window" => args.sc_convergence_window = raw.remove(i + 1).parse()?,
            "--sc-convergence-band" => args.sc_convergence_band = raw.remove(i + 1).parse()?,
            "--sc-anneal-frames" => args.sc_anneal_frames = raw.remove(i + 1).parse()?,
            "--sc-decay-frames" => args.sc_decay_frames = raw.remove(i + 1).parse()?,
            "--sc-max-log-step" => args.sc_max_log_step = raw.remove(i + 1).parse()?,
            "--sc-min-window-factors" => args.sc_min_window_factors = raw.remove(i + 1).parse()?,
            "--loop-closure" => {
                args.loop_closure = true;
                raw.remove(i);
                continue;
            }
            "--lc-backend-thresh" => args.lc_backend_thresh = raw.remove(i + 1).parse()?,
            "--lc-max-edge-age" => args.lc_max_edge_age = raw.remove(i + 1).parse()?,
            "--lc-global-opt-freq" => args.lc_global_opt_freq = raw.remove(i + 1).parse()?,
            "--lc-min-loop-gap" => args.lc_min_loop_gap = raw.remove(i + 1).parse()?,
            "--lc-max-edges-per-batch" => {
                args.lc_max_edges_per_batch = raw.remove(i + 1).parse()?
            }
            "--lc-nms-radius" => args.lc_nms_radius = raw.remove(i + 1).parse()?,
            "--lc-min-valid-fraction" => args.lc_min_valid_fraction = raw.remove(i + 1).parse()?,
            "--global-ba" => {
                args.global_ba = true;
                raw.remove(i);
                continue;
            }
            "--gba-frequency" => args.gba_frequency = raw.remove(i + 1).parse()?,
            "--gba-iterations" => args.gba_iterations = raw.remove(i + 1).parse()?,
            "--gba-ep" => args.gba_ep = raw.remove(i + 1).parse()?,
            "--gba-lmbda" => args.gba_lmbda = raw.remove(i + 1).parse()?,
            "--gba-inactive-edge-cap" => args.gba_inactive_edge_cap = raw.remove(i + 1).parse()?,
            "--gba-widen-t0" => {
                args.gba_widen_t0 = true;
                raw.remove(i);
                continue;
            }
            "--gba-max-free-poses" => args.gba_max_free_poses = raw.remove(i + 1).parse()?,
            "--sim3-backend" => {
                args.sim3_backend = true;
                raw.remove(i);
                continue;
            }
            "--s3b-frequency" => args.s3b_frequency = raw.remove(i + 1).parse()?,
            "--s3b-node-stride" => args.s3b_node_stride = raw.remove(i + 1).parse()?,
            "--s3b-loop-edge-weight" => args.s3b_loop_edge_weight = raw.remove(i + 1).parse()?,
            "--s3b-max-abs-log-scale-correction" => {
                args.s3b_max_abs_log_scale_correction = raw.remove(i + 1).parse()?
            }
            "--long-loop" => {
                args.long_loop = true;
                raw.remove(i);
                continue;
            }
            "--ll-superpoint-model" => args.ll_superpoint_model = PathBuf::from(raw.remove(i + 1)),
            "--ll-vocab-bootstrap-frames" => {
                args.ll_vocab_bootstrap_frames = raw.remove(i + 1).parse()?
            }
            "--ll-vocab-words" => args.ll_vocab_words = raw.remove(i + 1).parse()?,
            // A3 ranking slice B: like `--hover-response`, `RetrievalScorer`
            // has no general-purpose `FromStr` impl (this demo's own CLI
            // surface for it), so this arm matches the accepted spellings
            // directly.
            "--ll-retrieval-scorer" => {
                let raw_value = raw.remove(i + 1);
                args.ll_retrieval_scorer = match raw_value.as_str() {
                    "vlad" => RetrievalScorer::Vlad,
                    "mean-pool" => RetrievalScorer::MeanPool,
                    other => {
                        return Err(format!(
                        "--ll-retrieval-scorer: expected \"vlad\" or \"mean-pool\", got {other:?}"
                    )
                        .into())
                    }
                };
            }
            "--ll-query-frequency" => args.ll_query_frequency = raw.remove(i + 1).parse()?,
            "--ll-top-k" => args.ll_top_k = raw.remove(i + 1).parse()?,
            "--ll-min-similarity" => args.ll_min_similarity = Some(raw.remove(i + 1).parse()?),
            "--ll-min-temporal-gap" => args.ll_min_temporal_gap = raw.remove(i + 1).parse()?,
            "--ll-max-indexed-frames" => args.ll_max_indexed_frames = raw.remove(i + 1).parse()?,
            "--ll-patch-pixel-radius" => args.ll_patch_pixel_radius = raw.remove(i + 1).parse()?,
            "--ll-min-bridge-correspondences" => {
                args.ll_min_bridge_correspondences = raw.remove(i + 1).parse()?
            }
            "--ll-ransac-iterations" => args.ll_ransac_iterations = raw.remove(i + 1).parse()?,
            "--ll-min-ransac-inliers" => args.ll_min_ransac_inliers = raw.remove(i + 1).parse()?,
            "--ll-max-mean-residual-ratio" => {
                args.ll_max_mean_residual_ratio = raw.remove(i + 1).parse()?
            }
            "--ll-sp-anchored-patches" => {
                args.ll_sp_anchored_patches = true;
                raw.remove(i);
                continue;
            }
            "--ll-sp-patch-min-separation" => {
                args.ll_sp_patch_min_separation = raw.remove(i + 1).parse()?
            }
            "--ll-max-rotation-inconsistency-deg" => {
                args.ll_max_rotation_inconsistency_deg = raw.remove(i + 1).parse()?
            }
            "--ll-2d2d-geometry" => {
                args.ll_2d2d_geometry = true;
                raw.remove(i);
                continue;
            }
            "--ll-2d2d-low-baseline-diagnostic" => {
                args.ll_2d2d_low_baseline_diagnostic = true;
                raw.remove(i);
                continue;
            }
            "--ll-2d2d-match-ratio" => args.ll_2d2d_match_ratio = raw.remove(i + 1).parse()?,
            "--ll-2d2d-min-inliers" => args.ll_2d2d_min_inliers = raw.remove(i + 1).parse()?,
            "--ll-2d2d-min-coverage-fraction" => {
                args.ll_2d2d_min_coverage_fraction = raw.remove(i + 1).parse()?
            }
            "--ll-2d2d-max-mean-sampson-error" => {
                args.ll_2d2d_max_mean_sampson_error = raw.remove(i + 1).parse()?
            }
            "--ll-2d2d-umeyama-vs-e-rotation-max-deg" => {
                args.ll_2d2d_umeyama_vs_e_rotation_max_deg = raw.remove(i + 1).parse()?
            }
            "--ll-dump-frame-descriptors" => {
                args.ll_dump_frame_descriptors = Some(PathBuf::from(raw.remove(i + 1)))
            }
            "--hover-freeze" => {
                args.hover_freeze = true;
                raw.remove(i);
                continue;
            }
            "--hover-window" => args.hover_window = raw.remove(i + 1).parse()?,
            "--hover-enter-flow" => args.hover_enter_flow = raw.remove(i + 1).parse()?,
            "--hover-exit-flow" => args.hover_exit_flow = raw.remove(i + 1).parse()?,
            // Milestone M15 (`docs/dpvo_droid_port_plan.md`): `LowParallaxResponse`
            // has no `FromStr` impl of its own (it is not a general-purpose
            // parseable type, just this demo's own CLI surface for it), so
            // this arm matches the accepted spellings directly.
            "--hover-response" => {
                let raw_value = raw.remove(i + 1);
                args.hover_response = match raw_value.as_str() {
                    "freeze" => LowParallaxResponse::Freeze,
                    "depth_damp" => LowParallaxResponse::DepthDamp,
                    other => {
                        return Err(format!(
                            "--hover-response: expected \"freeze\" or \"depth_damp\", got {other:?}"
                        )
                        .into())
                    }
                };
            }
            "--hover-depth-damp-factor" => {
                args.hover_depth_damp_factor = raw.remove(i + 1).parse()?
            }
            "--hover-unflag-after-commits" => {
                args.hover_unflag_after_commits = raw.remove(i + 1).parse()?
            }
            "--hover-release-duration-commits" => {
                args.hover_release_duration_commits = raw.remove(i + 1).parse()?
            }
            "--hover-release-start-cap-frames" => {
                args.hover_release_start_cap_frames = raw.remove(i + 1).parse()?
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        raw.remove(i);
    }
    args.euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
    if args.onnx_cpu && args.onnx_cuda {
        return Err("--onnx-cpu and --onnx-cuda are mutually exclusive".into());
    }
    if args.onnx_correlation && !args.onnx_cuda {
        return Err("--onnx-correlation requires --onnx-cuda".into());
    }
    if args.native_cuda_correlation_dll.is_some() && !args.onnx_cuda {
        return Err("--native-cuda-correlation-dll requires --onnx-cuda".into());
    }
    if args.native_cuda_correlation_dll.is_some() && args.onnx_correlation {
        return Err(
            "--native-cuda-correlation-dll and --onnx-correlation are mutually exclusive".into(),
        );
    }
    if args.ll_dump_frame_descriptors.is_some() && !args.long_loop {
        return Err(
            "--ll-dump-frame-descriptors requires --long-loop (it dumps exactly what \
                     the long-loop index ingests)"
                .into(),
        );
    }
    if args.ll_2d2d_low_baseline_diagnostic && !args.ll_2d2d_geometry {
        return Err("--ll-2d2d-low-baseline-diagnostic requires --ll-2d2d-geometry".into());
    }
    if args.hover_release_duration_commits > 0 && args.hover_release_start_cap_frames == 0 {
        return Err(
            "--hover-release-start-cap-frames must be positive when gradual release is enabled"
                .into(),
        );
    }
    Ok(args)
}

/// Decompose an EuRoC `T_BS` 4×4 matrix into an [`SE3`]. EuRoC's own
/// convention (confirmed by cross-referencing this repo's existing
/// `examples/euroc_online_slam_vi_demo.rs::se3_from_t_bs`, which this is a
/// verbatim copy of — see `pipelines/slam/src/dpvo_vi_ba.rs`'s module doc,
/// "jacobian convention conversion" section, for exactly which direction
/// this must map): the rotation block is a proper rotation matrix, the last
/// column is the translation, and the last row is `(0, 0, 0, 1)` — exactly
/// the SE(3) layout this codebase uses, taken literally with no inversion.
fn se3_from_t_bs(t_bs: &Matrix4<f64>) -> SE3 {
    let rotation_matrix = t_bs.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(t_bs[(0, 3)], t_bs[(1, 3)], t_bs[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    SE3::new(rotation, translation)
}

/// Undistort a full grayscale image at the *same* intrinsics (matching
/// `cv2.undistort(image, K, dist)` — `dpvo/stream.py::image_stream`'s own
/// preprocessing): for every output (pinhole) pixel, map forward through
/// the distortion model to find the corresponding source pixel, then
/// bilinearly sample the original (distorted) image. Zero-pads samples
/// that land outside the source image.
fn undistort_image(
    source: &Array2<u8>,
    intrinsics: [f64; 4],
    distortion: &RadialTangential,
) -> Array2<u8> {
    let (h, w) = source.dim();
    if distortion.is_identity() {
        return source.clone();
    }
    let (fx, fy, cx, cy) = (intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let normalized = Point2::new((x as f64 - cx) / fx, (y as f64 - cy) / fy);
            let distorted_normalized = distortion.distort_normalized(normalized);
            let src_x = fx * distorted_normalized.x + cx;
            let src_y = fy * distorted_normalized.y + cy;
            out[(y, x)] = bilinear_sample_u8(source, src_x, src_y);
        }
    }
    out
}

fn bilinear_sample_u8(image: &Array2<u8>, x: f64, y: f64) -> u8 {
    let (h, w) = image.dim();
    if x < 0.0 || y < 0.0 || x >= (w - 1) as f64 || y >= (h - 1) as f64 {
        return 0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let v00 = image[(y0, x0)] as f64;
    let v01 = image[(y0, x0 + 1)] as f64;
    let v10 = image[(y0 + 1, x0)] as f64;
    let v11 = image[(y0 + 1, x0 + 1)] as f64;
    let value = v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy;
    value.round().clamp(0.0, 255.0) as u8
}

/// Milestone M9: a tracked frame's FINAL best pose, by `arrival_index` —
/// checks `retained_poses` (folded frames, possibly Sim3-corrected) first,
/// then falls back to scanning still-live `frames()` (also possibly
/// corrected, by either the global-BA pass or the Sim3 backend). See the
/// main loop's own comment on why this is a POST-HOC lookup rather than the
/// live-at-commit-time pose `process_frame` returns.
fn final_pose_of(graph: &DpvoPatchGraph, arrival_index: usize) -> Option<SE3> {
    if let Some(pose) = graph.retained_poses().get(&arrival_index) {
        return Some(pose.clone());
    }
    graph
        .frames()
        .iter()
        .find(|f| f.arrival_index == arrival_index)
        .map(|f| f.pose.clone())
}

/// A3 ranking-lab offline dump (`docs/visual_slam_sequential_sfm_plan.md`,
/// "A3 — Sound long-range loop closure", ranking slice A): write one
/// ingested frame's raw SP keypoints + descriptors to `dir` as two bare
/// `.npy` files (`{arrival:06}_keypoints.npy` shape `(N, 2)`,
/// `{arrival:06}_descriptors.npy` shape `(N, D)`), exactly as
/// `crate::dpvo_long_loop::DpvoLongLoopIndex::ingest_frame` received them
/// (patch-grid coordinates, i.e. already divided by `RES` — see
/// `DpvoOdometry::long_loop_last_ingested`'s own doc). Returns
/// `(keypoint_count, descriptor_dim)` for the caller's manifest row. See
/// `crates/vision/src/dpvo/npz.rs::write_npy_f32`'s own doc for why this is
/// a bare `.npy` per array rather than a `.npz` (no CRC32 bookkeeping
/// needed, and `numpy.load(path)` reads a bare `.npy` with zero extra code).
fn dump_long_loop_frame(
    dir: &std::path::Path,
    arrival_index: usize,
    keypoints: &[Point2<f64>],
    descriptors: &[Vec<f32>],
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let n = keypoints.len();
    debug_assert_eq!(
        n,
        descriptors.len(),
        "long-loop index always ingests one descriptor per keypoint"
    );
    let descriptor_dim = descriptors.first().map(|d| d.len()).unwrap_or(0);

    let mut keypoint_flat: Vec<f32> = Vec::with_capacity(n * 2);
    for k in keypoints {
        keypoint_flat.push(k.x as f32);
        keypoint_flat.push(k.y as f32);
    }
    let keypoints_path = dir.join(format!("{arrival_index:06}_keypoints.npy"));
    write_npy_f32(&keypoints_path, &[n, 2], &keypoint_flat)?;

    let mut descriptor_flat: Vec<f32> = Vec::with_capacity(n * descriptor_dim);
    for d in descriptors {
        // `debug_assert_eq!` below is the honest-failure backstop: every
        // descriptor a real SuperPoint extraction produces is the SAME
        // fixed dimension, but this dump must not silently truncate/pad a
        // ragged row if that assumption is ever violated in release builds.
        debug_assert_eq!(d.len(), descriptor_dim, "ragged descriptor dimension");
        descriptor_flat.extend_from_slice(d);
    }
    let descriptors_path = dir.join(format!("{arrival_index:06}_descriptors.npy"));
    write_npy_f32(&descriptors_path, &[n, descriptor_dim], &descriptor_flat)?;

    Ok((n, descriptor_dim))
}

fn nearest_ground_truth(
    samples: &[EurocGroundTruthSample],
    target_ts: i128,
) -> Option<&EurocGroundTruthSample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples
        .binary_search_by_key(&target_ts, |s| s.timestamp_nanoseconds)
        .unwrap_or_else(|insert| {
            if insert == 0 {
                0
            } else if insert >= samples.len() {
                samples.len() - 1
            } else {
                let before = samples[insert - 1].timestamp_nanoseconds;
                let after = samples[insert].timestamp_nanoseconds;
                if (target_ts - before).abs() <= (after - target_ts).abs() {
                    insert - 1
                } else {
                    insert
                }
            }
        });
    Some(&samples[idx])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out_dir)?;

    let dataset = read_euroc_dataset_dir(&args.euroc_dir)?;
    println!(
        "loaded euroc cam0_frames={} gt_samples={} resolution={:?} intrinsics={:?} distortion={:?}",
        dataset.cam0_images.len(),
        dataset.ground_truth.len(),
        dataset.cam0_calibration.resolution,
        dataset.cam0_calibration.intrinsics,
        dataset.cam0_calibration.distortion_coefficients,
    );

    let (width, height) = (
        dataset.cam0_calibration.resolution.0 as usize,
        dataset.cam0_calibration.resolution.1 as usize,
    );
    let intrinsics = dataset.cam0_calibration.intrinsics;
    let distortion = RadialTangential::from_euroc_coefficients(
        &dataset.cam0_calibration.distortion_coefficients,
    )
    .unwrap_or(RadialTangential::IDENTITY);

    let (backend, onnx_backend_requested) = if args.onnx_cpu {
        (OnnxBackend::Cpu, "cpu")
    } else if args.onnx_cuda {
        (OnnxBackend::Cuda, "cuda")
    } else {
        (OnnxBackend::default(), "cuda_then_cpu")
    };
    // A3 ranking slice B: resolve the scorer-appropriate `min_similarity`
    // floor UNLESS the caller explicitly passed `--ll-min-similarity` — see
    // `CliArgs::ll_min_similarity`'s own doc and
    // `DEFAULT_MIN_SIMILARITY_MEAN_POOL`'s own doc for the calibration.
    let ll_min_similarity = args
        .ll_min_similarity
        .unwrap_or(match args.ll_retrieval_scorer {
            RetrievalScorer::Vlad => DEFAULT_MIN_SIMILARITY_VLAD,
            RetrievalScorer::MeanPool => DEFAULT_MIN_SIMILARITY_MEAN_POOL,
        });
    let odometry_config = DpvoOdometryConfig {
        vo: DpvoVoConfig {
            buffer_size: 4096,
            patches_per_frame: args.patches_per_frame,
            removal_window: args.removal_window,
            optimization_window: args.optimization_window,
            patch_lifetime: args.patch_lifetime,
            keyframe_index: args.keyframe_index,
            keyframe_thresh: args.keyframe_thresh,
            motion_damping: args.motion_damping,
        },
        width,
        height,
        intrinsics: DpvoIntrinsics {
            fx: intrinsics[0],
            fy: intrinsics[1],
            cx: intrinsics[2],
            cy: intrinsics[3],
        },
        ba_lmbda: 1.0e-4,
        ba_ep: 100.0,
        motion_probe_min_flow: 2.0,
        seed: args.seed,
        native_cuda_correlation_dll: args.native_cuda_correlation_dll.clone(),
        fused_correlation: args.onnx_correlation,
        // Milestone M5 (`docs/dpvo_droid_port_plan.md`): `--imu` couples
        // `mav0/imu0/data.csv` into the joint solve via
        // `crate::dpvo_vi_ba`; omitting the flag reproduces M4/M4-perf's
        // visual-only behavior exactly (`imu: None`).
        imu: args.imu.then(|| DpvoImuConfig {
            body_to_camera: se3_from_t_bs(&dataset.cam0_calibration.t_body_sensor),
            noise: ImuNoiseModel {
                gyroscope_noise_density: dataset.imu_calibration.gyroscope_noise_density
                    * args.imu_noise_scale,
                accelerometer_noise_density: dataset.imu_calibration.accelerometer_noise_density
                    * args.imu_noise_scale,
            },
            gravity_magnitude: 9.81,
            gravity_norm_deviation_ratio: args.imu_gravity_norm_deviation_ratio,
            min_bootstrap_factors: args.imu_min_bootstrap_factors,
            // Milestone M5b (`docs/dpvo_droid_port_plan.md`'s "M5b
            // results"): the monocular-aware bootstrap's own gates.
            max_gyro_bias_magnitude_rad_s: args.imu_max_gyro_bias_magnitude_rad_s,
            gyro_bias_max_rms_after: args.imu_gyro_bias_max_rms_after,
            gyro_bias_max_rms_fraction: args.imu_gyro_bias_max_rms_fraction,
            min_mono_scale: args.imu_min_mono_scale,
            max_mono_scale: args.imu_max_mono_scale,
            max_mono_alignment_condition_number: args.imu_max_mono_alignment_condition_number,
            rollback_mean_nis_bound: args.imu_rollback_mean_nis_bound,
            rollback_consecutive_frames: args.imu_rollback_consecutive_frames,
            // Milestone M7 (`docs/dpvo_droid_port_plan.md`): `--scale-coupling`
            // replaces the M5b one-shot bootstrap above with
            // `crate::dpvo_scale_coupling`'s continuous mechanism; omitting
            // it reproduces M5b's exact behavior (`scale_coupling: None`).
            scale_coupling: args.scale_coupling.then_some(DpvoScaleCouplingConfig {
                scale: ScaleCouplingConfig {
                    huber_delta: args.sc_huber_delta,
                    convergence_std: args.sc_convergence_std,
                    convergence_window: args.sc_convergence_window,
                    convergence_band: args.sc_convergence_band,
                    anneal_frames: args.sc_anneal_frames,
                    decay_frames: args.sc_decay_frames,
                    max_log_step: args.sc_max_log_step,
                    ..ScaleCouplingConfig::default()
                },
                min_window_factors: args.sc_min_window_factors,
            }),
        }),
        // Milestone M6 (`docs/dpvo_droid_port_plan.md`): `--loop-closure`
        // enables the mid-term proximity backend; omitting it reproduces
        // M4/M4-perf/M5/M5b's exact visual-graph behavior (`loop_closure: None`).
        loop_closure: args.loop_closure.then_some(DpvoLoopClosureConfig {
            backend_thresh: args.lc_backend_thresh,
            max_edge_age: args.lc_max_edge_age,
            global_opt_freq: args.lc_global_opt_freq,
            min_loop_gap: args.lc_min_loop_gap,
            max_edges_per_batch: args.lc_max_edges_per_batch,
            nms_radius: args.lc_nms_radius,
            min_valid_fraction: args.lc_min_valid_fraction,
        }),
        // Milestone M8 (`docs/dpvo_droid_port_plan.md`): `--global-ba` enables
        // the periodic full-graph BA over retained active + inactive edges;
        // omitting it reproduces M4-M7's exact behavior (`global_ba: None`,
        // no inactive-edge retention enabled on the patch graph at all).
        global_ba: args.global_ba.then_some(DpvoGlobalBaConfig {
            frequency: args.gba_frequency,
            iterations: args.gba_iterations,
            ep: args.gba_ep,
            lmbda: args.gba_lmbda,
            inactive_edge_cap: args.gba_inactive_edge_cap,
            // Milestone M10: `--gba-widen-t0` is the actual "M10 on/off"
            // switch; omitting it reproduces M8's exact `t0 = min(active
            // edges' owner frame)` behavior byte-for-byte even with
            // `--global-ba` itself on.
            widen_t0_with_loop_edges: args.gba_widen_t0,
            max_free_poses: (args.gba_max_free_poses != 0).then_some(args.gba_max_free_poses),
        }),
        // Milestone M9 (`docs/dpvo_droid_port_plan.md`): `--sim3-backend`
        // enables the Sim(3) pose-graph scale-drift backend; omitting it
        // reproduces M4-M8's exact behavior (`sim3_backend: None`).
        sim3_backend: args.sim3_backend.then(|| DpvoSim3BackendConfig {
            frequency: args.s3b_frequency,
            node_stride: args.s3b_node_stride,
            loop_edge_weight: args.s3b_loop_edge_weight,
            max_abs_log_scale_correction: args.s3b_max_abs_log_scale_correction,
            ..DpvoSim3BackendConfig::default()
        }),
        // Milestone M11 (`docs/dpvo_droid_port_plan.md`): `--long-loop`
        // enables the long-range appearance loop-candidate source; omitting
        // it reproduces M4-M10's exact behavior (`long_loop: None`, no
        // SuperPoint inference at all).
        long_loop: args.long_loop.then_some(DpvoLongLoopConfig {
            vocab_bootstrap_frames: args.ll_vocab_bootstrap_frames,
            vocab_words: args.ll_vocab_words,
            retrieval_scorer: args.ll_retrieval_scorer,
            query_frequency: args.ll_query_frequency,
            top_k: args.ll_top_k,
            min_similarity: ll_min_similarity,
            min_temporal_gap: args.ll_min_temporal_gap,
            max_indexed_frames: args.ll_max_indexed_frames,
            patch_pixel_radius: args.ll_patch_pixel_radius,
            min_bridge_correspondences: args.ll_min_bridge_correspondences,
            ransac_iterations: args.ll_ransac_iterations,
            min_ransac_inliers: args.ll_min_ransac_inliers,
            max_mean_residual_ratio: args.ll_max_mean_residual_ratio,
            // Milestone M12: `--ll-sp-anchored-patches` is the actual M12
            // on/off switch; omitting it reproduces M11's exact fully-random
            // patch-sampling behavior byte-for-byte even with `--long-loop`
            // itself on.
            sp_anchored_patches: args.ll_sp_anchored_patches,
            sp_patch_min_separation: args.ll_sp_patch_min_separation,
            max_rotation_inconsistency_deg: args.ll_max_rotation_inconsistency_deg,
            // A3 stage 2, first slice: `--ll-2d2d-geometry` is the actual
            // on/off switch; omitting it reproduces M1-M12's exact behavior
            // byte-for-byte even with `--long-loop` itself on.
            stage2_2d2d_geometry: args.ll_2d2d_geometry,
            stage2_low_baseline_diagnostic: args.ll_2d2d_low_baseline_diagnostic,
            stage2_match_ratio: Some(args.ll_2d2d_match_ratio),
            stage2_min_inliers: args.ll_2d2d_min_inliers,
            stage2_min_coverage_fraction: args.ll_2d2d_min_coverage_fraction,
            stage2_max_mean_sampson_error: args.ll_2d2d_max_mean_sampson_error,
            stage2_umeyama_vs_e_rotation_max_deg: args.ll_2d2d_umeyama_vs_e_rotation_max_deg,
            ..DpvoLongLoopConfig::default()
        }),
        // Milestone M14 (`docs/dpvo_droid_port_plan.md`): `--hover-freeze`
        // enables the low-parallax hover freeze; omitting it reproduces
        // M4-M13's exact behavior byte-for-byte (`low_parallax: None`, the
        // detector never evaluated at all).
        low_parallax: args.hover_freeze.then_some(DpvoLowParallaxConfig {
            window: args.hover_window,
            enter_flow: args.hover_enter_flow,
            exit_flow: args.hover_exit_flow,
            // Milestone M15: `--hover-response depth_damp` switches to
            // Option B; omitting it keeps M14's own `Freeze` default.
            response: args.hover_response,
            depth_damp_factor: args.hover_depth_damp_factor,
            unflag_after_commits: args.hover_unflag_after_commits,
            gradual_release_duration_commits: args.hover_release_duration_commits,
            gradual_release_start_cap_frames: args.hover_release_start_cap_frames,
        }),
        // A3 ranking-lab offline dump: `--ll-dump-frame-descriptors <dir>` is
        // the actual on/off switch; omitting it leaves this `false` and
        // `DpvoOdometry::long_loop_last_ingested` stays `None` for the whole
        // run — zero behavior/perf change even with `--long-loop` itself on.
        long_loop_dump_enabled: args.ll_dump_frame_descriptors.is_some(),
    };

    if args.loop_closure {
        println!(
            "loop closure enabled: backend_thresh={:.1} max_edge_age={} global_opt_freq={} \
             min_loop_gap={} max_edges_per_batch={} nms_radius={} min_valid_fraction={:.2}",
            args.lc_backend_thresh,
            args.lc_max_edge_age,
            args.lc_global_opt_freq,
            args.lc_min_loop_gap,
            args.lc_max_edges_per_batch,
            args.lc_nms_radius,
            args.lc_min_valid_fraction,
        );
    }

    if args.global_ba {
        println!(
            "global BA enabled (Milestone M8): frequency={} iterations={} ep={:.2} lmbda={:.2e} \
             inactive_edge_cap={} widen_t0_with_loop_edges={} max_free_poses={}",
            args.gba_frequency,
            args.gba_iterations,
            args.gba_ep,
            args.gba_lmbda,
            args.gba_inactive_edge_cap,
            args.gba_widen_t0,
            if args.gba_max_free_poses == 0 {
                "none".to_string()
            } else {
                args.gba_max_free_poses.to_string()
            },
        );
    }

    if args.sim3_backend {
        println!(
            "sim3 backend enabled (Milestone M9): frequency={} node_stride={} loop_edge_weight={:.2}",
            args.s3b_frequency, args.s3b_node_stride, args.s3b_loop_edge_weight,
        );
    }

    if args.long_loop {
        let retrieval_scorer_str = match args.ll_retrieval_scorer {
            RetrievalScorer::Vlad => "vlad",
            RetrievalScorer::MeanPool => "mean-pool",
        };
        println!(
            "long-range loop enabled (Milestone M11): superpoint_model={} vocab_bootstrap_frames={} \
             vocab_words={} retrieval_scorer={} (A3 ranking slice B) query_frequency={} top_k={} \
             min_similarity={:.3}{} min_temporal_gap={} \
             max_indexed_frames={} patch_pixel_radius={:.2} min_bridge_correspondences={} \
             ransac_iterations={} min_ransac_inliers={} max_mean_residual_ratio={:.3} \
             sp_anchored_patches={} (Milestone M12) sp_patch_min_separation={:.2} \
             max_rotation_inconsistency_deg={:.1}",
            args.ll_superpoint_model.display(),
            args.ll_vocab_bootstrap_frames,
            args.ll_vocab_words,
            retrieval_scorer_str,
            args.ll_query_frequency,
            args.ll_top_k,
            ll_min_similarity,
            if args.ll_min_similarity.is_some() {
                " (explicit override)"
            } else {
                " (scorer default)"
            },
            args.ll_min_temporal_gap,
            args.ll_max_indexed_frames,
            args.ll_patch_pixel_radius,
            args.ll_min_bridge_correspondences,
            args.ll_ransac_iterations,
            args.ll_min_ransac_inliers,
            args.ll_max_mean_residual_ratio,
            args.ll_sp_anchored_patches,
            args.ll_sp_patch_min_separation,
            args.ll_max_rotation_inconsistency_deg,
        );
        if args.ll_2d2d_geometry {
            println!(
                "  A3 stage 2 (2D-2D-first loop geometry) enabled: match_ratio={:.2} \
                 min_inliers={} min_coverage_fraction={:.3} max_mean_sampson_error={:.4} \
                 umeyama_vs_e_rotation_max_deg={:.1}",
                args.ll_2d2d_match_ratio,
                args.ll_2d2d_min_inliers,
                args.ll_2d2d_min_coverage_fraction,
                args.ll_2d2d_max_mean_sampson_error,
                args.ll_2d2d_umeyama_vs_e_rotation_max_deg,
            );
            if args.ll_2d2d_low_baseline_diagnostic {
                println!(
                    "  A3 stage 2 low-baseline diagnostic enabled: E/F/H + E/Umeyama + old-3D/current-2D PnP logged; acceptance unchanged"
                );
            }
        }
    }

    if args.hover_freeze {
        println!(
            "hover detector enabled (Milestone M14-M16): window={} enter_flow={:.3} exit_flow={:.3} \
             response={:?} depth_damp_factor={:.3} unflag_after_commits={} \
             release_duration_commits={} release_start_cap_frames={}",
            args.hover_window,
            args.hover_enter_flow,
            args.hover_exit_flow,
            args.hover_response,
            args.hover_depth_damp_factor,
            args.hover_unflag_after_commits,
            args.hover_release_duration_commits,
            args.hover_release_start_cap_frames,
        );
    }

    if args.imu {
        println!(
            "imu enabled: samples={} gyro_noise_density={:.6e} accel_noise_density={:.6e} \
             body_to_camera_t=[{:.4},{:.4},{:.4}]",
            dataset.imu_samples.len(),
            dataset.imu_calibration.gyroscope_noise_density,
            dataset.imu_calibration.accelerometer_noise_density,
            odometry_config
                .imu
                .as_ref()
                .unwrap()
                .body_to_camera
                .translation
                .x,
            odometry_config
                .imu
                .as_ref()
                .unwrap()
                .body_to_camera
                .translation
                .y,
            odometry_config
                .imu
                .as_ref()
                .unwrap()
                .body_to_camera
                .translation
                .z,
        );
    }

    if args.scale_coupling {
        println!(
            "scale coupling enabled (Milestone M7): huber_delta={:.2} convergence_std={:.4} \
             convergence_window={} convergence_band={:.4} anneal_frames={:.1} decay_frames={:.1} \
             max_log_step={:.4} min_window_factors={}",
            args.sc_huber_delta,
            args.sc_convergence_std,
            args.sc_convergence_window,
            args.sc_convergence_band,
            args.sc_anneal_frames,
            args.sc_decay_frames,
            args.sc_max_log_step,
            args.sc_min_window_factors,
        );
    }

    let superpoint_model_path = args.long_loop.then(|| args.ll_superpoint_model.clone());
    let mut odometry = DpvoOdometry::new(
        odometry_config,
        args.model_dir.join("fnet.onnx"),
        args.model_dir.join("inet.onnx"),
        args.model_dir.join("dpvo_update_pre_agg.onnx"),
        args.model_dir.join("dpvo_update_post_agg.onnx"),
        args.model_dir
            .join("fixtures")
            .join("softagg_weights_fixture.npz"),
        backend,
        superpoint_model_path,
    )?;

    let frame_cap = if args.max_frames == 0 {
        usize::MAX
    } else {
        args.max_frames
    };
    let frames: Vec<_> = dataset
        .cam0_images
        .iter()
        .step_by(args.stride.max(1))
        .take(frame_cap)
        .collect();
    println!(
        "processing {} frames (stride={})",
        frames.len(),
        args.stride
    );

    // Milestone M9: `(timestamp_ns, arrival_index)` per successfully tracked
    // frame — see the main loop's own comment (right before this Vec is
    // pushed to) for why the trajectory CSV/ATE are no longer built
    // incrementally here.
    let mut tracked_entries: Vec<(i128, usize)> = Vec::new();
    let mut tracked_frames = 0usize;

    // Coarse timing split for everything *outside* `DpvoOdometry` itself
    // (`DpvoOdometryStats` only covers ONNX/BA time inside `process_frame`;
    // this repo's own decode/undistort path turned out to dominate total
    // wall time on this machine — see the M4 results section of
    // `docs/dpvo_droid_port_plan.md` for the measured split — so it is
    // timed here rather than silently folded into "everything else").
    let mut io_ms_total = 0.0_f64;
    let mut undistort_ms_total = 0.0_f64;
    // Milestone M5: running cursor into `dataset.imu_samples` (file-order,
    // i.e. already timestamp-sorted per `read_euroc_imu_csv`'s own doc
    // comment) — every sample up to and including each camera frame's own
    // timestamp is pushed just before that frame is processed, mirroring
    // how a real-time streaming caller would interleave the two sensors.
    let mut imu_cursor = 0usize;
    // Milestone M5b: track bootstrap/rollback TRANSITIONS (not just final
    // state) so the acceptance run's own console log reports exactly which
    // frame index the bootstrap first fired and any subsequent rollback —
    // the task's own required "report bootstrap frame ... rollback count".
    let mut prev_bootstrapped = false;
    let mut prev_rollback_count = 0usize;
    // Milestone M6: track the cumulative accepted-loop count so a console
    // log line only fires the frame a NEW batch is found, not every frame.
    let mut prev_loop_accepted_total = 0usize;
    // Milestone M8: same "log only on change" philosophy for the global-BA
    // call count.
    let mut prev_gba_calls = 0usize;
    // Milestone M9: same "log only on change" philosophy for the Sim3
    // backend's own call count.
    let mut prev_s3b_calls = 0usize;
    // Milestone M11: same "log only on change" philosophy for the long-range
    // loop mechanism's own accepted-total count.
    let mut prev_ll_accepted_total = 0usize;
    // A3 ranking-lab offline dump: `--ll-dump-frame-descriptors <dir>`'s own
    // state. `ll_dump_last_arrival` dedupes against
    // `DpvoOdometry::long_loop_last_ingested` staying `Some` (unchanged)
    // across frames where nothing new was ingested; `ll_dump_manifest`
    // accumulates one `(arrival_index, keypoint_count, descriptor_dim)` row
    // per actually-dumped frame, written to `manifest.csv` once after the
    // main loop (see `dump_long_loop_frame`'s own doc for the per-frame
    // `.npy` file naming).
    if let Some(dump_dir) = args.ll_dump_frame_descriptors.as_ref() {
        fs::create_dir_all(dump_dir)?;
    }
    let mut ll_dump_last_arrival: Option<usize> = None;
    let mut ll_dump_manifest: Vec<(usize, usize, usize)> = Vec::new();
    // Milestone M7: track weight/convergence/rollback TRANSITIONS (same
    // "log only on change" philosophy as M5b/M6 above).
    let mut prev_sc_converged = false;
    let mut prev_sc_rollback_count = 0usize;

    let run_start = Instant::now();
    for (idx, entry) in frames.iter().enumerate() {
        let image_path = dataset.cam0_image_dir.join(&entry.filename);
        let io_start = Instant::now();
        let grayscale = read_common_image(&image_path)?;
        // `GrayscaleImage` stores normalized `[0,1]` f32 samples
        // (`crates/vision/src/features/mod.rs`); DPVO's own contract is raw
        // `[0,255]` pixels (`dpvo_vo.rs`'s `grayscale_to_input_tensor` doc),
        // so convert back to `u8` here at the loader boundary.
        let mut raw = Array2::<u8>::zeros((grayscale.height(), grayscale.width()));
        for y in 0..grayscale.height() {
            for x in 0..grayscale.width() {
                let normalized = grayscale.get(x, y).unwrap_or(0.0);
                raw[(y, x)] = (normalized * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
        io_ms_total += io_start.elapsed().as_secs_f64() * 1000.0;

        let undistort_start = Instant::now();
        let undistorted = undistort_image(&raw, intrinsics, &distortion);
        undistort_ms_total += undistort_start.elapsed().as_secs_f64() * 1000.0;

        let timestamp_seconds = entry.timestamp_nanoseconds as f64 * 1.0e-9;

        if args.imu {
            while imu_cursor < dataset.imu_samples.len()
                && dataset.imu_samples[imu_cursor].timestamp_nanoseconds
                    <= entry.timestamp_nanoseconds
            {
                let sample: &EurocImuSample = &dataset.imu_samples[imu_cursor];
                odometry.push_imu(
                    sample.timestamp_nanoseconds as f64 * 1.0e-9,
                    sample.gyro,
                    sample.accel,
                );
                imu_cursor += 1;
            }
        }

        let pose = odometry.process_frame(undistorted.view(), timestamp_seconds)?;

        // A3 ranking-lab offline dump: `long_loop_last_ingested` is
        // overwritten (not accumulated) on every actual ingest inside
        // `process_frame`, so reading it once right here and deduping on
        // `arrival_index` captures EVERY ingested frame exactly once,
        // regardless of whether this particular `process_frame` call itself
        // committed a frame (a rejected candidate frame leaves the prior
        // ingest's snapshot in place, which the dedupe correctly skips).
        if let Some(dump_dir) = args.ll_dump_frame_descriptors.as_ref() {
            if let Some((arrival_index, keypoints, descriptors)) =
                odometry.long_loop_last_ingested()
            {
                if ll_dump_last_arrival != Some(arrival_index) {
                    ll_dump_last_arrival = Some(arrival_index);
                    let (keypoint_count, descriptor_dim) =
                        dump_long_loop_frame(dump_dir, arrival_index, keypoints, descriptors)?;
                    ll_dump_manifest.push((arrival_index, keypoint_count, descriptor_dim));
                }
            }
        }

        if args.imu {
            let imu_diag = odometry.imu_diagnostics();
            if imu_diag.bootstrapped && !prev_bootstrapped {
                println!(
                    "*** frame {idx}: IMU BOOTSTRAP SUCCEEDED — recovered_scale={:.6} gravity_world=[{:.4},{:.4},{:.4}] bias_gyro=[{:.6},{:.6},{:.6}]",
                    imu_diag.recovered_scale.unwrap_or(f64::NAN),
                    imu_diag.gravity_world.map(|g| g.x).unwrap_or(f64::NAN),
                    imu_diag.gravity_world.map(|g| g.y).unwrap_or(f64::NAN),
                    imu_diag.gravity_world.map(|g| g.z).unwrap_or(f64::NAN),
                    imu_diag.bias_gyro.x,
                    imu_diag.bias_gyro.y,
                    imu_diag.bias_gyro.z,
                );
            }
            if imu_diag.rollback_count > prev_rollback_count {
                println!(
                    "*** frame {idx}: IMU ROLLBACK #{} (bootstrap_attempts={} bootstrap_rejections={})",
                    imu_diag.rollback_count, imu_diag.bootstrap_attempts, imu_diag.bootstrap_rejections,
                );
            }
            prev_bootstrapped = imu_diag.bootstrapped;
            prev_rollback_count = imu_diag.rollback_count;
        }

        if args.scale_coupling {
            let sc_diag = odometry.scale_coupling_diagnostics();
            if sc_diag.converged && !prev_sc_converged {
                println!(
                    "*** frame {idx}: SCALE COUPLING CONVERGED — recovered_scale={:.6} \
                     posterior_log_std={:.5} bias_gyro=[{:.6},{:.6},{:.6}] weight={:.3}",
                    sc_diag.recovered_scale.unwrap_or(f64::NAN),
                    sc_diag.posterior_log_std.unwrap_or(f64::NAN),
                    sc_diag.bias_gyro.x,
                    sc_diag.bias_gyro.y,
                    sc_diag.bias_gyro.z,
                    sc_diag.weight,
                );
            }
            if sc_diag.soft_rollback_count > prev_sc_rollback_count {
                println!(
                    "*** frame {idx}: SCALE COUPLING SOFT ROLLBACK #{} (weight now {:.3}, \
                     measurements_taken={} measurements_rejected={})",
                    sc_diag.soft_rollback_count,
                    sc_diag.weight,
                    sc_diag.measurements_taken,
                    sc_diag.measurements_rejected,
                );
            }
            prev_sc_converged = sc_diag.converged;
            prev_sc_rollback_count = sc_diag.soft_rollback_count;
        }

        if args.loop_closure {
            let lc_diag = odometry.loop_closure_diagnostics();
            if lc_diag.accepted_loops_total > prev_loop_accepted_total {
                println!(
                    "*** frame {idx}: LOOP CLOSURE — accepted {} new pair(s) this batch \
                     (cumulative accepted={}, cumulative candidates={}, cumulative patch edges added={}, \
                     correction_max_m={:.4})",
                    lc_diag.last_batch_accepted_loops,
                    lc_diag.accepted_loops_total,
                    lc_diag.candidates_evaluated_total,
                    lc_diag.patch_edges_added_total,
                    lc_diag.correction_magnitude_max_m,
                );
            }
            prev_loop_accepted_total = lc_diag.accepted_loops_total;
        }

        if args.global_ba {
            let gba_diag = odometry.global_ba_diagnostics();
            if gba_diag.calls > prev_gba_calls {
                println!(
                    "*** frame {idx}: GLOBAL BA — call #{} (free_pose_count={} max_free_pose_count={} \
                     edge_count={} resolved_inactive={} unresolved_inactive={} widened={} \
                     folded_poses_included={} capped={} pose_delta_max_m={:.4} \
                     pose_delta_mean_m={:.4} elapsed_ms={:.2})",
                    gba_diag.calls,
                    gba_diag.last_free_pose_count,
                    gba_diag.max_free_pose_count,
                    gba_diag.last_edge_count,
                    gba_diag.last_resolved_inactive_edges,
                    gba_diag.last_unresolved_inactive_edges,
                    gba_diag.last_t0_widened_by_loop_edge,
                    gba_diag.last_folded_poses_included,
                    gba_diag.last_free_pose_count_capped,
                    gba_diag.last_pose_delta_max_m,
                    gba_diag.last_pose_delta_mean_m,
                    gba_diag.last_elapsed_ms,
                );
            }
            prev_gba_calls = gba_diag.calls;
        }

        if args.sim3_backend {
            let s3b_diag = odometry.sim3_backend_diagnostics();
            if s3b_diag.calls > prev_s3b_calls {
                println!(
                    "*** frame {idx}: SIM3 BACKEND — call #{} (node_count={} edge_count={} \
                     loop_edges_used={} corrections_applied={} pose_delta_max_m={:.4} \
                     pose_delta_mean_m={:.4} scale_min={:.4} scale_max={:.4} committed={} \
                     rejection={:?} elapsed_ms={:.2})",
                    s3b_diag.calls,
                    s3b_diag.last_node_count,
                    s3b_diag.last_edge_count,
                    s3b_diag.last_loop_edges_used,
                    s3b_diag.last_scale_corrections_applied,
                    s3b_diag.last_pose_delta_max_m,
                    s3b_diag.last_pose_delta_mean_m,
                    s3b_diag.last_scale_min,
                    s3b_diag.last_scale_max,
                    s3b_diag.last_committed,
                    s3b_diag.last_rejection,
                    s3b_diag.last_elapsed_ms,
                );
            }
            prev_s3b_calls = s3b_diag.calls;
        }

        if args.long_loop {
            let ll_diag = odometry.long_loop_diagnostics();
            if ll_diag.accepted_total > prev_ll_accepted_total {
                println!(
                    "*** frame {idx}: LONG-RANGE LOOP — accepted #{} (arrival_i={} arrival_j={} gap={} \
                     similarity={:.3} scale={:.4} inliers={} residual_ratio={:.4} frames_indexed={} \
                     vocab_built={} queries_attempted={} verification_attempts={})",
                    ll_diag.accepted_total,
                    ll_diag.last_accepted_arrival_i,
                    ll_diag.last_accepted_arrival_j,
                    ll_diag.last_accepted_gap,
                    ll_diag.last_accepted_similarity,
                    ll_diag.last_accepted_scale,
                    ll_diag.last_accepted_inliers,
                    ll_diag.last_accepted_mean_residual_ratio,
                    ll_diag.frames_indexed,
                    ll_diag.vocab_built,
                    ll_diag.queries_attempted,
                    ll_diag.verification_attempts,
                );
            }
            prev_ll_accepted_total = ll_diag.accepted_total;
        }

        // Milestone M9: record only `(timestamp, arrival_index)` here — the
        // trajectory CSV and ATE evaluation are built in a POST-HOC pass
        // after the whole run finishes (see `final_pose_of`, called below
        // the main loop), so that a LATER correction (global BA widening the
        // window, or this milestone's own Sim3 backend correcting a frame
        // that already committed many iterations ago) is actually reflected
        // in the exported trajectory instead of the STALE pose this frame's
        // own `process_frame` call happened to return at commit time. Prior
        // milestones built `traj_csv`/`aligned_estimated` incrementally,
        // right here, which silently froze each frame's pose at ITS OWN
        // commit time — harmless before M9 (a corrected OLD frame's pose was
        // never written back this far outside the BA window), but exactly
        // the gap M9's own retained-pose-history + Sim3 corrections need
        // this pass to close.
        if pose.is_some() {
            tracked_frames += 1;
            let arrival_index = odometry
                .graph()
                .frames()
                .last()
                .expect("process_frame returned Some(pose) => at least one live frame exists")
                .arrival_index;
            tracked_entries.push((entry.timestamp_nanoseconds, arrival_index));
        }

        if idx % 10 == 0 || idx + 1 == frames.len() {
            let stats = odometry.stats();
            let n = stats.frames_processed.max(1) as f64;
            let imu_diag = odometry.imu_diagnostics();
            println!(
                "frame {}/{} tracked={} frames_graph_n={} io_ms_avg={:.2} undistort_ms_avg={:.2} encode_ms_avg={:.2} corr_ms_avg={:.2} update_ms_avg={:.2} ba_ms_avg={:.2} imu_bootstrapped={} imu_attempts={} imu_rejections={} imu_rollbacks={}",
                idx + 1,
                frames.len(),
                tracked_frames,
                odometry.graph().n_frames(),
                io_ms_total / n,
                undistort_ms_total / n,
                stats.encode_ms_total / n,
                stats.correlation_ms_total / n,
                stats.update_ms_total / n,
                stats.ba_ms_total / n,
                imu_diag.bootstrapped,
                imu_diag.bootstrap_attempts,
                imu_diag.bootstrap_rejections,
                imu_diag.rollback_count,
            );
            if args.imu {
                println!(
                    "  imu_rejection_counts={:?} last_rejection={:?}",
                    imu_diag.rejection_counts, imu_diag.last_rejection,
                );
            }
            if args.scale_coupling {
                let sc_diag = odometry.scale_coupling_diagnostics();
                println!(
                    "  sc_weight={:.3} sc_converged={} sc_recovered_scale={:.4} sc_posterior_log_std={:.5} \
                     sc_measurements_taken={} sc_measurements_rejected={} sc_soft_rollbacks={}",
                    sc_diag.weight,
                    sc_diag.converged,
                    sc_diag.recovered_scale.unwrap_or(f64::NAN),
                    sc_diag.posterior_log_std.unwrap_or(f64::NAN),
                    sc_diag.measurements_taken,
                    sc_diag.measurements_rejected,
                    sc_diag.soft_rollback_count,
                );
                println!(
                    "  sc_rejection_counts={:?} sc_last_rejection={:?}",
                    sc_diag.rejection_counts, sc_diag.last_rejection,
                );
            }
            if args.loop_closure {
                let lc_diag = odometry.loop_closure_diagnostics();
                println!(
                    "  loop_batches={} loop_candidates={} loop_accepted={} loop_patch_edges_added={} \
                     loop_correction_max_m={:.4} loop_correction_mean_m={:.4}",
                    lc_diag.batches_attempted,
                    lc_diag.candidates_evaluated_total,
                    lc_diag.accepted_loops_total,
                    lc_diag.patch_edges_added_total,
                    lc_diag.correction_magnitude_max_m,
                    lc_diag.correction_magnitude_mean_m,
                );
            }
            if args.global_ba {
                let gba_diag = odometry.global_ba_diagnostics();
                println!(
                    "  gba_calls={} gba_inactive_edges_retained={} gba_inactive_edges_evicted_total={} \
                     gba_last_free_pose_count={} gba_max_free_pose_count={} gba_last_edge_count={} \
                     gba_last_widened={} gba_last_folded_poses_included={} gba_last_capped={} \
                     gba_last_pose_delta_max_m={:.4} gba_total_elapsed_ms={:.2}",
                    gba_diag.calls,
                    gba_diag.inactive_edges_retained,
                    gba_diag.inactive_edges_evicted_total,
                    gba_diag.last_free_pose_count,
                    gba_diag.max_free_pose_count,
                    gba_diag.last_edge_count,
                    gba_diag.last_t0_widened_by_loop_edge,
                    gba_diag.last_folded_poses_included,
                    gba_diag.last_free_pose_count_capped,
                    gba_diag.last_pose_delta_max_m,
                    gba_diag.total_elapsed_ms,
                );
            }
            if args.sim3_backend {
                let s3b_diag = odometry.sim3_backend_diagnostics();
                println!(
                    "  s3b_calls={} s3b_loop_edges_total={} s3b_last_node_count={} s3b_last_edge_count={} \
                     s3b_last_corrections_applied={} s3b_last_pose_delta_max_m={:.4} \
                     s3b_last_scale_min={:.4} s3b_last_scale_max={:.4} s3b_last_committed={} \
                     s3b_last_rejection={:?} s3b_total_elapsed_ms={:.2}",
                    s3b_diag.calls,
                    s3b_diag.loop_edges_total,
                    s3b_diag.last_node_count,
                    s3b_diag.last_edge_count,
                    s3b_diag.last_scale_corrections_applied,
                    s3b_diag.last_pose_delta_max_m,
                    s3b_diag.last_scale_min,
                    s3b_diag.last_scale_max,
                    s3b_diag.last_committed,
                    s3b_diag.last_rejection,
                    s3b_diag.total_elapsed_ms,
                );
            }
            if args.hover_freeze {
                let hf_diag = odometry.low_parallax_diagnostics();
                println!(
                    "  hover_regime_active={} hover_times_entered={} hover_times_exited={} \
                     hover_frames_suppressed_total={} hover_disarmed={} hover_last_flow={:.4}",
                    hf_diag.regime_active,
                    hf_diag.times_entered,
                    hf_diag.times_exited,
                    hf_diag.frames_suppressed_total,
                    hf_diag.disarmed,
                    hf_diag.last_flow,
                );
                // Milestone M15: only ever nonzero under `--hover-response
                // depth_damp` — see `LowParallaxDampState`'s own doc.
                println!(
                    "  hover_response={:?} hover_currently_damped_frames={} hover_frames_flagged_total={} \
                     hover_patches_flagged_total={} hover_unflagged_total={} hover_damped_solve_count={}",
                    hf_diag.response,
                    hf_diag.currently_damped_frames,
                    hf_diag.frames_flagged_total,
                    hf_diag.patches_flagged_total,
                    hf_diag.unflagged_total,
                    hf_diag.damped_solve_count,
                );
                println!(
                    "  hover_releasing_frames={} hover_release_started_total={} \
                     hover_release_start_max={} hover_release_histogram_frames={:?}",
                    hf_diag.currently_releasing_frames,
                    hf_diag.release_started_total,
                    hf_diag.max_release_started_per_advance,
                    hf_diag.release_histogram_frames,
                );
            }
        }
    }
    let total_elapsed_s = run_start.elapsed().as_secs_f64();

    // Milestone M9: build the trajectory CSV / ATE alignment vectors NOW,
    // reading each tracked frame's FINAL pose (`final_pose_of`) — live
    // frames read straight from `odometry.graph().frames()`, folded frames
    // from `odometry.graph().retained_poses()` (both already reflect every
    // correction applied up to this point, including the Sim3 backend's
    // own write-back) — rather than the stale live-at-commit-time pose the
    // old incremental approach captured. See the main loop's own comment at
    // the `tracked_entries.push(...)` call site for the full rationale.
    let mut traj_csv = String::from("timestamp_ns,tx,ty,tz,qw,qx,qy,qz\n");
    let mut aligned_estimated: Vec<Point3<f64>> = Vec::new();
    let mut aligned_reference: Vec<Point3<f64>> = Vec::new();
    for &(timestamp_ns, arrival_index) in &tracked_entries {
        let Some(pose_world_to_camera) = final_pose_of(odometry.graph(), arrival_index) else {
            continue;
        };
        // DPVO poses are `T_world_to_camera` (see `dpvo_patch_ba.rs`'s
        // convention-mapping doc) — the camera center in world is the
        // inverse's translation.
        let camera_in_world = pose_world_to_camera.inverse();
        let q = camera_in_world.rotation.quaternion();
        traj_csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            timestamp_ns,
            camera_in_world.translation.x,
            camera_in_world.translation.y,
            camera_in_world.translation.z,
            q.w,
            q.i,
            q.j,
            q.k,
        ));
        if let Some(gt) = nearest_ground_truth(&dataset.ground_truth, timestamp_ns) {
            aligned_estimated.push(Point3::from(camera_in_world.translation));
            aligned_reference.push(Point3::from(gt.position_world));
        }
    }

    let traj_path = args.out_dir.join("dpvo_trajectory.csv");
    fs::write(&traj_path, &traj_csv)?;

    let stats = odometry.stats();
    let ms_per_frame = if stats.frames_processed > 0 {
        total_elapsed_s * 1000.0 / stats.frames_processed as f64
    } else {
        0.0
    };

    let (ate_rigid_rmse, ate_rigid_max, ate_sim_rmse, ate_sim_max, ate_sim_scale) =
        if aligned_estimated.len() >= 3 {
            let rigid = umeyama_similarity_transform(&aligned_estimated, &aligned_reference, false)
                .unwrap_or_else(TrajectorySimilarityTransform::identity);
            let similarity =
                umeyama_similarity_transform(&aligned_estimated, &aligned_reference, true)
                    .unwrap_or_else(TrajectorySimilarityTransform::identity);
            let mut rmse_rigid_sq = 0.0;
            let mut max_rigid = 0.0_f64;
            let mut rmse_sim_sq = 0.0;
            let mut max_sim = 0.0_f64;
            for (est, gt) in aligned_estimated.iter().zip(aligned_reference.iter()) {
                let rigid_err = (rigid.apply(est) - gt).norm();
                let sim_err = (similarity.apply(est) - gt).norm();
                rmse_rigid_sq += rigid_err * rigid_err;
                rmse_sim_sq += sim_err * sim_err;
                max_rigid = max_rigid.max(rigid_err);
                max_sim = max_sim.max(sim_err);
            }
            let n = aligned_estimated.len() as f64;
            (
                (rmse_rigid_sq / n).sqrt(),
                max_rigid,
                (rmse_sim_sq / n).sqrt(),
                max_sim,
                similarity.scale,
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        };

    let tracked_fraction = if !frames.is_empty() {
        tracked_frames as f64 / frames.len() as f64
    } else {
        0.0
    };
    let imu_diag = odometry.imu_diagnostics();
    let (gravity_x, gravity_y, gravity_z) = imu_diag
        .gravity_world
        .map(|g| (g.x, g.y, g.z))
        .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let lc_diag = odometry.loop_closure_diagnostics();
    let sc_diag = odometry.scale_coupling_diagnostics();
    let gba_diag = odometry.global_ba_diagnostics();
    let s3b_diag = odometry.sim3_backend_diagnostics();
    let ll_diag = odometry.long_loop_diagnostics();
    let hf_diag = odometry.low_parallax_diagnostics();
    let full_update_graph_enabled = odometry.full_update_graph_enabled();
    let correlation_graph_enabled = odometry.correlation_graph_enabled();
    let native_cuda_correlation_enabled = odometry.native_cuda_correlation_enabled();

    let summary = format!(
        "euroc_dir={}\n\
         model_dir={}\n\
         onnx_backend_requested={onnx_backend_requested}\n\
         onnx_full_update_graph_enabled={full_update_graph_enabled}\n\
         onnx_correlation_graph_enabled={correlation_graph_enabled}\n\
         onnx_correlation_requested={}\n\
         native_cuda_correlation_enabled={native_cuda_correlation_enabled}\n\
         native_cuda_correlation_dll={}\n\
         frames_requested={frame_count}\n\
         frames_tracked={tracked_frames}\n\
         tracked_fraction={tracked_fraction:.4}\n\
         total_elapsed_s={total_elapsed_s:.2}\n\
         ms_per_frame_total={ms_per_frame:.2}\n\
         ms_per_frame_io={io_ms:.2}\n\
         ms_per_frame_undistort={undistort_ms:.2}\n\
         ms_per_frame_encode={encode_ms:.2}\n\
         ms_per_frame_correlation={corr_ms:.2}\n\
         ms_per_frame_native_correlation_device={native_corr_device_ms:.2}\n\
         ms_per_frame_update={update_ms:.2}\n\
         ms_per_frame_ba={ba_ms:.2}\n\
         ate_rigid_rmse_m={ate_rigid_rmse:.4}\n\
         ate_rigid_max_m={ate_rigid_max:.4}\n\
         ate_similarity_rmse_m={ate_sim_rmse:.4}\n\
         ate_similarity_max_m={ate_sim_max:.4}\n\
         ate_similarity_scale={ate_sim_scale:.6}\n\
         gt_matched_samples={matched}\n\
         imu_enabled={imu_enabled}\n\
         imu_bootstrapped={imu_bootstrapped}\n\
         imu_gravity_world_x={gravity_x:.4}\n\
         imu_gravity_world_y={gravity_y:.4}\n\
         imu_gravity_world_z={gravity_z:.4}\n\
         imu_bias_gyro_x={bias_gyro_x:.6}\n\
         imu_bias_gyro_y={bias_gyro_y:.6}\n\
         imu_bias_gyro_z={bias_gyro_z:.6}\n\
         imu_bias_accel_x={bias_accel_x:.6}\n\
         imu_bias_accel_y={bias_accel_y:.6}\n\
         imu_bias_accel_z={bias_accel_z:.6}\n\
         imu_recovered_scale={recovered_scale:.6}\n\
         imu_bootstrap_attempts={bootstrap_attempts}\n\
         imu_bootstrap_rejections={bootstrap_rejections}\n\
         imu_rollback_count={rollback_count}\n\
         imu_reject_gyro_estimator_none={rc_gyro_none}\n\
         imu_reject_gyro_magnitude={rc_gyro_mag}\n\
         imu_reject_gyro_rms_absolute={rc_gyro_rms_abs}\n\
         imu_reject_gyro_rms_fraction={rc_gyro_rms_frac}\n\
         imu_reject_mono_not_enough_factors={rc_mono_few}\n\
         imu_reject_mono_underdetermined={rc_mono_underdet}\n\
         imu_reject_mono_ill_conditioned={rc_mono_illcond}\n\
         imu_reject_mono_degenerate_solve={rc_mono_degen}\n\
         imu_reject_mono_gravity_norm={rc_mono_grav}\n\
         imu_reject_mono_scale_range={rc_mono_scale}\n\
         imu_last_rejection={last_rejection}\n\
         scale_coupling_enabled={sc_enabled}\n\
         scale_coupling_weight={sc_weight:.4}\n\
         scale_coupling_converged={sc_converged}\n\
         scale_coupling_recovered_scale={sc_recovered_scale:.6}\n\
         scale_coupling_posterior_log_std={sc_posterior_log_std:.6}\n\
         scale_coupling_bias_gyro_x={sc_bias_gyro_x:.6}\n\
         scale_coupling_bias_gyro_y={sc_bias_gyro_y:.6}\n\
         scale_coupling_bias_gyro_z={sc_bias_gyro_z:.6}\n\
         scale_coupling_measurements_taken={sc_measurements_taken}\n\
         scale_coupling_measurements_rejected={sc_measurements_rejected}\n\
         scale_coupling_soft_rollback_count={sc_soft_rollback_count}\n\
         scale_coupling_reject_not_enough_factors={sc_rc_few}\n\
         scale_coupling_reject_underdetermined={sc_rc_underdet}\n\
         scale_coupling_reject_ill_conditioned={sc_rc_illcond}\n\
         scale_coupling_reject_degenerate_solve={sc_rc_degen}\n\
         scale_coupling_reject_gravity_norm={sc_rc_grav}\n\
         scale_coupling_reject_scale_range={sc_rc_scale}\n\
         scale_coupling_last_rejection={sc_last_rejection}\n\
         loop_closure_enabled={lc_enabled}\n\
         loop_batches_attempted={lc_batches}\n\
         loop_candidates_evaluated={lc_candidates}\n\
         loop_accepted={lc_accepted}\n\
         loop_patch_edges_added={lc_edges_added}\n\
         loop_correction_events={lc_correction_events}\n\
         loop_correction_magnitude_max_m={lc_correction_max:.6}\n\
         loop_correction_magnitude_mean_m={lc_correction_mean:.6}\n\
         global_ba_enabled={gba_enabled}\n\
         global_ba_calls={gba_calls}\n\
         global_ba_inactive_edges_retained={gba_inactive_retained}\n\
         global_ba_inactive_edges_evicted_total={gba_inactive_evicted}\n\
         global_ba_last_free_pose_count={gba_last_free_pose_count}\n\
         global_ba_max_free_pose_count={gba_max_free_pose_count}\n\
         global_ba_last_edge_count={gba_last_edge_count}\n\
         global_ba_last_resolved_inactive_edges={gba_last_resolved_inactive}\n\
         global_ba_last_unresolved_inactive_edges={gba_last_unresolved_inactive}\n\
         global_ba_last_t0_widened_by_loop_edge={gba_last_widened}\n\
         global_ba_last_folded_poses_included={gba_last_folded_included}\n\
         global_ba_last_free_pose_count_capped={gba_last_capped}\n\
         global_ba_last_pose_delta_max_m={gba_last_pose_delta_max:.6}\n\
         global_ba_last_pose_delta_mean_m={gba_last_pose_delta_mean:.6}\n\
         global_ba_last_elapsed_ms={gba_last_elapsed_ms:.3}\n\
         global_ba_total_elapsed_ms={gba_total_elapsed_ms:.3}\n\
         sim3_backend_enabled={s3b_enabled}\n\
         sim3_backend_calls={s3b_calls}\n\
         sim3_backend_loop_edges_total={s3b_loop_edges_total}\n\
         sim3_backend_last_node_count={s3b_last_node_count}\n\
         sim3_backend_last_edge_count={s3b_last_edge_count}\n\
         sim3_backend_last_loop_edges_used={s3b_last_loop_edges_used}\n\
         sim3_backend_last_scale_corrections_applied={s3b_last_corrections}\n\
         sim3_backend_last_pose_delta_max_m={s3b_last_pose_delta_max:.6}\n\
         sim3_backend_last_pose_delta_mean_m={s3b_last_pose_delta_mean:.6}\n\
         sim3_backend_last_scale_min={s3b_last_scale_min:.6}\n\
         sim3_backend_last_scale_max={s3b_last_scale_max:.6}\n\
         sim3_backend_max_abs_log_scale_correction={s3b_max_abs_log_scale_correction:.6}\n\
         sim3_backend_max_committed_abs_log_scale={s3b_max_committed_abs_log_scale:.6}\n\
         sim3_backend_scale_jump_rejections_total={s3b_scale_jump_rejections_total}\n\
         sim3_backend_last_committed={s3b_last_committed}\n\
         sim3_backend_last_rejection={s3b_last_rejection:?}\n\
         sim3_backend_last_elapsed_ms={s3b_last_elapsed_ms:.3}\n\
         sim3_backend_total_elapsed_ms={s3b_total_elapsed_ms:.3}\n\
         long_loop_enabled={ll_enabled}\n\
         long_loop_frames_indexed={ll_frames_indexed}\n\
         long_loop_vocab_built={ll_vocab_built}\n\
         long_loop_estimated_index_bytes={ll_estimated_bytes}\n\
         long_loop_queries_attempted={ll_queries_attempted}\n\
         long_loop_queries_issued_total={ll_queries_issued_total}\n\
         long_loop_queries_with_zero_candidates={ll_queries_zero_candidates}\n\
         long_loop_candidates_considered={ll_candidates_considered}\n\
         long_loop_verification_attempts={ll_verification_attempts}\n\
         long_loop_bridge_sufficient_total={ll_bridge_sufficient}\n\
         long_loop_rejected_rotation_inconsistent_total={ll_rejected_rotation}\n\
         long_loop_accepted_total={ll_accepted_total}\n\
         long_loop_rejected_insufficient_bridge_total={ll_rejected_bridge}\n\
         long_loop_rejected_ransac_total={ll_rejected_ransac}\n\
         long_loop_last_accepted_arrival_i={ll_last_arrival_i}\n\
         long_loop_last_accepted_arrival_j={ll_last_arrival_j}\n\
         long_loop_last_accepted_gap={ll_last_gap}\n\
         long_loop_last_accepted_similarity={ll_last_similarity:.6}\n\
         long_loop_last_accepted_scale={ll_last_scale:.6}\n\
         long_loop_last_accepted_inliers={ll_last_inliers}\n\
         long_loop_last_accepted_mean_residual_ratio={ll_last_residual_ratio:.6}\n\
         long_loop_total_elapsed_ms={ll_total_elapsed_ms:.3}\n\
         long_loop_sp_anchored_patches={ll_sp_anchored}\n\
         long_loop_sp_patch_min_separation={ll_sp_min_separation:.2}\n\
         long_loop_max_rotation_inconsistency_deg={ll_max_rot_inconsistency:.1}\n\
         long_loop_query_log_entries={ll_query_log_len}\n\
         long_loop_stage2_enabled={ll_stage2_enabled}\n\
         long_loop_stage2_attempts_total={ll_stage2_attempts}\n\
         long_loop_stage2_passed_total={ll_stage2_passed}\n\
         long_loop_stage2_rejected_insufficient_matches_total={ll_stage2_rej_matches}\n\
         long_loop_stage2_rejected_insufficient_inliers_total={ll_stage2_rej_inliers}\n\
         long_loop_stage2_rejected_insufficient_coverage_total={ll_stage2_rej_coverage}\n\
         long_loop_stage2_rejected_rotation_inconsistent_total={ll_stage2_rej_rotation}\n\
         long_loop_stage2_rejected_high_residual_total={ll_stage2_rej_residual}\n\
         long_loop_stage2_rejected_umeyama_vs_e_rotation_total={ll_stage2_rej_umeyama_vs_e}\n\
         hover_freeze_enabled={hf_enabled}\n\
         hover_regime_active={hf_regime_active}\n\
         hover_times_entered={hf_times_entered}\n\
         hover_times_exited={hf_times_exited}\n\
         hover_frames_suppressed_total={hf_frames_suppressed}\n\
         hover_disarmed={hf_disarmed}\n\
         hover_last_flow={hf_last_flow:.6}\n\
         hover_last_enter_frame={hf_last_enter_frame}\n\
         hover_last_exit_frame={hf_last_exit_frame}\n\
         hover_window={hf_window}\n\
         hover_enter_flow={hf_enter_flow:.4}\n\
         hover_exit_flow={hf_exit_flow:.4}\n\
         hover_response={hf_response:?}\n\
         hover_depth_damp_factor={hf_depth_damp_factor:.3}\n\
         hover_unflag_after_commits={hf_unflag_after_commits}\n\
         hover_release_duration_commits={hf_release_duration_commits}\n\
         hover_release_start_cap_frames={hf_release_start_cap_frames}\n\
         hover_currently_damped_frames={hf_currently_damped_frames}\n\
         hover_frames_flagged_total={hf_frames_flagged_total}\n\
         hover_patches_flagged_total={hf_patches_flagged_total}\n\
         hover_unflagged_total={hf_unflagged_total}\n\
         hover_damped_solve_count={hf_damped_solve_count}\n\
         hover_currently_releasing_frames={hf_currently_releasing_frames}\n\
         hover_release_started_total={hf_release_started_total}\n\
         hover_release_start_max={hf_release_start_max}\n\
         hover_release_histogram_frames={hf_release_histogram_frames:?}\n",
        args.euroc_dir.display(),
        args.model_dir.display(),
        args.onnx_correlation,
        args.native_cuda_correlation_dll
            .as_deref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".into()),
        frame_count = frames.len(),
        io_ms = io_ms_total / stats.frames_processed.max(1) as f64,
        undistort_ms = undistort_ms_total / stats.frames_processed.max(1) as f64,
        encode_ms = stats.encode_ms_total / stats.frames_processed.max(1) as f64,
        corr_ms = stats.correlation_ms_total / stats.frames_processed.max(1) as f64,
        native_corr_device_ms =
            stats.native_correlation_device_ms_total / stats.frames_processed.max(1) as f64,
        update_ms = stats.update_ms_total / stats.frames_processed.max(1) as f64,
        ba_ms = stats.ba_ms_total / stats.frames_processed.max(1) as f64,
        matched = aligned_estimated.len(),
        imu_enabled = args.imu,
        imu_bootstrapped = imu_diag.bootstrapped,
        bias_gyro_x = imu_diag.bias_gyro.x,
        bias_gyro_y = imu_diag.bias_gyro.y,
        bias_gyro_z = imu_diag.bias_gyro.z,
        bias_accel_x = imu_diag.bias_accel.x,
        bias_accel_y = imu_diag.bias_accel.y,
        bias_accel_z = imu_diag.bias_accel.z,
        recovered_scale = imu_diag.recovered_scale.unwrap_or(f64::NAN),
        bootstrap_attempts = imu_diag.bootstrap_attempts,
        bootstrap_rejections = imu_diag.bootstrap_rejections,
        rollback_count = imu_diag.rollback_count,
        rc_gyro_none = imu_diag.rejection_counts.gyro_estimator_none,
        rc_gyro_mag = imu_diag.rejection_counts.gyro_magnitude,
        rc_gyro_rms_abs = imu_diag.rejection_counts.gyro_rms_absolute,
        rc_gyro_rms_frac = imu_diag.rejection_counts.gyro_rms_fraction,
        rc_mono_few = imu_diag.rejection_counts.mono_not_enough_factors,
        rc_mono_underdet = imu_diag.rejection_counts.mono_underdetermined,
        rc_mono_illcond = imu_diag.rejection_counts.mono_ill_conditioned,
        rc_mono_degen = imu_diag.rejection_counts.mono_degenerate_solve,
        rc_mono_grav = imu_diag.rejection_counts.mono_gravity_norm,
        rc_mono_scale = imu_diag.rejection_counts.mono_scale_range,
        last_rejection = imu_diag
            .last_rejection
            .map(|r| format!("{r:?}"))
            .unwrap_or_else(|| "none".to_string()),
        sc_enabled = sc_diag.enabled,
        sc_weight = sc_diag.weight,
        sc_converged = sc_diag.converged,
        sc_recovered_scale = sc_diag.recovered_scale.unwrap_or(f64::NAN),
        sc_posterior_log_std = sc_diag.posterior_log_std.unwrap_or(f64::NAN),
        sc_bias_gyro_x = sc_diag.bias_gyro.x,
        sc_bias_gyro_y = sc_diag.bias_gyro.y,
        sc_bias_gyro_z = sc_diag.bias_gyro.z,
        sc_measurements_taken = sc_diag.measurements_taken,
        sc_measurements_rejected = sc_diag.measurements_rejected,
        sc_soft_rollback_count = sc_diag.soft_rollback_count,
        sc_rc_few = sc_diag.rejection_counts.not_enough_factors,
        sc_rc_underdet = sc_diag.rejection_counts.underdetermined,
        sc_rc_illcond = sc_diag.rejection_counts.ill_conditioned,
        sc_rc_degen = sc_diag.rejection_counts.degenerate_solve,
        sc_rc_grav = sc_diag.rejection_counts.gravity_norm,
        sc_rc_scale = sc_diag.rejection_counts.scale_range,
        sc_last_rejection = sc_diag
            .last_rejection
            .map(|r| format!("{r:?}"))
            .unwrap_or_else(|| "none".to_string()),
        lc_enabled = lc_diag.enabled,
        lc_batches = lc_diag.batches_attempted,
        lc_candidates = lc_diag.candidates_evaluated_total,
        lc_accepted = lc_diag.accepted_loops_total,
        lc_edges_added = lc_diag.patch_edges_added_total,
        lc_correction_events = lc_diag.correction_events,
        lc_correction_max = lc_diag.correction_magnitude_max_m,
        lc_correction_mean = lc_diag.correction_magnitude_mean_m,
        gba_enabled = gba_diag.enabled,
        gba_calls = gba_diag.calls,
        gba_inactive_retained = gba_diag.inactive_edges_retained,
        gba_inactive_evicted = gba_diag.inactive_edges_evicted_total,
        gba_last_free_pose_count = gba_diag.last_free_pose_count,
        gba_max_free_pose_count = gba_diag.max_free_pose_count,
        gba_last_edge_count = gba_diag.last_edge_count,
        gba_last_resolved_inactive = gba_diag.last_resolved_inactive_edges,
        gba_last_unresolved_inactive = gba_diag.last_unresolved_inactive_edges,
        gba_last_widened = gba_diag.last_t0_widened_by_loop_edge,
        gba_last_folded_included = gba_diag.last_folded_poses_included,
        gba_last_capped = gba_diag.last_free_pose_count_capped,
        gba_last_pose_delta_max = gba_diag.last_pose_delta_max_m,
        gba_last_pose_delta_mean = gba_diag.last_pose_delta_mean_m,
        gba_last_elapsed_ms = gba_diag.last_elapsed_ms,
        gba_total_elapsed_ms = gba_diag.total_elapsed_ms,
        s3b_enabled = s3b_diag.enabled,
        s3b_calls = s3b_diag.calls,
        s3b_loop_edges_total = s3b_diag.loop_edges_total,
        s3b_last_node_count = s3b_diag.last_node_count,
        s3b_last_edge_count = s3b_diag.last_edge_count,
        s3b_last_loop_edges_used = s3b_diag.last_loop_edges_used,
        s3b_last_corrections = s3b_diag.last_scale_corrections_applied,
        s3b_last_pose_delta_max = s3b_diag.last_pose_delta_max_m,
        s3b_last_pose_delta_mean = s3b_diag.last_pose_delta_mean_m,
        s3b_last_scale_min = s3b_diag.last_scale_min,
        s3b_last_scale_max = s3b_diag.last_scale_max,
        s3b_max_abs_log_scale_correction = args.s3b_max_abs_log_scale_correction,
        s3b_max_committed_abs_log_scale = s3b_diag.max_committed_abs_log_scale,
        s3b_scale_jump_rejections_total = s3b_diag.scale_jump_rejections_total,
        s3b_last_committed = s3b_diag.last_committed,
        s3b_last_rejection = s3b_diag.last_rejection,
        s3b_last_elapsed_ms = s3b_diag.last_elapsed_ms,
        s3b_total_elapsed_ms = s3b_diag.total_elapsed_ms,
        ll_enabled = ll_diag.enabled,
        ll_frames_indexed = ll_diag.frames_indexed,
        ll_vocab_built = ll_diag.vocab_built,
        ll_estimated_bytes = ll_diag.estimated_index_bytes,
        ll_queries_attempted = ll_diag.queries_attempted,
        ll_queries_issued_total = ll_diag.queries_issued_total,
        ll_queries_zero_candidates = ll_diag.queries_with_zero_candidates,
        ll_candidates_considered = ll_diag.candidates_considered,
        ll_verification_attempts = ll_diag.verification_attempts,
        ll_bridge_sufficient = ll_diag.bridge_sufficient_total,
        ll_rejected_rotation = ll_diag.rejected_rotation_inconsistent_total,
        ll_accepted_total = ll_diag.accepted_total,
        ll_rejected_bridge = ll_diag.rejected_insufficient_bridge_total,
        ll_rejected_ransac = ll_diag.rejected_ransac_total,
        ll_last_arrival_i = ll_diag.last_accepted_arrival_i,
        ll_last_arrival_j = ll_diag.last_accepted_arrival_j,
        ll_last_gap = ll_diag.last_accepted_gap,
        ll_last_similarity = ll_diag.last_accepted_similarity,
        ll_last_scale = ll_diag.last_accepted_scale,
        ll_last_inliers = ll_diag.last_accepted_inliers,
        ll_last_residual_ratio = ll_diag.last_accepted_mean_residual_ratio,
        ll_total_elapsed_ms = ll_diag.total_elapsed_ms,
        ll_sp_anchored = args.ll_sp_anchored_patches,
        ll_sp_min_separation = args.ll_sp_patch_min_separation,
        ll_max_rot_inconsistency = args.ll_max_rotation_inconsistency_deg,
        ll_query_log_len = odometry.long_loop_query_log().len(),
        ll_stage2_enabled = ll_diag.stage2_enabled,
        ll_stage2_attempts = ll_diag.stage2_attempts_total,
        ll_stage2_passed = ll_diag.stage2_passed_total,
        ll_stage2_rej_matches = ll_diag.stage2_rejected_insufficient_matches_total,
        ll_stage2_rej_inliers = ll_diag.stage2_rejected_insufficient_inliers_total,
        ll_stage2_rej_coverage = ll_diag.stage2_rejected_insufficient_coverage_total,
        ll_stage2_rej_rotation = ll_diag.stage2_rejected_rotation_inconsistent_total,
        ll_stage2_rej_residual = ll_diag.stage2_rejected_high_residual_total,
        ll_stage2_rej_umeyama_vs_e = ll_diag.stage2_rejected_umeyama_vs_e_rotation_total,
        hf_enabled = hf_diag.enabled,
        hf_regime_active = hf_diag.regime_active,
        hf_times_entered = hf_diag.times_entered,
        hf_times_exited = hf_diag.times_exited,
        hf_frames_suppressed = hf_diag.frames_suppressed_total,
        hf_disarmed = hf_diag.disarmed,
        hf_last_flow = hf_diag.last_flow,
        hf_last_enter_frame = hf_diag
            .last_enter_frame
            .map(|f| f.to_string())
            .unwrap_or_else(|| "none".to_string()),
        hf_last_exit_frame = hf_diag
            .last_exit_frame
            .map(|f| f.to_string())
            .unwrap_or_else(|| "none".to_string()),
        hf_window = args.hover_window,
        hf_enter_flow = args.hover_enter_flow,
        hf_exit_flow = args.hover_exit_flow,
        hf_response = hf_diag.response,
        hf_depth_damp_factor = args.hover_depth_damp_factor,
        hf_unflag_after_commits = args.hover_unflag_after_commits,
        hf_release_duration_commits = args.hover_release_duration_commits,
        hf_release_start_cap_frames = args.hover_release_start_cap_frames,
        hf_currently_damped_frames = hf_diag.currently_damped_frames,
        hf_frames_flagged_total = hf_diag.frames_flagged_total,
        hf_patches_flagged_total = hf_diag.patches_flagged_total,
        hf_unflagged_total = hf_diag.unflagged_total,
        hf_damped_solve_count = hf_diag.damped_solve_count,
        hf_currently_releasing_frames = hf_diag.currently_releasing_frames,
        hf_release_started_total = hf_diag.release_started_total,
        hf_release_start_max = hf_diag.max_release_started_per_advance,
        hf_release_histogram_frames = hf_diag.release_histogram_frames,
    );
    println!("{summary}");
    fs::write(args.out_dir.join("summary.txt"), &summary)?;

    // Milestone M12 (`docs/dpvo_droid_port_plan.md`, open item 2 carried
    // forward from M11): dump EVERY top-K long-range retrieval candidate
    // this run ever surfaced (accepted or not) to a CSV — the data needed to
    // answer "was the tightest GT revisit (i=42, j=456 per the M11 GT
    // precheck) ever even surfaced as a candidate", which M11's own
    // diagnostics (last-accepted-only) could not answer.
    if args.long_loop {
        let query_log = odometry.long_loop_query_log();
        // A3 stage 2, first slice (`docs/visual_slam_sequential_sfm_plan.md`):
        // Stage-2 diagnostic columns are appended at the end, so
        // `scripts/eval_dpvo_long_loop_recall.py`'s existing `csv.DictReader`-
        // based loader (which only checks its own required column SUBSET is
        // present, ignoring anything extra) keeps working unmodified.
        let mut csv = String::from(
            "query_arrival,rank,candidate_arrival,gap,similarity,accepted,rotation_disagreement_deg,\
             stage2_2d2d_inliers,stage2_e_rotation_disagreement_deg,stage2_e_rotation_qw,\
             stage2_e_rotation_qx,stage2_e_rotation_qy,stage2_e_rotation_qz,stage2_model,\
             stage2_h_inliers,stage2_h_rotation_disagreement_deg,stage2_diagnostic_umeyama_scale,\
             stage2_diagnostic_umeyama_inliers,stage2_umeyama_vs_e_rotation_deg,stage2_pnp_correspondences,\
             stage2_pnp_inliers,stage2_pnp_mean_reprojection_error,stage2_pnp_vs_e_rotation_deg,\
             stage2_pnp_scale_ratio,stage_reached,final_accepted\n",
        );
        for entry in query_log {
            let rot = entry
                .rotation_disagreement_deg
                .map(|d| format!("{d:.3}"))
                .unwrap_or_default();
            let stage2_inliers = entry
                .stage2_2d2d_inliers
                .map(|n| n.to_string())
                .unwrap_or_default();
            let stage2_e_rot = entry
                .stage2_e_rotation_disagreement_deg
                .map(|d| format!("{d:.3}"))
                .unwrap_or_default();
            let stage2_e_quat = entry.stage2_e_rotation_wxyz.unwrap_or([f64::NAN; 4]);
            let format_quaternion_component = |value: f64| {
                if value.is_finite() {
                    format!("{value:.9}")
                } else {
                    String::new()
                }
            };
            let stage2_h_inliers = entry
                .stage2_h_inliers
                .map(|n| n.to_string())
                .unwrap_or_default();
            let stage2_h_rot = entry
                .stage2_h_rotation_disagreement_deg
                .map(|d| format!("{d:.3}"))
                .unwrap_or_default();
            let diagnostic_scale = entry
                .stage2_diagnostic_umeyama_scale
                .map(|scale| format!("{scale:.9}"))
                .unwrap_or_default();
            let diagnostic_inliers = entry
                .stage2_diagnostic_umeyama_inliers
                .map(|n| n.to_string())
                .unwrap_or_default();
            let umeyama_vs_e = entry
                .stage2_umeyama_vs_e_rotation_deg
                .map(|d| format!("{d:.3}"))
                .unwrap_or_default();
            let pnp_correspondences = entry
                .stage2_pnp_correspondences
                .map(|n| n.to_string())
                .unwrap_or_default();
            let pnp_inliers = entry
                .stage2_pnp_inliers
                .map(|n| n.to_string())
                .unwrap_or_default();
            let pnp_reprojection = entry
                .stage2_pnp_mean_reprojection_error
                .map(|value| format!("{value:.6}"))
                .unwrap_or_default();
            let pnp_vs_e = entry
                .stage2_pnp_vs_e_rotation_deg
                .map(|value| format!("{value:.3}"))
                .unwrap_or_default();
            let pnp_scale = entry
                .stage2_pnp_scale_ratio
                .map(|value| format!("{value:.9}"))
                .unwrap_or_default();
            csv.push_str(&format!(
                "{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                entry.query_arrival,
                entry.rank,
                entry.candidate_arrival,
                entry.gap,
                entry.similarity,
                entry.accepted,
                rot,
                stage2_inliers,
                stage2_e_rot,
                format_quaternion_component(stage2_e_quat[0]),
                format_quaternion_component(stage2_e_quat[1]),
                format_quaternion_component(stage2_e_quat[2]),
                format_quaternion_component(stage2_e_quat[3]),
                entry.stage2_model,
                stage2_h_inliers,
                stage2_h_rot,
                diagnostic_scale,
                diagnostic_inliers,
                umeyama_vs_e,
                pnp_correspondences,
                pnp_inliers,
                pnp_reprojection,
                pnp_vs_e,
                pnp_scale,
                entry.stage_reached,
                entry.final_accepted,
            ));
        }
        // A3 stage-1 (`docs/visual_slam_sequential_sfm_plan.md`, "densify
        // query cadence" slice): an issued query that surfaced ZERO
        // candidates would otherwise never appear in this CSV at all — a
        // dense `query_frequency` looks identical to "never issued" without
        // this. One marker row per empty query, using the sentinel values
        // the task itself specifies; the header is unchanged, so existing
        // consumers that only read `rank >= 0` rows are unaffected.
        let empty_arrivals = odometry.long_loop_empty_query_arrivals();
        for &arrival in empty_arrivals {
            csv.push_str(&format!(
                "{arrival},-1,-1,-1,0.000000,false,,,,,,,,not_run,,,,,,,,,,,no_candidates,false\n"
            ));
        }
        let csv_path = args.out_dir.join("long_loop_candidates.csv");
        fs::write(&csv_path, &csv)?;
        println!(
            "wrote {} long-range candidate log entries ({} empty-query markers) to {}",
            query_log.len() + empty_arrivals.len(),
            empty_arrivals.len(),
            csv_path.display()
        );
    }

    // Milestone M14: dump every frame the low-parallax detector was
    // evaluated on — the acceptance evidence for "did the regime enter/exit
    // at the right place, for the right duration" (fed straight into
    // `m13_scale_profile.py`-style profiling alongside the trajectory CSV).
    if args.hover_freeze {
        let flow_log = odometry.low_parallax_flow_log();
        let mut csv = String::from("frames_processed,flow,regime_active\n");
        for &(frame, flow, active) in flow_log {
            csv.push_str(&format!("{frame},{flow:.6},{active}\n"));
        }
        let csv_path = args.out_dir.join("hover_flow_trace.csv");
        fs::write(&csv_path, &csv)?;
        println!(
            "wrote {} hover flow-trace entries to {}",
            flow_log.len(),
            csv_path.display()
        );
    }

    // A3 ranking-lab offline dump (`docs/visual_slam_sequential_sfm_plan.md`,
    // "A3 — Sound long-range loop closure", ranking slice A): `manifest.csv`
    // (arrival index, keypoint count, descriptor dim, the two `.npy`
    // filenames `dump_long_loop_frame` already wrote per frame) plus a
    // `README.md` documenting the on-disk format, so
    // `scripts/eval_dpvo_retrieval_ranking_offline.py` can load everything
    // with `numpy.load(...)` alone — no framework/Rust re-run needed to try
    // a new ranking method.
    if let Some(dump_dir) = args.ll_dump_frame_descriptors.as_ref() {
        let mut manifest = String::from(
            "arrival_index,keypoint_count,descriptor_dim,keypoints_file,descriptors_file\n",
        );
        for &(arrival_index, keypoint_count, descriptor_dim) in &ll_dump_manifest {
            manifest.push_str(&format!(
                "{arrival_index},{keypoint_count},{descriptor_dim},{arrival_index:06}_keypoints.npy,{arrival_index:06}_descriptors.npy\n",
            ));
        }
        let manifest_path = dump_dir.join("manifest.csv");
        fs::write(&manifest_path, &manifest)?;

        let readme = format!(
            "# A3 ranking-lab offline frame-descriptor dump\n\n\
             Written by `examples/euroc_dpvo_vo_demo.rs --ll-dump-frame-descriptors <dir>` \
             (`docs/visual_slam_sequential_sfm_plan.md`, \"A3 — Sound long-range loop \
             closure\", ranking slice A). One row per arrival index that \
             `crate::dpvo_long_loop::DpvoLongLoopIndex::ingest_frame` actually indexed \
             during this run (i.e. skips frames whose SuperPoint extraction failed or \
             returned zero keypoints — see that function's own `descriptors.is_empty()` \
             early return).\n\n\
             ## Files\n\n\
             - `manifest.csv`: one row per ingested arrival —\n\
             \x20 `arrival_index,keypoint_count,descriptor_dim,keypoints_file,descriptors_file`.\n\
             - `<arrival_index:06>_keypoints.npy`: bare `.npy` (numpy `<f4`, C-order), shape \
             `(keypoint_count, 2)`, columns `(x, y)` in PATCH-GRID coordinates — the raw \
             SuperPoint keypoint pixel coordinates already divided by `RES` (this crate's \
             DPVO downsample stride), the EXACT space `crate::dpvo_patch_ba::DpvoPatch::x`/`y` \
             and `DpvoLongLoopIndex`'s own internal `keypoints` field live in. NOT full-resolution \
             image pixels.\n\
             - `<arrival_index:06>_descriptors.npy`: bare `.npy` (numpy `<f4`, C-order), shape \
             `(keypoint_count, descriptor_dim)` — row `i` is the SuperPoint descriptor for \
             keypoint row `i` of the matching `_keypoints.npy` (same ordering, same count).\n\n\
             ## Loading in Python\n\n\
             ```python\n\
             import numpy as np\n\
             import csv\n\n\
             manifest = list(csv.DictReader(open(\"manifest.csv\")))\n\
             row = manifest[0]\n\
             keypoints = np.load(row[\"keypoints_file\"])      # (N, 2) float32\n\
             descriptors = np.load(row[\"descriptors_file\"])  # (N, D) float32\n\
             ```\n\n\
             ## Why bare `.npy`, not `.npz`\n\n\
             `crates/vision/src/dpvo/npz.rs` only ever needed to READ `.npz` fixtures before \
             this dump existed; writing a real `.npz` (ZIP) would require computing correct \
             per-entry CRC32 values (Python's `zipfile` verifies them on read) for no benefit \
             over one bare `.npy` per array, which `numpy.load(path)` already reads directly \
             with zero extra code. See `write_npy_f32`'s own doc in that module for the exact \
             format (numpy `.npy` v1.0: `\\x93NUMPY` magic, version, header dict, raw \
             little-endian `<f4` data).\n"
        );
        let readme_path = dump_dir.join("README.md");
        fs::write(&readme_path, &readme)?;

        println!(
            "wrote {} frame-descriptor dumps ({} manifest rows) to {}",
            ll_dump_manifest.len(),
            ll_dump_manifest.len(),
            dump_dir.display()
        );
    }

    println!(
        "wrote {} and summary.txt to {}",
        traj_path.display(),
        args.out_dir.display()
    );
    Ok(())
}
