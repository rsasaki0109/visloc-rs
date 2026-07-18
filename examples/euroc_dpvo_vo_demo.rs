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
    DpvoGlobalBaConfig, DpvoImuConfig, DpvoOdometry, DpvoOdometryConfig, DpvoScaleCouplingConfig,
};
use visloc_rs::slam::{
    DpvoIntrinsics, DpvoLongLoopConfig, DpvoLoopClosureConfig, DpvoSim3BackendConfig, ImuNoiseModel,
    ScaleCouplingConfig,
};
use visloc_rs::vision::distortion::RadialTangential;
use visloc_rs::vision::features::superpoint_onnx::OnnxBackend;
use visloc_rs::{umeyama_similarity_transform, TrajectorySimilarityTransform};

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
    ll_query_frequency: usize,
    ll_top_k: usize,
    ll_min_similarity: f32,
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
            long_loop: false,
            ll_superpoint_model: PathBuf::new(),
            // Mirror `DpvoLongLoopConfig::default()` exactly so omitting
            // these flags reproduces that struct's own defaults.
            ll_vocab_bootstrap_frames: DpvoLongLoopConfig::default().vocab_bootstrap_frames,
            ll_vocab_words: DpvoLongLoopConfig::default().vocab_words,
            ll_query_frequency: DpvoLongLoopConfig::default().query_frequency,
            ll_top_k: DpvoLongLoopConfig::default().top_k,
            ll_min_similarity: DpvoLongLoopConfig::default().min_similarity,
            ll_min_temporal_gap: DpvoLongLoopConfig::default().min_temporal_gap,
            ll_max_indexed_frames: DpvoLongLoopConfig::default().max_indexed_frames,
            ll_patch_pixel_radius: DpvoLongLoopConfig::default().patch_pixel_radius,
            ll_min_bridge_correspondences: DpvoLongLoopConfig::default().min_bridge_correspondences,
            ll_ransac_iterations: DpvoLongLoopConfig::default().ransac_iterations,
            ll_min_ransac_inliers: DpvoLongLoopConfig::default().min_ransac_inliers,
            ll_max_mean_residual_ratio: DpvoLongLoopConfig::default().max_mean_residual_ratio,
            ll_sp_anchored_patches: DpvoLongLoopConfig::default().sp_anchored_patches,
            ll_sp_patch_min_separation: DpvoLongLoopConfig::default().sp_patch_min_separation,
            ll_max_rotation_inconsistency_deg: DpvoLongLoopConfig::default().max_rotation_inconsistency_deg,
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
            "--imu" => {
                args.imu = true;
                raw.remove(i);
                continue;
            }
            "--imu-gravity-norm-deviation-ratio" => {
                args.imu_gravity_norm_deviation_ratio = raw.remove(i + 1).parse()?
            }
            "--imu-min-bootstrap-factors" => args.imu_min_bootstrap_factors = raw.remove(i + 1).parse()?,
            "--imu-noise-scale" => args.imu_noise_scale = raw.remove(i + 1).parse()?,
            "--imu-max-gyro-bias-magnitude-rad-s" => {
                args.imu_max_gyro_bias_magnitude_rad_s = raw.remove(i + 1).parse()?
            }
            "--imu-gyro-bias-max-rms-after" => args.imu_gyro_bias_max_rms_after = raw.remove(i + 1).parse()?,
            "--imu-gyro-bias-max-rms-fraction" => {
                args.imu_gyro_bias_max_rms_fraction = raw.remove(i + 1).parse()?
            }
            "--imu-min-mono-scale" => args.imu_min_mono_scale = raw.remove(i + 1).parse()?,
            "--imu-max-mono-scale" => args.imu_max_mono_scale = raw.remove(i + 1).parse()?,
            "--imu-max-mono-alignment-condition-number" => {
                args.imu_max_mono_alignment_condition_number = raw.remove(i + 1).parse()?
            }
            "--imu-rollback-mean-nis-bound" => args.imu_rollback_mean_nis_bound = raw.remove(i + 1).parse()?,
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
            "--lc-max-edges-per-batch" => args.lc_max_edges_per_batch = raw.remove(i + 1).parse()?,
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
            "--long-loop" => {
                args.long_loop = true;
                raw.remove(i);
                continue;
            }
            "--ll-superpoint-model" => args.ll_superpoint_model = PathBuf::from(raw.remove(i + 1)),
            "--ll-vocab-bootstrap-frames" => args.ll_vocab_bootstrap_frames = raw.remove(i + 1).parse()?,
            "--ll-vocab-words" => args.ll_vocab_words = raw.remove(i + 1).parse()?,
            "--ll-query-frequency" => args.ll_query_frequency = raw.remove(i + 1).parse()?,
            "--ll-top-k" => args.ll_top_k = raw.remove(i + 1).parse()?,
            "--ll-min-similarity" => args.ll_min_similarity = raw.remove(i + 1).parse()?,
            "--ll-min-temporal-gap" => args.ll_min_temporal_gap = raw.remove(i + 1).parse()?,
            "--ll-max-indexed-frames" => args.ll_max_indexed_frames = raw.remove(i + 1).parse()?,
            "--ll-patch-pixel-radius" => args.ll_patch_pixel_radius = raw.remove(i + 1).parse()?,
            "--ll-min-bridge-correspondences" => args.ll_min_bridge_correspondences = raw.remove(i + 1).parse()?,
            "--ll-ransac-iterations" => args.ll_ransac_iterations = raw.remove(i + 1).parse()?,
            "--ll-min-ransac-inliers" => args.ll_min_ransac_inliers = raw.remove(i + 1).parse()?,
            "--ll-max-mean-residual-ratio" => args.ll_max_mean_residual_ratio = raw.remove(i + 1).parse()?,
            "--ll-sp-anchored-patches" => {
                args.ll_sp_anchored_patches = true;
                raw.remove(i);
                continue;
            }
            "--ll-sp-patch-min-separation" => args.ll_sp_patch_min_separation = raw.remove(i + 1).parse()?,
            "--ll-max-rotation-inconsistency-deg" => args.ll_max_rotation_inconsistency_deg = raw.remove(i + 1).parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
        raw.remove(i);
    }
    args.euroc_dir = euroc_dir.ok_or("--euroc-dir <path/to/MH_01_easy> is required")?;
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
    graph.frames().iter().find(|f| f.arrival_index == arrival_index).map(|f| f.pose.clone())
}

fn nearest_ground_truth(samples: &[EurocGroundTruthSample], target_ts: i128) -> Option<&EurocGroundTruthSample> {
    if samples.is_empty() {
        return None;
    }
    let idx = samples.binary_search_by_key(&target_ts, |s| s.timestamp_nanoseconds).unwrap_or_else(|insert| {
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

    let (width, height) = (dataset.cam0_calibration.resolution.0 as usize, dataset.cam0_calibration.resolution.1 as usize);
    let intrinsics = dataset.cam0_calibration.intrinsics;
    let distortion = RadialTangential::from_euroc_coefficients(&dataset.cam0_calibration.distortion_coefficients)
        .unwrap_or(RadialTangential::IDENTITY);

    let backend = if args.onnx_cpu { OnnxBackend::Cpu } else { OnnxBackend::default() };
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
        intrinsics: DpvoIntrinsics { fx: intrinsics[0], fy: intrinsics[1], cx: intrinsics[2], cy: intrinsics[3] },
        ba_lmbda: 1.0e-4,
        ba_ep: 100.0,
        motion_probe_min_flow: 2.0,
        seed: args.seed,
        // Milestone M5 (`docs/dpvo_droid_port_plan.md`): `--imu` couples
        // `mav0/imu0/data.csv` into the joint solve via
        // `crate::dpvo_vi_ba`; omitting the flag reproduces M4/M4-perf's
        // visual-only behavior exactly (`imu: None`).
        imu: args.imu.then(|| DpvoImuConfig {
            body_to_camera: se3_from_t_bs(&dataset.cam0_calibration.t_body_sensor),
            noise: ImuNoiseModel {
                gyroscope_noise_density: dataset.imu_calibration.gyroscope_noise_density * args.imu_noise_scale,
                accelerometer_noise_density: dataset.imu_calibration.accelerometer_noise_density * args.imu_noise_scale,
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
            ..DpvoSim3BackendConfig::default()
        }),
        // Milestone M11 (`docs/dpvo_droid_port_plan.md`): `--long-loop`
        // enables the long-range appearance loop-candidate source; omitting
        // it reproduces M4-M10's exact behavior (`long_loop: None`, no
        // SuperPoint inference at all).
        long_loop: args.long_loop.then_some(DpvoLongLoopConfig {
            vocab_bootstrap_frames: args.ll_vocab_bootstrap_frames,
            vocab_words: args.ll_vocab_words,
            query_frequency: args.ll_query_frequency,
            top_k: args.ll_top_k,
            min_similarity: args.ll_min_similarity,
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
            ..DpvoLongLoopConfig::default()
        }),
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
            if args.gba_max_free_poses == 0 { "none".to_string() } else { args.gba_max_free_poses.to_string() },
        );
    }

    if args.sim3_backend {
        println!(
            "sim3 backend enabled (Milestone M9): frequency={} node_stride={} loop_edge_weight={:.2}",
            args.s3b_frequency, args.s3b_node_stride, args.s3b_loop_edge_weight,
        );
    }

    if args.long_loop {
        println!(
            "long-range loop enabled (Milestone M11): superpoint_model={} vocab_bootstrap_frames={} \
             vocab_words={} query_frequency={} top_k={} min_similarity={:.3} min_temporal_gap={} \
             max_indexed_frames={} patch_pixel_radius={:.2} min_bridge_correspondences={} \
             ransac_iterations={} min_ransac_inliers={} max_mean_residual_ratio={:.3} \
             sp_anchored_patches={} (Milestone M12) sp_patch_min_separation={:.2} \
             max_rotation_inconsistency_deg={:.1}",
            args.ll_superpoint_model.display(),
            args.ll_vocab_bootstrap_frames,
            args.ll_vocab_words,
            args.ll_query_frequency,
            args.ll_top_k,
            args.ll_min_similarity,
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
    }

    if args.imu {
        println!(
            "imu enabled: samples={} gyro_noise_density={:.6e} accel_noise_density={:.6e} \
             body_to_camera_t=[{:.4},{:.4},{:.4}]",
            dataset.imu_samples.len(),
            dataset.imu_calibration.gyroscope_noise_density,
            dataset.imu_calibration.accelerometer_noise_density,
            odometry_config.imu.as_ref().unwrap().body_to_camera.translation.x,
            odometry_config.imu.as_ref().unwrap().body_to_camera.translation.y,
            odometry_config.imu.as_ref().unwrap().body_to_camera.translation.z,
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
        args.model_dir.join("fixtures").join("softagg_weights_fixture.npz"),
        backend,
        superpoint_model_path,
    )?;

    let frame_cap = if args.max_frames == 0 { usize::MAX } else { args.max_frames };
    let frames: Vec<_> = dataset.cam0_images.iter().step_by(args.stride.max(1)).take(frame_cap).collect();
    println!("processing {} frames (stride={})", frames.len(), args.stride);

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
                && dataset.imu_samples[imu_cursor].timestamp_nanoseconds <= entry.timestamp_nanoseconds
            {
                let sample: &EurocImuSample = &dataset.imu_samples[imu_cursor];
                odometry.push_imu(sample.timestamp_nanoseconds as f64 * 1.0e-9, sample.gyro, sample.accel);
                imu_cursor += 1;
            }
        }

        let pose = odometry.process_frame(undistorted.view(), timestamp_seconds)?;

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
                    sc_diag.soft_rollback_count, sc_diag.weight, sc_diag.measurements_taken,
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
                     pose_delta_mean_m={:.4} scale_min={:.4} scale_max={:.4} elapsed_ms={:.2})",
                    s3b_diag.calls,
                    s3b_diag.last_node_count,
                    s3b_diag.last_edge_count,
                    s3b_diag.last_loop_edges_used,
                    s3b_diag.last_scale_corrections_applied,
                    s3b_diag.last_pose_delta_max_m,
                    s3b_diag.last_pose_delta_mean_m,
                    s3b_diag.last_scale_min,
                    s3b_diag.last_scale_max,
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
                     s3b_last_scale_min={:.4} s3b_last_scale_max={:.4} s3b_total_elapsed_ms={:.2}",
                    s3b_diag.calls,
                    s3b_diag.loop_edges_total,
                    s3b_diag.last_node_count,
                    s3b_diag.last_edge_count,
                    s3b_diag.last_scale_corrections_applied,
                    s3b_diag.last_pose_delta_max_m,
                    s3b_diag.last_scale_min,
                    s3b_diag.last_scale_max,
                    s3b_diag.total_elapsed_ms,
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
    let ms_per_frame = if stats.frames_processed > 0 { total_elapsed_s * 1000.0 / stats.frames_processed as f64 } else { 0.0 };

    let (ate_rigid_rmse, ate_rigid_max, ate_sim_rmse, ate_sim_max, ate_sim_scale) = if aligned_estimated.len() >= 3 {
        let rigid = umeyama_similarity_transform(&aligned_estimated, &aligned_reference, false)
            .unwrap_or_else(TrajectorySimilarityTransform::identity);
        let similarity = umeyama_similarity_transform(&aligned_estimated, &aligned_reference, true)
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

    let tracked_fraction = if !frames.is_empty() { tracked_frames as f64 / frames.len() as f64 } else { 0.0 };
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

    let summary = format!(
        "euroc_dir={}\n\
         model_dir={}\n\
         frames_requested={frame_count}\n\
         frames_tracked={tracked_frames}\n\
         tracked_fraction={tracked_fraction:.4}\n\
         total_elapsed_s={total_elapsed_s:.2}\n\
         ms_per_frame_total={ms_per_frame:.2}\n\
         ms_per_frame_io={io_ms:.2}\n\
         ms_per_frame_undistort={undistort_ms:.2}\n\
         ms_per_frame_encode={encode_ms:.2}\n\
         ms_per_frame_correlation={corr_ms:.2}\n\
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
         sim3_backend_last_elapsed_ms={s3b_last_elapsed_ms:.3}\n\
         sim3_backend_total_elapsed_ms={s3b_total_elapsed_ms:.3}\n\
         long_loop_enabled={ll_enabled}\n\
         long_loop_frames_indexed={ll_frames_indexed}\n\
         long_loop_vocab_built={ll_vocab_built}\n\
         long_loop_estimated_index_bytes={ll_estimated_bytes}\n\
         long_loop_queries_attempted={ll_queries_attempted}\n\
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
         long_loop_query_log_entries={ll_query_log_len}\n",
        args.euroc_dir.display(),
        args.model_dir.display(),
        frame_count = frames.len(),
        io_ms = io_ms_total / stats.frames_processed.max(1) as f64,
        undistort_ms = undistort_ms_total / stats.frames_processed.max(1) as f64,
        encode_ms = stats.encode_ms_total / stats.frames_processed.max(1) as f64,
        corr_ms = stats.correlation_ms_total / stats.frames_processed.max(1) as f64,
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
        s3b_last_elapsed_ms = s3b_diag.last_elapsed_ms,
        s3b_total_elapsed_ms = s3b_diag.total_elapsed_ms,
        ll_enabled = ll_diag.enabled,
        ll_frames_indexed = ll_diag.frames_indexed,
        ll_vocab_built = ll_diag.vocab_built,
        ll_estimated_bytes = ll_diag.estimated_index_bytes,
        ll_queries_attempted = ll_diag.queries_attempted,
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
        let mut csv =
            String::from("query_arrival,rank,candidate_arrival,gap,similarity,accepted,rotation_disagreement_deg\n");
        for entry in query_log {
            let rot = entry.rotation_disagreement_deg.map(|d| format!("{d:.3}")).unwrap_or_default();
            csv.push_str(&format!(
                "{},{},{},{},{:.6},{},{}\n",
                entry.query_arrival, entry.rank, entry.candidate_arrival, entry.gap, entry.similarity, entry.accepted, rot,
            ));
        }
        let csv_path = args.out_dir.join("long_loop_candidates.csv");
        fs::write(&csv_path, &csv)?;
        println!("wrote {} long-range candidate log entries to {}", query_log.len(), csv_path.display());
    }

    println!("wrote {} and summary.txt to {}", traj_path.display(), args.out_dir.display());
    Ok(())
}
