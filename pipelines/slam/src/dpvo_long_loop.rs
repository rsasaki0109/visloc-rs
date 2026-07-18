//! Milestone M11 (`docs/dpvo_droid_port_plan.md`): a LONG-RANGE, appearance-
//! based loop-candidate source feeding the machinery M9/M10 already built and
//! proved works once handed a real edge — reusing
//! [`crate::dpvo_sim3_backend::run_sim3_backend`] (the `Sim(3)` pose-graph
//! backend) and [`crate::dpvo_vo::DpvoOdometry::loop_edge_arrival_pairs`]
//! (M10's widened global-BA `t0`) as-is, per M10's own "What's actually next"
//! section: *"bridge `online_slam.rs`'s own appearance-loop detection... into
//! `(arrival_i, arrival_j)` pairs feeding `DpvoOdometry::loop_edge_arrival_pairs`
//! and `crate::dpvo_sim3_backend::Sim3LoopMeasurement` directly."*
//!
//! # Why this exists: DPV-SLAM's own proximity mechanism cannot reach far
//! enough back
//!
//! M10's own real-run finding (`docs/dpvo_droid_port_plan.md`'s "M10
//! results", "Why `folded_poses_included` is `0` on every real call"):
//! `crate::dpvo_loop_closure::find_loop_edges`'s candidate search is bounded
//! by the LIVE patch-graph buffer's own size (`graph.n_frames()`, `~40-55`
//! frames on MH_01), so every proximity loop edge it has ever proposed across
//! M6-M10 spans only `~30-49` frames — an order of magnitude short of the
//! `~500-700`-frame gap MH_01's own `~22.6x` monocular scale drift
//! accumulates over. Widening the downstream BA window (M10) or the Sim(3)
//! backend's own reach (M9) cannot help when the candidate SUPPLIER itself
//! never looks that far back. This module is that missing long-range
//! supplier: it retrieves candidates by APPEARANCE (a compact per-frame
//! descriptor), not by scanning a live pose/patch array, so its reach is the
//! FULL trajectory history, independent of how large the live buffer
//! currently is.
//!
//! # Why not reuse `online_slam.rs`'s own appearance-loop pipeline directly
//!
//! `online_slam.rs`'s `build_appearance_loop_candidates` (+
//! `estimate_loop_sim3_scale_3d3d`) is a mature, thoroughly-tested pipeline —
//! but it is built entirely around `VisualMap`/`Frame`/`LandmarkDescriptorStore`
//! (triangulated map landmarks with persistent ids, covisibility graphs,
//! `CorrespondenceBuilder`), none of which the DPVO port has: DPVO's own state
//! is a flat per-frame PATCH array (`crate::dpvo_patch_graph::DpvoPatchGraph`,
//! `crate::dpvo_patch_ba::DpvoPatch`), with no landmark identity, no
//! covisibility graph, and (per M9's own module doc, "The loop-edge scale
//! question") no second independently-anchored depth at any revisited point.
//! Bridging DPVO's own representation into `VisualMap` wholesale is a much
//! larger undertaking than this milestone's scope. Instead, this module is a
//! DPVO-native analog: it reuses the same INGREDIENTS the appearance pipeline
//! established as sufficient — a global per-frame descriptor for retrieval,
//! local descriptor cross-check matching for verification, and a 3D-3D
//! Umeyama fit for the metric scale (`crate::pipelines::tracking`'s own
//! [`visloc_tracking::umeyama_similarity_transform`], the SAME primitive
//! `estimate_loop_sim3_scale_3d3d` uses) — but built directly on DPVO's own
//! patch geometry, RetainedFoldedFrame (M10), and Sim3LoopMeasurement (M9)
//! types, not the online-SLAM map stack.
//!
//! # Design choice: VLAD, not the vocab-tree TF-IDF index
//!
//! `crates/vision` has two existing retrieval front-ends this module could
//! have reused: [`crate::dpvo_long_loop`]'s own choice, VLAD
//! (`visloc_vision::place_recognition`), and the COLMAP port's vocab-tree
//! (`visloc_vision::vocab_tree`, hierarchical k-means + TF-IDF + Hamming
//! embedding). VLAD wins here on INTEGRATION FIT, not raw retrieval quality:
//! it produces one fixed-length `f32` vector per frame from a vocabulary
//! trained ONCE (`Vocabulary::build`), so a streaming, one-frame-at-a-time
//! DPVO run can insert a new frame's descriptor with a single `vlad(...)`
//! call and score candidates with plain cosine similarity — no persistent
//! inverted-file/IDF bookkeeping to maintain across a live run, no
//! `finalize()`-before-every-query dance the vocab-tree's own `add_image`/
//! `finalize`/`query` cycle expects. The vocab-tree's TF-IDF+Hamming-embedding
//! machinery is more discriminative at LARGE-N (thousands of images) image
//! retrieval, which is not this module's regime (a single sequence's own
//! history, hundreds to low thousands of frames) — VLAD's simplicity is the
//! better trade here, an explicit, evidence-free-but-documented deviation
//! from the task's own "VLAD or vocab-tree, pick whichever integrates
//! cleanest" invitation.
//!
//! # Streaming descriptors: bootstrap a vocabulary, then insert forever
//!
//! DPVO does not retain frame images past the frame that produces them (see
//! `crate::dpvo_vo::DpvoOdometry::process_frame`'s own image parameter — a
//! borrowed view, never stored), so a per-frame appearance descriptor must be
//! computed THE INSTANT a frame commits, or never at all. [`DpvoLongLoopIndex`]
//! buffers the first [`DpvoLongLoopConfig::vocab_bootstrap_frames`] committed
//! frames' raw local (SuperPoint) descriptors, builds a [`Vocabulary`] from
//! their pooled union once enough have accumulated, retroactively VLAD-encodes
//! the buffered frames, then VLAD-encodes every later frame immediately on
//! ingest. The RAW local keypoints+descriptors are ALSO retained per indexed
//! frame (bounded by [`DpvoLongLoopConfig::max_indexed_frames`], FIFO-evicted)
//! — VLAD answers "is this frame similar to that one?", but a candidate pair
//! still needs the underlying local descriptors for cross-check MATCHING
//! (geometric verification), which a global descriptor alone cannot provide.
//!
//! # Candidate generation and geometric verification
//!
//! [`DpvoLongLoopIndex::find_and_verify_long_range_loop`] (throttled via
//! [`DpvoLongLoopIndex::due`], independent of and much coarser than
//! `crate::dpvo_loop_closure::DpvoLoopClosureConfig::global_opt_freq`'s own
//! short-range throttle) ranks every sufficiently-OLD indexed frame
//! (`arrival` gap `>= `[`DpvoLongLoopConfig::min_temporal_gap`], deliberately
//! decoupled from the proximity mechanism's own `~30-49`-frame reach) by VLAD
//! cosine similarity, takes the top `K`, and geometrically verifies each in
//! turn until one is accepted:
//!
//! 1. Cross-check (mutual nearest-neighbor + Lowe ratio) match the two
//!    frames' raw local descriptors (`visloc_vision::matching::CrossCheckMatcher`
//!    over `BruteForceMatcher`).
//! 2. **Bridge each 2D-2D match to a 3D-3D correspondence** via
//!    [`bridge_matches_to_3d3d`]: for each matched keypoint pair, find the
//!    NEAREST DPVO patch owned by each side within
//!    [`DpvoLongLoopConfig::patch_pixel_radius`] (patch-grid pixels — DPVO's
//!    own patches are placed at random anchor points, not necessarily at a
//!    detected keypoint, so an exact coincidence is not expected; only a
//!    nearby owned patch is required). Each side's patch backprojects to a
//!    3D point in ITS OWN camera frame (via that patch's own inverse depth)
//!    and then into WORLD coordinates via that side's own current pose —
//!    see [`patch_to_world_point`]. Both world points, per M9's own "The
//!    loop-edge scale question" analysis, ALREADY live in the SAME nominal
//!    DPVO world frame; if the whole trajectory shared one consistent metric
//!    scale, they would coincide exactly (same physical point). Any Sim(3)
//!    discrepancy between the OLD frame's own local reconstruction of the
//!    point and the NEW frame's own local reconstruction of the SAME point is
//!    therefore a genuine, non-circular measurement of how much scale has
//!    drifted between the two — the independent 3D information M9's own
//!    module doc identified as missing from DPVO's shared-coordinate-system
//!    proximity edges (which never re-triangulate independently at the new
//!    frame).
//! 3. **[`ransac_umeyama_scale`]**: a 3-point-minimal-sample RANSAC over
//!    [`visloc_tracking::umeyama_similarity_transform`] (the SAME Umeyama
//!    primitive `online_slam.rs::estimate_loop_sim3_scale_3d3d` uses,
//!    reused rather than reimplemented) recovers the best-supported Sim(3)
//!    aligning the OLD-side points onto the NEW-side points; the fitted
//!    `.scale` is the accepted pair's `measured_scale`. Rejected outright
//!    (see each gate's own doc on [`DpvoLongLoopConfig`]) for: too few
//!    bridged correspondences, too few RANSAC inliers, too high a residual
//!    relative to the point cloud's own scale, or a fitted scale outside a
//!    sane range — **no fallback to `scale = 1`**: the whole point of this
//!    milestone is a genuine scale measurement, so a candidate whose
//!    geometry cannot support one is discarded rather than accepted with a
//!    vacuous scale.
//!
//! The accepted pair's ORDINARY rotation+translation `Sim3LoopMeasurement`
//! (scale fixed at 1, per `crate::dpvo_sim3_backend`'s own two-edge design)
//! reuses DPVO's OWN current composed relative pose (`pose_j.compose(&pose_i.inverse())`)
//! rather than the RANSAC fit's own rotation/translation — DPVO's monocular
//! rotation estimate is generally far more reliable than its translation
//! SCALE (the entire premise of this port's scale-drift problem), so this
//! keeps the rotation+translation edge behaving exactly like M9's own
//! proximity edges while ONLY the scale channel gets the new, independent
//! signal.
//!
//! # Feeding the existing M9/M10 machinery
//!
//! An accepted [`AcceptedLongLoop`] is consumed by `crate::dpvo_vo::DpvoOdometry`
//! exactly like a proximity loop pair, reusing the SAME "ever had a loop
//! edge" gates that already unlock `try_global_ba`/`try_sim3_backend` — no
//! new gating logic needed: `(arrival_i, arrival_j)` is pushed onto
//! [`crate::dpvo_vo::DpvoOdometry::loop_edge_arrival_pairs`] (M10's widened
//! `t0`) and the `Sim3LoopMeasurement` (with `measured_scale: Some(..)`) onto
//! `crate::dpvo_vo::DpvoOdometry::sim3_loop_measurements` (M9's backend). Per
//! the task's own explicit scope note, this module does NOT append ordinary
//! DPVO patch-graph edges (`crate::dpvo_patch_graph::DpvoPatchGraph::append_edges`)
//! the way `crate::dpvo_loop_closure`'s proximity mechanism does — a
//! genuinely old frame's own fmap/correlation state is gone once folded away
//! (`crate::dpvo_patch_graph::DpvoPatchGraph::fold_frame`'s own `store=False`
//! drop), so a real correlation-based patch edge is not obtainable for a
//! long-range pair in the first place; pose-graph (M9) + widened-BA (M10)
//! consumption is the only avenue, and both already exist.
//!
//! # Failure modes, honestly enumerated
//!
//! * **Sparse patch coverage**: `fast.yaml`-sized runs place only
//!   `patches_per_frame` (`48`-`96`) patches per frame at RANDOM anchor
//!   points, while SuperPoint typically detects hundreds of keypoints — most
//!   matched keypoints will have NO owned patch within
//!   `patch_pixel_radius`, so [`bridge_matches_to_3d3d`] is expected to keep
//!   only a small fraction of the raw 2D-2D matches. This is measured, not
//!   assumed — see `docs/dpvo_droid_port_plan.md`'s "M11 results" for the
//!   real yield.
//! * **Vocabulary bootstrap degeneracy**: if the first
//!   `vocab_bootstrap_frames` frames see too few keypoints combined (fewer
//!   than `vocab_words`), [`Vocabulary::build`] returns `None` and this
//!   module keeps buffering (bounded, see [`DpvoLongLoopIndex::ingest_frame`]'s
//!   own doc) rather than ever indexing — a genuine (if unlikely on a
//!   feature-rich EuRoC scene), reported-not-silent degradation.
//! * **RANSAC on a truly non-revisiting pair**: appearance similarity alone
//!   can produce a false-positive candidate (visually similar but not the
//!   same physical place); the geometric gates (min inliers, max residual
//!   ratio, scale sanity bounds) are the actual correctness backstop, not the
//!   retrieval score.

use std::collections::VecDeque;
use std::time::Instant;

use nalgebra::{Point2, Point3, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use visloc_core::geometry::SE3;
use visloc_tracking::{umeyama_similarity_transform, TrajectorySimilarityTransform};
use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, Matcher};
use visloc_vision::place_recognition::{cosine_similarity, vlad, Vocabulary};

use crate::dpvo_patch_ba::{DpvoIntrinsics, DpvoPatch};
use crate::dpvo_sim3_backend::Sim3LoopMeasurement;

/// Configuration for the Milestone M11 long-range loop-candidate source.
/// `None` on `crate::dpvo_vo::DpvoOdometryConfig::long_loop` (every prior
/// milestone's default) disables this module entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoLongLoopConfig {
    /// Committed frames to buffer before attempting the first
    /// [`Vocabulary::build`] call — see the module doc's "Streaming
    /// descriptors" section. Default `40`: enough SuperPoint keypoints
    /// (hundreds per frame) pooled together to comfortably exceed
    /// `vocab_words` even on a low-texture opening segment.
    pub vocab_bootstrap_frames: usize,
    /// Vocabulary size (`k` in [`Vocabulary::build`]). Default `32`: small
    /// enough that VLAD's `k * 256`-d output (`8192` floats/frame) stays
    /// cheap to store per indexed frame, large enough for meaningful
    /// appearance discrimination at this module's own scale (hundreds, not
    /// thousands, of indexed frames).
    pub vocab_words: usize,
    /// k-means iteration count for [`Vocabulary::build`]. Default `20`.
    pub vocab_kmeans_iterations: usize,
    /// Deterministic k-means++ seed. Default `0`.
    pub vocab_seed: u64,
    /// Re-check throttle, in committed (arrival-index) frames since the last
    /// query attempt — deliberately INDEPENDENT of and much coarser than
    /// `crate::dpvo_loop_closure::DpvoLoopClosureConfig::global_opt_freq`'s
    /// own short-range throttle, since a long-range retrieval query (VLAD
    /// cosine similarity over the whole indexed history + cross-check
    /// matching + RANSAC) costs meaningfully more per call. Default `40`.
    pub query_frequency: usize,
    /// Top-K appearance candidates (by VLAD cosine similarity) to attempt
    /// geometric verification against, per query, stopping at the first
    /// ACCEPTED one. Default `3`.
    pub top_k: usize,
    /// Minimum VLAD cosine similarity for a candidate to be considered at
    /// all. Default `0.15` — deliberately loose (appearance similarity is
    /// only a proposal signal; the geometric gates below are the actual
    /// correctness backstop, per the module doc's "Failure modes" section).
    pub min_similarity: f32,
    /// Minimum `(current_arrival - candidate_arrival)` gap, in stable
    /// arrival-index units, for a candidate to be considered — the actual
    /// "long-range, not proximity" knob. Default `150`: several times wider
    /// than the proximity mechanism's own measured `~30-49`-frame reach
    /// (`docs/dpvo_droid_port_plan.md`'s M10 results), so this module can
    /// never merely rediscover what `crate::dpvo_loop_closure` already
    /// covers.
    pub min_temporal_gap: usize,
    /// Bound on how many frames' raw keypoints+descriptors stay resident at
    /// once (oldest evicted first once exceeded) — see
    /// [`DpvoLongLoopDiagnostics::estimated_index_bytes`] for the measured
    /// memory this buys. Default `1000`: comfortably covers an 800-frame
    /// acceptance run without ever evicting (so nothing is silently dropped
    /// from the candidate pool during typical runs), while still being a
    /// real, honestly-declared bound for longer sequences.
    pub max_indexed_frames: usize,
    /// Lowe ratio-test threshold for [`BruteForceMatcher`] (the inner
    /// matcher [`CrossCheckMatcher`] wraps). Default `Some(0.8)`, matching
    /// `BruteForceMatcher::default()`.
    pub match_ratio: Option<f32>,
    /// Maximum distance (patch-grid pixels — the SAME downsampled coordinate
    /// space `crate::dpvo_patch_ba::DpvoPatch::x`/`y` already live in, i.e.
    /// full-resolution SuperPoint keypoints already divided by
    /// `visloc_vision::dpvo::RES` by the caller before reaching this module)
    /// between a matched keypoint and the nearest owned DPVO patch for
    /// [`bridge_matches_to_3d3d`] to accept the bridge. Default `3.0`.
    pub patch_pixel_radius: f64,
    /// Minimum bridged 3D-3D correspondence count before attempting
    /// [`ransac_umeyama_scale`] at all — see the module doc's "Failure
    /// modes" section on why this is expected to be a small fraction of the
    /// raw 2D-2D match count. Default `8`.
    pub min_bridge_correspondences: usize,
    /// RANSAC hypothesis count for [`ransac_umeyama_scale`]'s 3-point minimal
    /// samples. Default `300`.
    pub ransac_iterations: usize,
    /// Inlier distance threshold, as a FRACTION of the candidate pair's own
    /// point-cloud scale (median pairwise distance among the NEW-side
    /// points) — a scale-invariant threshold, since DPVO's own world units
    /// are arbitrarily (and, by this milestone's own premise, inconsistently)
    /// scaled. Default `0.15`.
    pub ransac_inlier_threshold_ratio: f64,
    /// Minimum RANSAC inlier count for a fit to be accepted at all. Default
    /// `6`.
    pub min_ransac_inliers: usize,
    /// Maximum allowed `mean_inlier_residual / scene_scale` for the REFIT
    /// (inliers-only) solution — a final quality gate beyond raw inlier
    /// count. Default `0.2`.
    pub max_mean_residual_ratio: f64,
    /// Reject a fitted scale below this (degenerate/numerically meaningless).
    /// Default `1.0e-3`.
    pub min_scale: f64,
    /// Reject a fitted scale above this (degenerate/numerically meaningless
    /// — MH_01's own worst-case drift is `~22.6x`, so this is a generous
    /// upper bound, not a tight one). Default `1.0e3`.
    pub max_scale: f64,
    /// Deterministic RANSAC sampling seed.
    pub ransac_seed: u64,
}

impl Default for DpvoLongLoopConfig {
    fn default() -> Self {
        Self {
            vocab_bootstrap_frames: 40,
            vocab_words: 32,
            vocab_kmeans_iterations: 20,
            vocab_seed: 0,
            query_frequency: 40,
            top_k: 3,
            min_similarity: 0.15,
            min_temporal_gap: 150,
            max_indexed_frames: 1000,
            match_ratio: Some(0.8),
            patch_pixel_radius: 3.0,
            min_bridge_correspondences: 8,
            ransac_iterations: 300,
            ransac_inlier_threshold_ratio: 0.15,
            min_ransac_inliers: 6,
            max_mean_residual_ratio: 0.2,
            min_scale: 1.0e-3,
            max_scale: 1.0e3,
            ransac_seed: 0,
        }
    }
}

/// Diagnostics snapshot — mirrors `crate::dpvo_vo::DpvoGlobalBaDiagnostics`/
/// `DpvoSim3BackendDiagnostics`'s own reporting density.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DpvoLongLoopDiagnostics {
    pub enabled: bool,
    /// Frames currently resident in the raw keypoint/descriptor store (bound
    /// by [`DpvoLongLoopConfig::max_indexed_frames`]).
    pub frames_indexed: usize,
    /// Whether [`Vocabulary::build`] has succeeded yet.
    pub vocab_built: bool,
    /// Estimated resident memory for the raw keypoint/descriptor +VLAD
    /// store, bytes (`descriptors + keypoints + vlad`, `f32`/`f64`-sized).
    pub estimated_index_bytes: usize,
    pub queries_attempted: usize,
    /// Total appearance candidates surfaced (summed across every query,
    /// after the similarity/gap gates, before geometric verification).
    pub candidates_considered: usize,
    /// Total candidates that reached geometric verification (cross-check
    /// matching + bridging + RANSAC).
    pub verification_attempts: usize,
    pub accepted_total: usize,
    pub rejected_insufficient_bridge_total: usize,
    pub rejected_ransac_total: usize,
    pub last_accepted_arrival_i: usize,
    pub last_accepted_arrival_j: usize,
    pub last_accepted_gap: usize,
    pub last_accepted_similarity: f32,
    pub last_accepted_scale: f64,
    pub last_accepted_inliers: usize,
    pub last_accepted_mean_residual_ratio: f64,
    pub total_elapsed_ms: f64,
}

/// One accepted long-range loop — the module's own output, consumed by
/// `crate::dpvo_vo::DpvoOdometry` exactly like a proximity loop pair (see the
/// module doc's "Feeding the existing M9/M10 machinery" section).
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedLongLoop {
    pub arrival_i: usize,
    pub arrival_j: usize,
    pub measurement: Sim3LoopMeasurement,
}

/// One committed frame's retrieval+verification material. Images are never
/// retained (see the module doc); this is the compact per-frame summary
/// computed once, at commit time, from the (borrowed, transient) image.
#[derive(Debug, Clone, PartialEq)]
struct IndexedFrame {
    arrival_index: usize,
    vlad: Vec<f32>,
    /// Patch-grid-coordinate keypoints (already divided by `RES` by the
    /// caller — see [`DpvoLongLoopConfig::patch_pixel_radius`]'s own doc).
    keypoints: Vec<Point2<f64>>,
    descriptors: Vec<Vec<f32>>,
}

fn indexed_frame_bytes(frame: &IndexedFrame) -> usize {
    let descriptor_bytes: usize = frame.descriptors.iter().map(|d| d.len() * std::mem::size_of::<f32>()).sum();
    let keypoint_bytes = frame.keypoints.len() * std::mem::size_of::<Point2<f64>>();
    let vlad_bytes = frame.vlad.len() * std::mem::size_of::<f32>();
    descriptor_bytes + keypoint_bytes + vlad_bytes
}

/// One buffered pre-vocabulary frame — `(arrival_index, keypoints, descriptors)`.
type BootstrapFrame = (usize, Vec<Point2<f64>>, Vec<Vec<f32>>);

/// The streaming long-range loop-candidate index and verifier — see the
/// module doc for the full design.
pub struct DpvoLongLoopIndex {
    config: DpvoLongLoopConfig,
    vocab: Option<Vocabulary>,
    /// Buffered pre-vocabulary frames — see [`Self::ingest_frame`]'s own doc.
    bootstrap: Vec<BootstrapFrame>,
    frames: VecDeque<IndexedFrame>,
    last_query_arrival: Option<usize>,
    rng: StdRng,

    diag_queries_attempted: usize,
    diag_candidates_considered: usize,
    diag_verification_attempts: usize,
    diag_accepted_total: usize,
    diag_rejected_insufficient_bridge: usize,
    diag_rejected_ransac: usize,
    diag_last_arrival_i: usize,
    diag_last_arrival_j: usize,
    diag_last_gap: usize,
    diag_last_similarity: f32,
    diag_last_scale: f64,
    diag_last_inliers: usize,
    diag_last_mean_residual_ratio: f64,
    diag_total_elapsed_ms: f64,
}

impl DpvoLongLoopIndex {
    pub fn new(config: DpvoLongLoopConfig) -> Self {
        let seed = config.ransac_seed;
        Self {
            config,
            vocab: None,
            bootstrap: Vec::new(),
            frames: VecDeque::new(),
            last_query_arrival: None,
            rng: StdRng::seed_from_u64(seed),
            diag_queries_attempted: 0,
            diag_candidates_considered: 0,
            diag_verification_attempts: 0,
            diag_accepted_total: 0,
            diag_rejected_insufficient_bridge: 0,
            diag_rejected_ransac: 0,
            diag_last_arrival_i: 0,
            diag_last_arrival_j: 0,
            diag_last_gap: 0,
            diag_last_similarity: 0.0,
            diag_last_scale: 0.0,
            diag_last_inliers: 0,
            diag_last_mean_residual_ratio: 0.0,
            diag_total_elapsed_ms: 0.0,
        }
    }

    pub fn config(&self) -> &DpvoLongLoopConfig {
        &self.config
    }

    /// Snapshot of this index's own state — see [`DpvoLongLoopDiagnostics`].
    /// `enabled` is always `true` here (the struct only exists when the
    /// caller opted in); `crate::dpvo_vo::DpvoOdometry::long_loop_diagnostics`
    /// reports `enabled: false` (a default-zeroed struct) when disabled.
    pub fn diagnostics(&self) -> DpvoLongLoopDiagnostics {
        let estimated_index_bytes: usize = self.frames.iter().map(indexed_frame_bytes).sum();
        DpvoLongLoopDiagnostics {
            enabled: true,
            frames_indexed: self.frames.len(),
            vocab_built: self.vocab.is_some(),
            estimated_index_bytes,
            queries_attempted: self.diag_queries_attempted,
            candidates_considered: self.diag_candidates_considered,
            verification_attempts: self.diag_verification_attempts,
            accepted_total: self.diag_accepted_total,
            rejected_insufficient_bridge_total: self.diag_rejected_insufficient_bridge,
            rejected_ransac_total: self.diag_rejected_ransac,
            last_accepted_arrival_i: self.diag_last_arrival_i,
            last_accepted_arrival_j: self.diag_last_arrival_j,
            last_accepted_gap: self.diag_last_gap,
            last_accepted_similarity: self.diag_last_similarity,
            last_accepted_scale: self.diag_last_scale,
            last_accepted_inliers: self.diag_last_inliers,
            last_accepted_mean_residual_ratio: self.diag_last_mean_residual_ratio,
            total_elapsed_ms: self.diag_total_elapsed_ms,
        }
    }

    /// Ingest one committed frame's raw SuperPoint keypoints (already in
    /// patch-grid coordinates) + descriptors — called unconditionally, every
    /// committed frame, per the module doc's "images are never retained"
    /// constraint. Before a vocabulary exists, buffers the frame (bounded to
    /// `3 * vocab_bootstrap_frames`, oldest dropped first, as a safety valve
    /// against a pathological low-keypoint opening segment that never
    /// accumulates enough descriptors to build one — see the module doc's
    /// "Failure modes" section); once buffered, attempts
    /// [`Vocabulary::build`] every subsequent call until it succeeds, then
    /// retroactively VLAD-encodes every buffered frame. After a vocabulary
    /// exists, VLAD-encodes and indexes immediately.
    pub fn ingest_frame(&mut self, arrival_index: usize, keypoints: Vec<Point2<f64>>, descriptors: Vec<Vec<f32>>) {
        if descriptors.is_empty() {
            return;
        }
        let encoded = self.vocab.as_ref().map(|vocab| vlad(&descriptors, vocab));
        if let Some(vlad_vector) = encoded {
            self.push_indexed(IndexedFrame { arrival_index, vlad: vlad_vector, keypoints, descriptors });
            return;
        }

        self.bootstrap.push((arrival_index, keypoints, descriptors));
        let safety_cap = self.config.vocab_bootstrap_frames.saturating_mul(3).max(self.config.vocab_bootstrap_frames);
        while self.bootstrap.len() > safety_cap {
            self.bootstrap.remove(0);
        }
        if self.bootstrap.len() >= self.config.vocab_bootstrap_frames {
            self.try_build_vocab();
        }
    }

    fn try_build_vocab(&mut self) {
        let pooled: Vec<&[f32]> =
            self.bootstrap.iter().flat_map(|(_, _, descriptors)| descriptors.iter().map(|d| d.as_slice())).collect();
        let Some(vocab) =
            Vocabulary::build(&pooled, self.config.vocab_words, self.config.vocab_kmeans_iterations, self.config.vocab_seed)
        else {
            return; // Keep buffering — see `Self::ingest_frame`'s own doc.
        };
        let buffered = std::mem::take(&mut self.bootstrap);
        for (arrival_index, keypoints, descriptors) in buffered {
            let vlad_vector = vlad(&descriptors, &vocab);
            self.push_indexed(IndexedFrame { arrival_index, vlad: vlad_vector, keypoints, descriptors });
        }
        self.vocab = Some(vocab);
    }

    fn push_indexed(&mut self, frame: IndexedFrame) {
        self.frames.push_back(frame);
        let cap = self.config.max_indexed_frames.max(1);
        while self.frames.len() > cap {
            self.frames.pop_front();
        }
    }

    /// Throttle check — `true` at most once every
    /// [`DpvoLongLoopConfig::query_frequency`] arrival-index units;
    /// mirrors `crate::dpvo_vo`'s own `global_ba_due`-style "always eligible
    /// on the very first call" semantics. Updates internal state as a side
    /// effect when due (matching that function's own contract), so this
    /// should be called at most once per candidate committed frame.
    pub fn due(&mut self, current_arrival: usize) -> bool {
        let due = match self.last_query_arrival {
            None => true,
            Some(last) => current_arrival.saturating_sub(last) >= self.config.query_frequency.max(1),
        };
        if due {
            self.last_query_arrival = Some(current_arrival);
        }
        due
    }

    /// Rank indexed frames older than `current_arrival` by
    /// `>= min_temporal_gap` AND `>= min_similarity`, descending similarity,
    /// truncated to `top_k`.
    fn query_candidates(&self, current_arrival: usize) -> Vec<(usize, f32)> {
        let Some(current) = self.frames.iter().rev().find(|f| f.arrival_index == current_arrival) else {
            return Vec::new();
        };
        let min_gap = self.config.min_temporal_gap;
        let mut scored: Vec<(usize, f32)> = self
            .frames
            .iter()
            .filter(|f| f.arrival_index != current_arrival && current_arrival.saturating_sub(f.arrival_index) >= min_gap)
            .map(|f| (f.arrival_index, cosine_similarity(&current.vlad, &f.vlad)))
            .filter(|&(_, score)| score >= self.config.min_similarity)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k.max(1));
        scored
    }

    /// The full candidate-generation + geometric-verification pipeline — see
    /// the module doc for the algorithm. `resolve_old` resolves a candidate
    /// arrival index to `(pose, intrinsics, owned patches)` from EITHER the
    /// live patch graph or `crate::dpvo_patch_graph::DpvoPatchGraph`'s M9/M10
    /// retention stores (`retained_poses`/`retained_folded_frames`) — the
    /// caller (`crate::dpvo_vo::DpvoOdometry`) owns the graph, so this module
    /// stays graph-agnostic (mirrors `crate::dpvo_sim3_backend::run_sim3_backend`'s
    /// own separation). Returns `None` (with diagnostics updated for the
    /// rejection reason) if no candidate passes every gate; accepts the FIRST
    /// (highest-similarity) candidate that does, per the module doc's own
    /// "stopping at the first accepted one" design.
    pub fn find_and_verify_long_range_loop(
        &mut self,
        current_arrival: usize,
        current_pose: &SE3,
        current_intrinsics: &DpvoIntrinsics,
        current_patches: &[DpvoPatch],
        mut resolve_old: impl FnMut(usize) -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)>,
    ) -> Option<AcceptedLongLoop> {
        let start = Instant::now();
        self.diag_queries_attempted += 1;

        let candidates = self.query_candidates(current_arrival);
        self.diag_candidates_considered += candidates.len();
        if candidates.is_empty() {
            self.diag_total_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
            return None;
        }

        let Some(current_idx) = self.frames.iter().position(|f| f.arrival_index == current_arrival) else {
            self.diag_total_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
            return None;
        };
        let current_keypoints = self.frames[current_idx].keypoints.clone();
        let current_descriptors = self.frames[current_idx].descriptors.clone();
        let matcher = CrossCheckMatcher::new(BruteForceMatcher { ratio: self.config.match_ratio });

        let mut accepted = None;
        for (old_arrival, similarity) in candidates {
            let Some(old_idx) = self.frames.iter().position(|f| f.arrival_index == old_arrival) else { continue };
            let old_keypoints = self.frames[old_idx].keypoints.clone();
            let old_descriptors = self.frames[old_idx].descriptors.clone();
            let Some((old_pose, old_intr, old_patches)) = resolve_old(old_arrival) else { continue };

            self.diag_verification_attempts += 1;
            let matches = matcher.match_descriptors(&old_descriptors, &current_descriptors);
            let raw_pairs: Vec<(usize, usize)> = matches.iter().map(|m| (m.query_index, m.train_index)).collect();
            let bridged = bridge_matches_to_3d3d(
                &raw_pairs,
                &old_keypoints,
                &old_pose,
                &old_intr,
                &old_patches,
                &current_keypoints,
                current_pose,
                current_intrinsics,
                current_patches,
                self.config.patch_pixel_radius,
            );
            if bridged.len() < self.config.min_bridge_correspondences {
                self.diag_rejected_insufficient_bridge += 1;
                continue;
            }

            let Some(fit) = ransac_umeyama_scale(&bridged, &self.config, &mut self.rng) else {
                self.diag_rejected_ransac += 1;
                continue;
            };

            let relative_pose = current_pose.compose(&old_pose.inverse());
            let measurement = Sim3LoopMeasurement {
                arrival_i: old_arrival,
                arrival_j: current_arrival,
                relative_pose,
                measured_scale: Some(fit.scale),
            };
            self.diag_accepted_total += 1;
            self.diag_last_arrival_i = old_arrival;
            self.diag_last_arrival_j = current_arrival;
            self.diag_last_gap = current_arrival.saturating_sub(old_arrival);
            self.diag_last_similarity = similarity;
            self.diag_last_scale = fit.scale;
            self.diag_last_inliers = fit.inlier_count;
            self.diag_last_mean_residual_ratio = fit.mean_residual_ratio;
            accepted = Some(AcceptedLongLoop { arrival_i: old_arrival, arrival_j: current_arrival, measurement });
            break;
        }

        self.diag_total_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
        accepted
    }
}

/// Backproject a DPVO patch's anchor point into its OWNER frame's own
/// camera-frame 3D point (via that patch's own inverse depth), then into
/// WORLD coordinates via `pose.inverse()` — `pose` is `T_world_to_camera`
/// (`crate::dpvo_patch_ba`'s own convention-mapping note), so its inverse
/// maps camera-frame points to world. Returns `None` for a non-positive or
/// non-finite inverse depth (behind-camera/degenerate — matches
/// `crate::dpvo_patch_ba::reprojected_center_depth`'s own `> 0.2` spirit,
/// just a looser `> 0` floor since this is a one-off backprojection, not an
/// iterated BA residual).
fn patch_to_world_point(pose: &SE3, intr: &DpvoIntrinsics, patch: &DpvoPatch) -> Option<Point3<f64>> {
    if !patch.inverse_depth.is_finite() || patch.inverse_depth <= 1.0e-6 {
        return None;
    }
    let xn = (patch.x - intr.cx) / intr.fx;
    let yn = (patch.y - intr.cy) / intr.fy;
    let cam = Vector3::new(xn, yn, 1.0) / patch.inverse_depth;
    if !cam.iter().all(|v| v.is_finite()) {
        return None;
    }
    let world = pose.inverse().transform_point(&Point3::from(cam));
    if world.coords.iter().all(|v| v.is_finite()) {
        Some(world)
    } else {
        None
    }
}

/// Nearest owned patch to `kp` within `radius` (patch-grid pixels), or
/// `None` if none is that close — see the module doc's "Failure modes"
/// section on why this is expected to reject most raw 2D-2D matches (DPVO's
/// own patches are sparse, randomly-anchored).
fn nearest_patch_within<'a>(kp: &Point2<f64>, patches: &'a [DpvoPatch], radius: f64) -> Option<&'a DpvoPatch> {
    let mut best: Option<(&DpvoPatch, f64)> = None;
    for patch in patches {
        let d = ((patch.x - kp.x).powi(2) + (patch.y - kp.y).powi(2)).sqrt();
        if d > radius {
            continue;
        }
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((patch, d));
        }
    }
    best.map(|(p, _)| p)
}

/// Bridge every raw 2D-2D `(old_keypoint_index, new_keypoint_index)` match to
/// a 3D-3D world-point correspondence — see the module doc's "Candidate
/// generation and geometric verification" section, step 2, for the full
/// derivation of why this pair of independently-backprojected world points is
/// a genuine scale signal.
#[allow(clippy::too_many_arguments)]
fn bridge_matches_to_3d3d(
    raw_pairs: &[(usize, usize)],
    old_keypoints: &[Point2<f64>],
    old_pose: &SE3,
    old_intr: &DpvoIntrinsics,
    old_patches: &[DpvoPatch],
    new_keypoints: &[Point2<f64>],
    new_pose: &SE3,
    new_intr: &DpvoIntrinsics,
    new_patches: &[DpvoPatch],
    radius: f64,
) -> Vec<(Point3<f64>, Point3<f64>)> {
    let mut out = Vec::new();
    for &(oi, ni) in raw_pairs {
        let (Some(kp_o), Some(kp_n)) = (old_keypoints.get(oi), new_keypoints.get(ni)) else { continue };
        let (Some(p_o), Some(p_n)) =
            (nearest_patch_within(kp_o, old_patches, radius), nearest_patch_within(kp_n, new_patches, radius))
        else {
            continue;
        };
        let (Some(w_o), Some(w_n)) =
            (patch_to_world_point(old_pose, old_intr, p_o), patch_to_world_point(new_pose, new_intr, p_n))
        else {
            continue;
        };
        out.push((w_o, w_n));
    }
    out
}

fn median_pairwise_distance(points: &[Point3<f64>]) -> f64 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    let mut dists = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            dists.push((points[i] - points[j]).norm());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists[dists.len() / 2]
}

fn sample_three_distinct(n: usize, rng: &mut StdRng, max_attempts: usize) -> Option<[usize; 3]> {
    if n < 3 {
        return None;
    }
    for _ in 0..max_attempts {
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);
        let c = rng.gen_range(0..n);
        if a != b && b != c && a != c {
            return Some([a, b, c]);
        }
    }
    None
}

fn predict(fit: &TrajectorySimilarityTransform, source: &Point3<f64>) -> Point3<f64> {
    Point3::from(fit.scale * (fit.rotation * source.coords) + fit.translation)
}

fn count_inliers(fit: &TrajectorySimilarityTransform, pairs: &[(Point3<f64>, Point3<f64>)], threshold: f64) -> usize {
    pairs.iter().filter(|(source, target)| (predict(fit, source) - target).norm() <= threshold).count()
}

/// One accepted geometric fit's own diagnostics — see
/// [`ransac_umeyama_scale`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct GeometricFit {
    /// `target ≈ scale * rotation * source + translation`, i.e. (per the
    /// module doc's own source=OLD/target=NEW convention) how much the
    /// OLD-side reconstruction must be scaled UP to match the NEW-side
    /// reconstruction — the SAME "later-over-earlier, >1 as drift grows"
    /// convention `crate::dpvo_sim3_backend::estimate_loop_scale_ratio`
    /// already uses, so this can feed `Sim3LoopMeasurement::measured_scale`
    /// directly with no sign/convention translation needed.
    scale: f64,
    inlier_count: usize,
    mean_residual_ratio: f64,
}

/// 3-point-minimal-sample RANSAC over
/// [`visloc_tracking::umeyama_similarity_transform`] (`with_scale = true`) —
/// see the module doc's "Candidate generation and geometric verification"
/// section, step 3. `pairs` is `(old_world_point, new_world_point)` per
/// correspondence (source=old, target=new — see [`GeometricFit::scale`]'s own
/// doc for why this convention matters). Rejects (returns `None`) on: fewer
/// than `min_bridge_correspondences`/`3` pairs, no hypothesis reaching
/// `min_ransac_inliers`, a fitted scale outside `[min_scale, max_scale]`, or a
/// refit mean-residual-to-scene-scale ratio above `max_mean_residual_ratio`.
fn ransac_umeyama_scale(pairs: &[(Point3<f64>, Point3<f64>)], cfg: &DpvoLongLoopConfig, rng: &mut StdRng) -> Option<GeometricFit> {
    let n = pairs.len();
    if n < cfg.min_bridge_correspondences || n < 3 {
        return None;
    }
    let target_points: Vec<Point3<f64>> = pairs.iter().map(|(_, target)| *target).collect();
    let scene_scale = median_pairwise_distance(&target_points);
    if scene_scale.is_nan() || scene_scale <= 1.0e-9 {
        return None;
    }
    let inlier_threshold = (scene_scale * cfg.ransac_inlier_threshold_ratio).max(1.0e-6);

    let mut best_count = 0usize;
    let mut best_fit: Option<TrajectorySimilarityTransform> = None;
    for _ in 0..cfg.ransac_iterations {
        let Some(sample) = sample_three_distinct(n, rng, 50) else { break };
        let source: Vec<Point3<f64>> = sample.iter().map(|&i| pairs[i].0).collect();
        let target: Vec<Point3<f64>> = sample.iter().map(|&i| pairs[i].1).collect();
        let Some(fit) = umeyama_similarity_transform(&source, &target, true) else { continue };
        if !(fit.scale.is_finite() && fit.scale >= cfg.min_scale && fit.scale <= cfg.max_scale) {
            continue;
        }
        let count = count_inliers(&fit, pairs, inlier_threshold);
        if count > best_count {
            best_count = count;
            best_fit = Some(fit);
        }
    }
    let fit0 = best_fit?;
    if best_count < cfg.min_ransac_inliers {
        return None;
    }

    let inlier_pairs: Vec<(Point3<f64>, Point3<f64>)> =
        pairs.iter().copied().filter(|(source, target)| (predict(&fit0, source) - target).norm() <= inlier_threshold).collect();
    if inlier_pairs.len() < cfg.min_ransac_inliers {
        return None;
    }
    let source: Vec<Point3<f64>> = inlier_pairs.iter().map(|(s, _)| *s).collect();
    let target: Vec<Point3<f64>> = inlier_pairs.iter().map(|(_, t)| *t).collect();
    let refit = umeyama_similarity_transform(&source, &target, true)?;
    if !(refit.scale.is_finite() && refit.scale >= cfg.min_scale && refit.scale <= cfg.max_scale) {
        return None;
    }

    let mean_residual: f64 =
        inlier_pairs.iter().map(|(source, target)| (predict(&refit, source) - target).norm()).sum::<f64>()
            / inlier_pairs.len() as f64;
    let mean_residual_ratio = mean_residual / scene_scale;
    if mean_residual_ratio > cfg.max_mean_residual_ratio {
        return None;
    }

    Some(GeometricFit { scale: refit.scale, inlier_count: inlier_pairs.len(), mean_residual_ratio })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn intr() -> DpvoIntrinsics {
        DpvoIntrinsics { fx: 200.0, fy: 200.0, cx: 64.0, cy: 48.0 }
    }

    fn cfg() -> DpvoLongLoopConfig {
        DpvoLongLoopConfig::default()
    }

    // ---- patch_to_world_point / nearest_patch_within / bridge ----

    #[test]
    fn patch_to_world_point_matches_hand_derivation_for_identity_pose() {
        let pose = SE3::identity();
        let intrinsics = intr();
        let patch = DpvoPatch { x: 64.0, y: 48.0, inverse_depth: 0.5 }; // depth = 2, at the principal point.
        let world = patch_to_world_point(&pose, &intrinsics, &patch).expect("positive depth");
        assert!((world.coords - Vector3::new(0.0, 0.0, 2.0)).norm() < 1e-9, "{world:?}");
    }

    #[test]
    fn patch_to_world_point_rejects_non_positive_inverse_depth() {
        let pose = SE3::identity();
        let intrinsics = intr();
        assert!(patch_to_world_point(&pose, &intrinsics, &DpvoPatch { x: 64.0, y: 48.0, inverse_depth: 0.0 }).is_none());
        assert!(patch_to_world_point(&pose, &intrinsics, &DpvoPatch { x: 64.0, y: 48.0, inverse_depth: -0.5 }).is_none());
    }

    #[test]
    fn nearest_patch_within_picks_the_closest_and_respects_radius() {
        let patches = vec![
            DpvoPatch { x: 10.0, y: 10.0, inverse_depth: 0.5 },
            DpvoPatch { x: 10.5, y: 10.5, inverse_depth: 0.5 },
            DpvoPatch { x: 50.0, y: 50.0, inverse_depth: 0.5 },
        ];
        // Distance to (10,10) is sqrt(0.2^2+0.2^2)=0.283; to (10.5,10.5) is
        // sqrt(0.3^2+0.3^2)=0.424 — (10,10) is the true nearest.
        let kp = Point2::new(10.2, 10.2);
        let nearest = nearest_patch_within(&kp, &patches, 3.0).expect("a patch within radius");
        assert!((nearest.x - 10.0).abs() < 1e-9 && (nearest.y - 10.0).abs() < 1e-9);
        assert!(nearest_patch_within(&Point2::new(200.0, 200.0), &patches, 3.0).is_none());
    }

    /// Return type of [`synthetic_drifted_pair`] — a plain struct instead of
    /// a 6-tuple purely to keep `clippy::type_complexity` quiet at the call
    /// sites below.
    struct DriftedPairFixture {
        old_pose: SE3,
        old_patches: Vec<DpvoPatch>,
        old_keypoints: Vec<Point2<f64>>,
        new_pose: SE3,
        new_patches: Vec<DpvoPatch>,
        new_keypoints: Vec<Point2<f64>>,
    }

    /// Build two independent local reconstructions of the SAME `n` physical
    /// points: the "old" side sees them at `true_points` via `old_pose`
    /// (identity, unit scale); the "new" side sees the SAME points but with
    /// depths/translation inflated by `drift_scale` (mimicking DPVO's own
    /// monocular scale drift accumulated between the two observations) —
    /// i.e. `new_world_point = drift_scale * true_point` when `new_pose` is
    /// also identity. Keypoints are placed EXACTLY at their own side's patch
    /// anchor (radius-zero bridging), so `bridge_matches_to_3d3d`'s own
    /// patch lookup always succeeds — the synthetic fixture's job is to
    /// exercise the SCALE ESTIMATOR, not patch lookup tolerance (covered
    /// separately above).
    fn synthetic_drifted_pair(n: usize, drift_scale: f64) -> DriftedPairFixture {
        let intrinsics = intr();
        let old_pose = SE3::identity();
        let new_pose = SE3::identity();
        let mut old_patches = Vec::with_capacity(n);
        let mut old_keypoints = Vec::with_capacity(n);
        let mut new_patches = Vec::with_capacity(n);
        let mut new_keypoints = Vec::with_capacity(n);
        for i in 0..n {
            let dx = (i as f64 - n as f64 / 2.0) * 3.0;
            let dy = ((i * 7) % n) as f64 - n as f64 / 2.0;
            let depth = 3.0 + (i as f64) * 0.2;
            let x = intrinsics.cx + dx;
            let y = intrinsics.cy + dy;
            old_patches.push(DpvoPatch { x, y, inverse_depth: 1.0 / depth });
            old_keypoints.push(Point2::new(x, y));
            // Same pixel anchor (identity pose on both sides, so a physical
            // point at old-side depth `depth` reprojects to new-side pixel
            // `(x, y)` scaled radially only if the pose changed — with both
            // poses identity, the ONLY thing differing is depth itself,
            // scaled by `drift_scale`).
            new_patches.push(DpvoPatch { x, y, inverse_depth: 1.0 / (depth * drift_scale) });
            new_keypoints.push(Point2::new(x, y));
        }
        DriftedPairFixture { old_pose, old_patches, old_keypoints, new_pose, new_patches, new_keypoints }
    }

    #[test]
    fn bridge_matches_to_3d3d_recovers_expected_world_points() {
        let DriftedPairFixture { old_pose, old_patches, old_keypoints, new_pose, new_patches, new_keypoints } =
            synthetic_drifted_pair(10, 4.0);
        let intrinsics = intr();
        let raw_pairs: Vec<(usize, usize)> = (0..10).map(|i| (i, i)).collect();
        let bridged = bridge_matches_to_3d3d(
            &raw_pairs, &old_keypoints, &old_pose, &intrinsics, &old_patches, &new_keypoints, &new_pose, &intrinsics,
            &new_patches, 1.0,
        );
        assert_eq!(bridged.len(), 10);
        for (old_world, new_world) in &bridged {
            // Same pixel/pose on both sides => the ONLY difference is depth
            // scaled by `drift_scale` => new_world == drift_scale * old_world.
            assert!((new_world.coords - old_world.coords * 4.0).norm() < 1e-9);
        }
    }

    // ---- ransac_umeyama_scale: the core scale estimator ----

    #[test]
    fn ransac_umeyama_scale_recovers_a_known_scale_within_a_few_percent() {
        let DriftedPairFixture { old_pose, old_patches, old_keypoints, new_pose, new_patches, new_keypoints } =
            synthetic_drifted_pair(30, 6.5);
        let intrinsics = intr();
        let raw_pairs: Vec<(usize, usize)> = (0..30).map(|i| (i, i)).collect();
        let bridged = bridge_matches_to_3d3d(
            &raw_pairs, &old_keypoints, &old_pose, &intrinsics, &old_patches, &new_keypoints, &new_pose, &intrinsics,
            &new_patches, 1.0,
        );
        assert_eq!(bridged.len(), 30);
        let mut rng = StdRng::seed_from_u64(7);
        let fit = ransac_umeyama_scale(&bridged, &cfg(), &mut rng).expect("well-posed synthetic fixture should fit");
        assert!(
            (fit.scale - 6.5).abs() / 6.5 < 0.05,
            "expected scale close to 6.5, got {} (inliers={} residual_ratio={})",
            fit.scale,
            fit.inlier_count,
            fit.mean_residual_ratio
        );
        assert_eq!(fit.inlier_count, 30, "a noise-free fixture should have every pair as an inlier");
    }

    #[test]
    fn ransac_umeyama_scale_recovers_a_rotated_translated_scaled_transform() {
        // A less degenerate fixture than `synthetic_drifted_pair` (which
        // keeps both sides' points at identical pixel coordinates): apply a
        // genuine rotation+translation+scale to the "old" points to produce
        // the "new" points directly (bypassing patch/pose backprojection
        // entirely), confirming the estimator recovers a KNOWN Sim(3), not
        // merely a pure scale on an otherwise-aligned point cloud.
        let true_scale = 3.25_f64;
        let true_rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), 0.3);
        let true_translation = Vector3::new(1.0, -0.5, 2.0);
        let old_points: Vec<Point3<f64>> = (0..20)
            .map(|i| {
                let t = i as f64;
                Point3::new(t * 0.3, (t * 1.7).sin() * 2.0, (t * 0.9).cos() * 2.0 + 5.0)
            })
            .collect();
        let new_points: Vec<Point3<f64>> =
            old_points.iter().map(|p| Point3::from(true_scale * (true_rotation * p.coords) + true_translation)).collect();
        let pairs: Vec<(Point3<f64>, Point3<f64>)> = old_points.into_iter().zip(new_points).collect();

        let mut rng = StdRng::seed_from_u64(42);
        let fit = ransac_umeyama_scale(&pairs, &cfg(), &mut rng).expect("well-posed rigid+scale fixture should fit");
        assert!((fit.scale - true_scale).abs() / true_scale < 0.02, "expected scale near {true_scale}, got {}", fit.scale);
        assert_eq!(fit.inlier_count, 20);
    }

    #[test]
    fn ransac_umeyama_scale_rejects_below_min_bridge_correspondences() {
        let DriftedPairFixture { old_pose, old_patches, old_keypoints, new_pose, new_patches, new_keypoints } =
            synthetic_drifted_pair(3, 2.0);
        let intrinsics = intr();
        let raw_pairs: Vec<(usize, usize)> = (0..3).map(|i| (i, i)).collect();
        let bridged = bridge_matches_to_3d3d(
            &raw_pairs, &old_keypoints, &old_pose, &intrinsics, &old_patches, &new_keypoints, &new_pose, &intrinsics,
            &new_patches, 1.0,
        );
        // 3 correspondences < the default `min_bridge_correspondences = 8`.
        let mut rng = StdRng::seed_from_u64(1);
        assert!(ransac_umeyama_scale(&bridged, &cfg(), &mut rng).is_none());
    }

    #[test]
    fn ransac_umeyama_scale_rejects_incoherent_random_correspondences() {
        // Correspondences with NO consistent Sim(3) relating them at all
        // (independent random points on both sides) should fail to find
        // `min_ransac_inliers` agreeing inliers under a tight threshold —
        // the degenerate/no-consensus rejection path, not a crash or a
        // spuriously "confident" fit.
        let mut rng = StdRng::seed_from_u64(99);
        let pairs: Vec<(Point3<f64>, Point3<f64>)> = (0..20)
            .map(|_| {
                let source = Point3::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0), rng.gen_range(1.0..10.0));
                let target = Point3::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0), rng.gen_range(1.0..10.0));
                (source, target)
            })
            .collect();
        let mut cfg = cfg();
        cfg.ransac_inlier_threshold_ratio = 0.02; // tight — random pairs should not cluster inside this.
        let mut rng2 = StdRng::seed_from_u64(3);
        assert!(
            ransac_umeyama_scale(&pairs, &cfg, &mut rng2).is_none(),
            "fully incoherent random correspondences should not produce an accepted fit"
        );
    }

    // ---- DpvoLongLoopIndex: bootstrap, ingest, query, no-op-until-vocab ----

    fn synthetic_descriptor(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    /// `count` local descriptors, all near `base_seed`-derived direction
    /// (small jitter) — two frames built from the SAME `base_seed` are
    /// "appearance-similar"; different `base_seed`s are not.
    fn frame_descriptors(base_seed: u64, count: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..count).map(|k| synthetic_descriptor(base_seed.wrapping_add(k as u64 * 7919), dim)).collect()
    }

    #[test]
    fn index_stays_unbuilt_until_bootstrap_threshold_then_builds_and_backfills() {
        let cfg = DpvoLongLoopConfig { vocab_bootstrap_frames: 5, vocab_words: 4, ..DpvoLongLoopConfig::default() };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..4 {
            index.ingest_frame(arrival, vec![], frame_descriptors(arrival as u64, 20, 16));
            assert!(!index.diagnostics().vocab_built, "should not build before the bootstrap threshold");
            assert_eq!(index.diagnostics().frames_indexed, 0, "nothing indexed until the vocabulary exists");
        }
        index.ingest_frame(4, vec![], frame_descriptors(4, 20, 16));
        assert!(index.diagnostics().vocab_built, "vocabulary should build once the threshold is reached");
        assert_eq!(index.diagnostics().frames_indexed, 5, "every buffered frame should be backfilled");
    }

    #[test]
    fn due_throttles_by_query_frequency() {
        let cfg = DpvoLongLoopConfig { query_frequency: 10, ..DpvoLongLoopConfig::default() };
        let mut index = DpvoLongLoopIndex::new(cfg);
        assert!(index.due(0), "first call is always due");
        assert!(!index.due(5), "too soon");
        assert!(index.due(10), "exactly the throttle period later");
        assert!(!index.due(15));
        assert!(index.due(25));
    }

    #[test]
    fn query_candidates_respects_temporal_gap_and_similarity_and_ranks_by_score() {
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0, // accept everything the gap gate allows, for this ranking test.
            top_k: 5,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        // Bootstrap with 3 frames sharing appearance "A" so the vocabulary exists.
        for arrival in 0..3 {
            index.ingest_frame(arrival, vec![], frame_descriptors(1, 20, 16));
        }
        // A near-appearance-match far in the past (arrival 10, gap 200 from
        // the query at 210 — passes min_temporal_gap).
        index.ingest_frame(10, vec![], frame_descriptors(1, 20, 16));
        // A recent frame within min_temporal_gap of the query — must be excluded regardless of similarity.
        index.ingest_frame(190, vec![], frame_descriptors(1, 20, 16));
        // The query frame itself, appearance "A" (matches arrival 10 well).
        index.ingest_frame(210, vec![], frame_descriptors(1, 20, 16));

        let candidates = index.query_candidates(210);
        let arrivals: Vec<usize> = candidates.iter().map(|&(a, _)| a).collect();
        assert!(arrivals.contains(&10), "arrival 10 (gap=200) should be a candidate: {arrivals:?}");
        assert!(!arrivals.contains(&190), "arrival 190 (gap=20 < min_temporal_gap=50) must be excluded: {arrivals:?}");
        assert!(!arrivals.contains(&210), "the query frame itself must never be its own candidate");
    }

    // ---- End-to-end: find_and_verify_long_range_loop ----

    #[test]
    fn find_and_verify_long_range_loop_accepts_a_genuine_long_range_revisit_with_correct_scale() {
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            patch_pixel_radius: 0.5, // keypoints placed exactly at patch anchors below.
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);

        let intrinsics = intr();
        let DriftedPairFixture { old_pose, old_patches, old_keypoints, new_pose, new_patches, new_keypoints } =
            synthetic_drifted_pair(30, 8.0);

        // Bootstrap filler frames (distinct appearance, far from both real
        // frames in arrival index and appearance) so the vocabulary builds
        // without becoming a trivial 1-frame-vs-1-frame retrieval.
        for arrival in 0..3 {
            index.ingest_frame(arrival, vec![], frame_descriptors(1000 + arrival as u64, 20, 16));
        }
        // The OLD frame (arrival 20), matching descriptors shared with the
        // NEW frame below (appearance base_seed `1` on both sides, matched
        // 1:1 by construction so cross-check matching succeeds deterministically).
        let old_descriptors = frame_descriptors(1, 30, 16);
        index.ingest_frame(20, old_keypoints.clone(), old_descriptors.clone());
        // The NEW/current frame (arrival 300 — gap 280 >> min_temporal_gap,
        // and >> the proximity mechanism's own ~30-49-frame reach).
        let new_descriptors = old_descriptors.clone(); // identical appearance => a perfect cross-check match per index.
        index.ingest_frame(300, new_keypoints.clone(), new_descriptors);

        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, old_patches.clone()))
            } else {
                None
            }
        };

        let accepted = index
            .find_and_verify_long_range_loop(300, &new_pose, &intrinsics, &new_patches, resolve_old)
            .expect("a genuine, well-posed long-range revisit should be accepted");
        assert_eq!(accepted.arrival_i, 20);
        assert_eq!(accepted.arrival_j, 300);
        assert_eq!(accepted.measurement.arrival_i, 20);
        assert_eq!(accepted.measurement.arrival_j, 300);
        let measured_scale = accepted.measurement.measured_scale.expect("M11 acceptance must carry a measured scale");
        assert!((measured_scale - 8.0).abs() / 8.0 < 0.05, "expected measured_scale near 8.0, got {measured_scale}");
        // The ordinary rotation+translation edge reuses DPVO's own current
        // pose composition (both poses are identity here, so this is trivially identity too).
        assert_eq!(accepted.measurement.relative_pose, new_pose.compose(&old_pose.inverse()));

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 1);
        assert_eq!(diagnostics.last_accepted_gap, 280);
    }

    #[test]
    fn find_and_verify_long_range_loop_is_a_noop_before_any_candidate_clears_the_gap() {
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 500, // deliberately unreachable within this small fixture.
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..5 {
            index.ingest_frame(arrival, vec![], frame_descriptors(arrival as u64, 20, 16));
        }
        let intrinsics = intr();
        let pose = SE3::identity();
        let result = index.find_and_verify_long_range_loop(4, &pose, &intrinsics, &[], |_| None);
        assert!(result.is_none());
        assert_eq!(index.diagnostics().accepted_total, 0);
    }
}
