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
//!    relative to the point cloud's own scale, a fitted scale outside a sane
//!    range, or (Milestone M12, added after a real 800f corruption run —
//!    `docs/dpvo_droid_port_plan.md`'s "M12 results") the fit's own
//!    RECOVERED ROTATION disagreeing with DPVO's already-trusted relative
//!    rotation by more than `max_rotation_inconsistency_deg` — an
//!    independent physical check the RANSAC/residual gates alone cannot
//!    provide, since they only verify the bridged SAMPLE's own internal
//!    self-consistency, not agreement with a trajectory-level estimate — **no
//!    fallback to `scale = 1`**: the whole point of this milestone is a
//!    genuine scale measurement, so a candidate whose geometry cannot
//!    support one is discarded rather than accepted with a vacuous scale.
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
//!
//! # A3 stage 2, first slice: 2D-2D-first loop geometry (opt-in)
//!
//! `docs/visual_slam_sequential_sfm_plan.md`'s "A3 — Sound long-range loop
//! closure" section splits the problem into retrieval recall (stage 1,
//! covered above) and geometric precision (stage 2). Stage-1b's own
//! densified-cadence A/B found 5 newly-accepted long loops that passed EVERY
//! existing M11/M12 gate (bridge yield, RANSAC inliers/residual, the
//! existing rotation-vs-trusted-pose gate) yet were still physically wrong —
//! direct evidence that loop geometry sourced from the DRIFTED 3D-3D bridge
//! is not enough; a genuinely independent signal is needed. When
//! [`DpvoLongLoopConfig::stage2_2d2d_geometry`] is `true` (default `false`,
//! every prior milestone's behavior unchanged), each retrieval candidate is
//! additionally required to pass, BEFORE the existing 3D-3D bridge ever
//! runs:
//!
//! (a) a 2D-2D cross-check match of the two frames' raw SuperPoint
//!     descriptors (mutual nearest-neighbor + Lowe ratio, separate from the
//!     existing bridge's own match — [`DpvoLongLoopConfig::stage2_match_ratio`]);
//! (b) a calibrated relative pose from that match, via
//!     [`visloc_vision::two_view::RelativePoseEstimator`] (essential-matrix
//!     RANSAC + cheirality-checked decomposition) — reused wholesale, not
//!     reimplemented;
//! (c) three gates on that fit: minimum inlier count
//!     ([`DpvoLongLoopConfig::stage2_min_inliers`]), inlier spatial coverage
//!     ([`DpvoLongLoopConfig::stage2_min_coverage_fraction`],
//!     [`inlier_bbox_area_fraction`]), and rotation agreement with DPVO's
//!     trusted relative rotation — reusing
//!     [`DpvoLongLoopConfig::max_rotation_inconsistency_deg`]'s EXISTING
//!     threshold/comparison, not a new, looser one;
//! (d) only once (a)-(c) pass does the existing 3D-3D bridge + Umeyama
//!     Sim(3) fit run, with ONE additional gate: the Umeyama fit's own
//!     recovered rotation must agree with step (b)'s E-derived rotation
//!     within [`DpvoLongLoopConfig::stage2_umeyama_vs_e_rotation_max_deg`]
//!     (default `10.0°`, tighter than, and in addition to, the existing
//!     `20.0°` gate).
//!
//! Every candidate's own outcome (furthest stage reached, 2D-2D inlier
//! count, E-vs-trusted rotation disagreement) is logged unconditionally into
//! [`DpvoLongLoopIndex::query_log`] — see [`QueryCandidateLogEntry`]'s new
//! fields.
//!
//! # A3 ranking slice B: vocabulary-free mean-pool scorer (opt-in)
//!
//! `docs/visual_slam_sequential_sfm_plan.md`'s "A3" section, "Decisive
//! implication" paragraph: the ranking-lab's own offline evaluation
//! (`scripts/eval_dpvo_retrieval_ranking_offline.py`, `E:/visloc_archive/dpvo_a3_20260721/ranking_offline/`)
//! found that a per-frame signature as simple as the L2-normalized MEAN of a
//! frame's own raw SP descriptors — no vocabulary, no k-means, no VLAD
//! residual pooling — reaches the SAME recall@1 (`0.989`) as the streaming
//! VLAD-32 index's own best-case offline analog, using `32x` less per-frame
//! storage (`256` floats vs `8192`). [`RetrievalScorer::MeanPool`] (selected
//! via `DpvoLongLoopConfig::retrieval_scorer`, default
//! [`RetrievalScorer::Vlad`] — every prior milestone's behavior byte-for-byte
//! unchanged) makes this the live index's own scorer: every ingested frame's
//! signature is [`mean_pool`] of its raw descriptors, computed and indexed
//! IMMEDIATELY on ingest — no `vocab_bootstrap_frames` warm-up buffering, no
//! `Vocabulary::build` call, ever. Everything downstream of "rank candidates
//! by cosine similarity of two signatures" (`top_k`, `min_temporal_gap`,
//! `query_frequency`/[`DpvoLongLoopIndex::due`], the whole stage-2 geometric
//! pipeline, `query_log`) is scorer-agnostic and reused unchanged.

use std::collections::VecDeque;
use std::time::Instant;

use nalgebra::{Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use visloc_core::geometry::SE3;
use visloc_core::types::Camera;
use visloc_tracking::{umeyama_similarity_transform, TrajectorySimilarityTransform};
use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, DescriptorMatch, Matcher};
use visloc_vision::place_recognition::{cosine_similarity, mean_pool, vlad, Vocabulary};
use visloc_vision::two_view::{
    ConfigurationType, RelativePoseEstimator, TwoViewCorrespondence, TwoViewGeometryOptions,
    TwoViewGeometryVerifier,
};

use crate::dpvo_patch_ba::{DpvoIntrinsics, DpvoPatch};
use crate::dpvo_sim3_backend::Sim3LoopMeasurement;

/// A3 ranking slice B (`docs/visual_slam_sequential_sfm_plan.md`, "A3 —
/// Sound long-range loop closure", "Decisive implication" paragraph): which
/// per-frame global-descriptor SCORER [`DpvoLongLoopIndex`] uses to rank
/// retrieval candidates — see the module doc's own "A3 ranking slice B"
/// section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetrievalScorer {
    /// The original M11-M12 mechanism: a [`Vocabulary`] trained ONCE (see
    /// the module doc's "Streaming descriptors" section) from the first
    /// `vocab_bootstrap_frames` committed frames, then every frame
    /// (including the buffered ones, retroactively) VLAD-encoded against
    /// it. Default — every prior milestone's behavior (M1-M12, A3 stage 1
    /// and 2) reproduced byte-for-byte.
    #[default]
    Vlad,
    /// The A3 ranking-lab's own "decisive implication": [`mean_pool`] of a
    /// frame's raw SP descriptors, L2-normalized, cosine-ranked — NO
    /// vocabulary is ever trained (or buffered-while-waiting-to-train) when
    /// this is selected; every ingested frame is immediately queryable from
    /// the very first eligible arrival, removing the vocabulary bootstrap
    /// dependency entirely. See the module doc's own "A3 ranking slice B"
    /// section for the offline evidence motivating this.
    MeanPool,
}

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
    /// A3 ranking slice B: which per-frame global-descriptor scorer this
    /// index uses — see [`RetrievalScorer`]'s own doc. Default
    /// [`RetrievalScorer::Vlad`]: every prior milestone's behavior
    /// byte-for-byte unchanged, including `vocab_bootstrap_frames`,
    /// `vocab_words`, `vocab_kmeans_iterations`, and `vocab_seed` above,
    /// which are all unused (never read) when this is
    /// [`RetrievalScorer::MeanPool`].
    pub retrieval_scorer: RetrievalScorer,
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
    /// Minimum cosine similarity (of whichever `retrieval_scorer` signature
    /// is active) for a candidate to be considered at all. Default `0.15`,
    /// calibrated for [`RetrievalScorer::Vlad`] — deliberately loose
    /// (appearance similarity is only a proposal signal; the geometric
    /// gates below are the actual correctness backstop, per the module
    /// doc's "Failure modes" section). [`RetrievalScorer::MeanPool`]'s
    /// cosine similarities run structurally higher (offline hit
    /// similarities cluster `~0.58`-`0.99` on MH_01, median `~0.91` — see
    /// the module doc's "A3 ranking slice B" section); a caller selecting
    /// `MeanPool` should override this field (`--ll-min-similarity` in
    /// `examples/euroc_dpvo_vo_demo.rs`, whose own default switches to
    /// `0.5` when `--ll-retrieval-scorer mean-pool` is selected without an
    /// explicit override) rather than keep this VLAD-calibrated default.
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
    /// Milestone M12 (`docs/dpvo_droid_port_plan.md`): when `true`, this
    /// frame's DPVO patch centers are chosen from its OWN SuperPoint
    /// keypoints (the SAME ones this module already extracts for retrieval
    /// indexing — see `crate::dpvo_vo::DpvoOdometry::process_frame`'s own
    /// doc, "SuperPoint feature extraction is moved earlier") instead of
    /// pure per-patch uniform-random sampling, ranked by keypoint score, up
    /// to `patches_per_frame`; any shortfall (fewer usable keypoints than
    /// `patches_per_frame` after the `sp_patch_min_separation` de-duplication
    /// below) is filled by the SAME uniform-random sampler M1-M11 already
    /// used, byte-identically. `false` (default) reproduces the exact M1-M11
    /// fully-random sampling — see [`sp_anchored_patch_centers`] for the
    /// actual coordinate mapping and the M11 motivation
    /// (`docs/dpvo_droid_port_plan.md`'s "M11 results": at `fast.yaml` patch
    /// density, a matched appearance keypoint essentially never lands near
    /// an existing RANDOMLY-placed patch, so the bridge from a long-range
    /// retrieval match to a DPVO-owned 3D point almost always fails).
    pub sp_anchored_patches: bool,
    /// Minimum center-to-center distance (patch-grid pixels, the SAME space
    /// `DpvoPatch::x`/`y` live in) between two SP-anchored patch centers
    /// chosen for the SAME frame — a simple de-duplication so two SuperPoint
    /// keypoints that map to (almost) the same patch-grid cell do not spend
    /// two of the frame's `patches_per_frame` budget on effectively the same
    /// 3×3 correlation window. Default `2.0`: `patchify_cpu`'s own sampled
    /// window is `2 * radius + 2 = 4` patch-grid pixels wide at DPVO's
    /// `radius = 1`, so `2.0` keeps any two chosen centers from overlapping
    /// more than half their own window. Unused when `sp_anchored_patches` is
    /// `false`.
    pub sp_patch_min_separation: f64,
    /// Milestone M12 (post-mortem on a real 800f corruption this milestone's
    /// own SP-anchoring diagnosis produced — `docs/dpvo_droid_port_plan.md`'s
    /// "M12 results", "measurement-vs-application" section): reject a
    /// candidate whenever [`ransac_umeyama_scale`]'s own RECOVERED ROTATION
    /// disagrees with DPVO's own trusted relative rotation
    /// (`current_pose.compose(&old_pose.inverse())`, the SAME rotation the
    /// accepted measurement's ordinary edge already reuses — see the module
    /// doc's "Feeding the existing M9/M10 machinery" section) by more than
    /// this many DEGREES. An independent, physically-motivated check
    /// RANSAC's own inlier-count/residual-ratio gates cannot provide — they
    /// only check the bridged SAMPLE's own internal self-consistency, never
    /// agreement with a trajectory-level rotation estimate DPVO's own
    /// monocular tracking is generally reliable for (this port's own
    /// standing assumption, restated in `crate::dpvo_sim3_backend`'s module
    /// doc). Default `20.0`: on real MH_01 data, this milestone's own two
    /// CONFIRMED-bad candidates (measured scale `386.9`/`173.8`, both an
    /// order of magnitude past MH_01's own worst documented drift) disagreed
    /// with the trusted rotation by tens of degrees while both CONFIRMED-good
    /// candidates (scale `0.18`/`1.13`) disagreed by only a few degrees —
    /// `20.0` sits comfortably between the two, loose enough to tolerate
    /// ordinary VO rotation uncertainty over a `~150`-`450`-frame gap, tight
    /// enough to catch a grossly wrong correspondence set.
    pub max_rotation_inconsistency_deg: f64,
    /// A3 stage 2, first slice (`docs/visual_slam_sequential_sfm_plan.md`,
    /// "A3 — Sound long-range loop closure"): when `true`, every retrieval
    /// candidate that clears the appearance/gap filter is additionally
    /// required to pass a 2D-2D-FIRST geometric check (see the module doc's
    /// "A3 stage 2" section) BEFORE the existing M11/M12 3D-3D bridge
    /// ([`bridge_matches_to_3d3d`]) ever runs. Motivation: Stage-1b's own
    /// densified-cadence A/B (`docs/visual_slam_sequential_sfm_plan.md`'s
    /// "Stage-1b" block) found 5 newly-accepted long loops (`220->388` ..
    /// `282->443`, rotation disagreement `15.7`-`19.8°`, all just under the
    /// existing `20°` gate) that passed EVERY existing M11/M12 gate (bridge
    /// yield, RANSAC inliers/residual, `max_rotation_inconsistency_deg`) yet
    /// were still physically wrong — the M12 rotation gate compares the
    /// DRIFTED 3D-3D bridge's own recovered rotation against DPVO's own
    /// (also monocular, also drift-prone) trusted pose, so a candidate whose
    /// bridge AND trusted pose are both subtly wrong in a mutually
    /// consistent way slips through. Loop geometry sourced from a
    /// CALIBRATED essential-matrix fit over cross-checked 2D keypoints is a
    /// genuinely independent signal (no DPVO patch depth or trusted pose
    /// involved at all in the fit itself). Default `false`: every prior
    /// milestone's behavior (M1-M12) is reproduced byte-for-byte when this
    /// stays off — see `--ll-2d2d-geometry`.
    pub stage2_2d2d_geometry: bool,
    /// A3 stage-2 low-baseline diagnostic: additionally run the existing
    /// COLMAP-style E/F/H classifier for every stage-2 candidate and expose
    /// its model, homography inlier count, and homography-vs-trusted
    /// rotation disagreement in [`QueryCandidateLogEntry`]. Diagnostic
    /// only: this flag never changes acceptance or any existing gate.
    /// Default `false`.
    pub stage2_low_baseline_diagnostic: bool,
    /// Lowe ratio-test threshold for the STAGE-2 2D-2D cross-check match —
    /// separate from `match_ratio`, which still governs the EXISTING 3D-3D
    /// bridge's own cross-check match, unchanged. Default `Some(0.9)`, per
    /// this slice's own task brief ("Lowe ratio ~0.9, symmetric").
    pub stage2_match_ratio: Option<f32>,
    /// Minimum essential-matrix RANSAC inlier count for the 2D-2D fit
    /// ([`visloc_vision::two_view::RelativePoseEstimator`]). Default `30`.
    pub stage2_min_inliers: usize,
    /// Minimum fraction of the frame's own patch-grid area the 2D-2D
    /// inliers' own pixel bounding box must cover, checked on BOTH the OLD
    /// and NEW side independently (the tighter of the two must still clear
    /// this bound) — see [`inlier_bbox_area_fraction`]. Default `0.25`
    /// (25%).
    pub stage2_min_coverage_fraction: f64,
    /// Maximum acceptable mean Sampson error (normalized bearing units, the
    /// SAME patch-grid-space intrinsics [`DpvoIntrinsics`] already uses) for
    /// the 2D-2D essential-matrix fit — a final "residual is sane" backstop
    /// on top of [`visloc_vision::two_view::EssentialRansac`]'s own
    /// per-correspondence inlier threshold (`EssentialRansacConfig::default()`'s
    /// `sampson_threshold = 5.0e-3`). Default `1.0e-2` (`2x` that
    /// per-inlier threshold).
    pub stage2_max_mean_sampson_error: f64,
    /// Step (d) of this slice's own task brief: an ADDITIONAL, TIGHTER gate
    /// (never looser than `max_rotation_inconsistency_deg`) comparing
    /// [`ransac_umeyama_scale`]'s own recovered rotation against the 2D-2D
    /// essential-matrix fit's own recovered rotation (NOT DPVO's trusted
    /// pose, which `max_rotation_inconsistency_deg` already checks
    /// independently) — the two INDEPENDENT rotation estimates (one from
    /// calibrated 2D-2D epipolar geometry, one from the 3D-3D bridge) must
    /// themselves agree, catching exactly the "bridge and trusted pose are
    /// both wrong in a mutually consistent way" failure mode
    /// `stage2_2d2d_geometry`'s own doc describes. Default `10.0` degrees —
    /// tighter than `max_rotation_inconsistency_deg`'s `20.0`, never looser
    /// (this port's own standing rule: never loosen a loop-verification
    /// gate). Only evaluated when `stage2_2d2d_geometry` is `true` (there is
    /// no independent E-matrix rotation to compare against otherwise).
    pub stage2_umeyama_vs_e_rotation_max_deg: f64,
}

impl Default for DpvoLongLoopConfig {
    fn default() -> Self {
        Self {
            vocab_bootstrap_frames: 40,
            vocab_words: 32,
            vocab_kmeans_iterations: 20,
            vocab_seed: 0,
            retrieval_scorer: RetrievalScorer::Vlad,
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
            sp_anchored_patches: false,
            sp_patch_min_separation: 2.0,
            max_rotation_inconsistency_deg: 20.0,
            stage2_2d2d_geometry: false,
            stage2_low_baseline_diagnostic: false,
            stage2_match_ratio: Some(0.9),
            stage2_min_inliers: 30,
            stage2_min_coverage_fraction: 0.25,
            stage2_max_mean_sampson_error: 1.0e-2,
            stage2_umeyama_vs_e_rotation_max_deg: 10.0,
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
    /// A3 stage-1 (`docs/visual_slam_sequential_sfm_plan.md`, "Stage-1
    /// baseline" -> "densify query cadence" slice): every call to
    /// [`DpvoLongLoopIndex::find_and_verify_long_range_loop`] counts as one
    /// ISSUED query, regardless of outcome — the SAME count
    /// `queries_attempted` already carries (this field mirrors it under the
    /// name `scripts/eval_dpvo_long_loop_recall.py`'s own `issued_query_count`
    /// metric expects), kept as its own field rather than a rename so
    /// existing `queries_attempted` call sites are untouched.
    pub queries_issued_total: usize,
    /// A3 stage-1: of `queries_issued_total`, how many returned ZERO
    /// candidates after [`DpvoLongLoopIndex::query_candidates`]'s own
    /// similarity/gap filtering — i.e. never even reached cross-check
    /// matching. Densifying `query_frequency` (the whole point of this
    /// slice) was previously invisible in `query_log`/the CSV export
    /// whenever a query landed empty (no row was ever written for it,
    /// indistinguishable from "never issued"); this counter, plus
    /// [`DpvoLongLoopIndex::empty_query_arrivals`], makes that case visible
    /// without changing `query_log`'s own per-candidate contract.
    pub queries_with_zero_candidates: usize,
    /// Total appearance candidates surfaced (summed across every query,
    /// after the similarity/gap gates, before geometric verification).
    pub candidates_considered: usize,
    /// Total candidates that reached geometric verification (cross-check
    /// matching + bridging + RANSAC).
    pub verification_attempts: usize,
    /// Milestone M12: candidates whose bridged 3D-3D correspondence count
    /// reached `min_bridge_correspondences` and so were actually handed to
    /// [`ransac_umeyama_scale`] — the funnel step between
    /// `verification_attempts` (bridge ATTEMPTED) and `accepted_total`
    /// (RANSAC PASSED). Because [`DpvoLongLoopIndex::find_and_verify_long_range_loop`]
    /// stops at the first accepted candidate, `accepted_total` IS this
    /// milestone's own "ransac-passed" count for a query that found any
    /// acceptable candidate at all (there is no separate "passed RANSAC but
    /// not the first one tried" case in this design) — this field's own
    /// purpose is isolating "insufficient bridge" from "bridge OK, RANSAC/
    /// residual rejected" from "bridge+RANSAC OK, rotation-inconsistent"
    /// (Milestone M12's own physical-consistency gate, added after a real
    /// 800f corruption run — see [`DpvoLongLoopConfig::max_rotation_inconsistency_deg`]'s
    /// own doc) among the `verification_attempts - accepted_total`
    /// candidates that were NOT accepted:
    /// `rejected_insufficient_bridge_total + bridge_sufficient_total ==
    /// verification_attempts`, `rejected_ransac_total +
    /// rejected_rotation_inconsistent_total + accepted_total ==
    /// bridge_sufficient_total` (per query where a candidate is accepted,
    /// the loop stops, so candidates ranked below it are never attempted at
    /// all and count toward none of these fields).
    pub bridge_sufficient_total: usize,
    pub accepted_total: usize,
    pub rejected_insufficient_bridge_total: usize,
    pub rejected_ransac_total: usize,
    /// Milestone M12: candidates that passed RANSAC/residual verification
    /// but were rejected because the RANSAC fit's own recovered rotation
    /// disagreed with DPVO's own trusted relative rotation by more than
    /// `max_rotation_inconsistency_deg` — see that config field's own doc.
    pub rejected_rotation_inconsistent_total: usize,
    pub last_accepted_arrival_i: usize,
    pub last_accepted_arrival_j: usize,
    pub last_accepted_gap: usize,
    pub last_accepted_similarity: f32,
    pub last_accepted_scale: f64,
    pub last_accepted_inliers: usize,
    pub last_accepted_mean_residual_ratio: f64,
    pub total_elapsed_ms: f64,
    /// A3 stage 2, first slice: whether [`DpvoLongLoopConfig::stage2_2d2d_geometry`]
    /// is on for this index.
    pub stage2_enabled: bool,
    /// Candidates for which the stage-2 2D-2D-first check was actually
    /// attempted (i.e. reached after `resolve_old` succeeded). Own funnel,
    /// independent of `verification_attempts`/`bridge_sufficient_total`
    /// (which now only count the EXISTING 3D-3D bridge, reached only after
    /// stage 2 passes when stage 2 is on):
    /// `stage2_passed_total + stage2_rejected_insufficient_matches_total +
    /// stage2_rejected_insufficient_inliers_total +
    /// stage2_rejected_insufficient_coverage_total +
    /// stage2_rejected_rotation_inconsistent_total +
    /// stage2_rejected_high_residual_total == stage2_attempts_total`.
    pub stage2_attempts_total: usize,
    /// Candidates that cleared every stage-2 gate (a-c) and proceeded to the
    /// existing 3D-3D bridge.
    pub stage2_passed_total: usize,
    /// Rejected: too few raw 2D-2D matches, or the essential-matrix RANSAC
    /// itself found no model.
    pub stage2_rejected_insufficient_matches_total: usize,
    /// Rejected: essential-matrix RANSAC found a model but its own inlier
    /// count is below `stage2_min_inliers`.
    pub stage2_rejected_insufficient_inliers_total: usize,
    /// Rejected: the 2D-2D inliers' own pixel bounding box (on the tighter
    /// of the OLD/NEW side) covers less than `stage2_min_coverage_fraction`
    /// of the frame's own patch-grid area.
    pub stage2_rejected_insufficient_coverage_total: usize,
    /// Rejected: the 2D-2D essential-matrix fit's own recovered rotation
    /// disagrees with DPVO's trusted relative rotation by more than
    /// `max_rotation_inconsistency_deg` (step (c) of this slice's own task
    /// brief — the SAME threshold/comparison the existing M12 3D-3D gate
    /// uses, reused rather than loosened).
    pub stage2_rejected_rotation_inconsistent_total: usize,
    /// Rejected: the 2D-2D essential-matrix fit's own mean Sampson error
    /// exceeds `stage2_max_mean_sampson_error`.
    pub stage2_rejected_high_residual_total: usize,
    /// Milestone A3 stage 2 step (d): candidates that passed EVERY prior
    /// gate (stage 2 a-c, the existing 3D-3D bridge/RANSAC/residual gates,
    /// and the existing M12 rotation-vs-trusted-pose gate) but were
    /// rejected because [`ransac_umeyama_scale`]'s own recovered rotation
    /// disagreed with the 2D-2D essential-matrix fit's own recovered
    /// rotation by more than `stage2_umeyama_vs_e_rotation_max_deg`.
    pub stage2_rejected_umeyama_vs_e_rotation_total: usize,
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

/// Milestone M12 (`docs/dpvo_droid_port_plan.md`, open item 2 carried
/// forward from M11): one top-`K` retrieval candidate FOR ONE QUERY, logged
/// unconditionally (accepted or not) — M11's own diagnostics recorded only
/// the last ACCEPTED candidate's identity, leaving "was the tightest GT
/// revisit ever even surfaced as a candidate" unanswerable from the run's
/// own data. [`DpvoLongLoopIndex::query_log`] accumulates one of these per
/// candidate returned by every query, across the whole run, so a post-hoc
/// pass (e.g. `examples/euroc_dpvo_vo_demo.rs`'s own CSV export) can grep for
/// a specific `candidate_arrival` and answer that question directly instead
/// of arguing from absence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryCandidateLogEntry {
    /// The querying (current, "new-side") frame's arrival index.
    pub query_arrival: usize,
    /// This candidate's ("old-side") arrival index.
    pub candidate_arrival: usize,
    /// `query_arrival - candidate_arrival` (always `>= min_temporal_gap`,
    /// per [`DpvoLongLoopIndex::query_candidates`]'s own filter).
    pub gap: usize,
    /// VLAD cosine similarity that ranked this candidate.
    pub similarity: f32,
    /// `0` = the highest-similarity candidate for this query, ascending.
    pub rank: usize,
    /// Whether THIS candidate is the one
    /// [`DpvoLongLoopIndex::find_and_verify_long_range_loop`] accepted for
    /// this query (at most one `true` entry per `query_arrival`) — `false`
    /// does not necessarily mean this candidate was geometrically verified
    /// and REJECTED: a candidate ranked below an accepted one is never
    /// attempted at all (the search stops at the first acceptance), so
    /// `false` covers both "tried and rejected" and "never tried because an
    /// earlier-ranked candidate already won."
    pub accepted: bool,
    /// Milestone M12 (real-corruption post-mortem instrumentation): the
    /// disagreement, in DEGREES, between [`ransac_umeyama_scale`]'s own
    /// recovered rotation and DPVO's already-trusted relative rotation
    /// (`current_pose.compose(&old_pose.inverse())`) — `Some(..)` whenever
    /// this candidate reached that check (bridging AND RANSAC both
    /// succeeded), regardless of whether it then passed or failed
    /// `max_rotation_inconsistency_deg`; `None` if it never reached that far
    /// (rejected earlier at bridging or RANSAC, or never attempted because
    /// an earlier-ranked candidate already won this query). This is the
    /// concrete, per-candidate number `docs/dpvo_droid_port_plan.md`'s "M12
    /// results" needed to state, not merely infer, why each rejected
    /// candidate was rejected.
    pub rotation_disagreement_deg: Option<f64>,
    /// A3 stage 2, first slice: essential-matrix RANSAC inlier count for
    /// this candidate's own 2D-2D fit — `Some(..)` whenever
    /// `stage2_2d2d_geometry` is on AND this candidate's raw 2D-2D matches
    /// were sufficient for [`visloc_vision::two_view::RelativePoseEstimator`]
    /// to return a fit at all (regardless of whether it then passed
    /// `stage2_min_inliers`); `None` when stage 2 is off, or the fit itself
    /// failed, or this candidate was never attempted (ranked below an
    /// earlier acceptance this query).
    pub stage2_2d2d_inliers: Option<usize>,
    /// A3 stage 2, first slice: disagreement, in DEGREES, between the 2D-2D
    /// essential-matrix fit's own recovered rotation and DPVO's
    /// already-trusted relative rotation — `Some(..)` whenever this
    /// candidate's 2D-2D fit reached the rotation check (i.e.
    /// `stage2_2d2d_inliers` cleared `stage2_min_inliers` and the coverage
    /// gate), regardless of whether it then passed `max_rotation_inconsistency_deg`.
    pub stage2_e_rotation_disagreement_deg: Option<f64>,
    /// A3 stage-2 convention diagnostic: the essential fit's recovered
    /// previous-to-current rotation as `(w, x, y, z)`. Logged independently
    /// of the trusted-pose comparison so a real pair can be checked against
    /// an external GT rotation without reconstructing the fit.
    pub stage2_e_rotation_wxyz: Option<[f64; 4]>,
    /// A3 stage-2 low-baseline diagnostic: COLMAP-style E/F/H classification
    /// (`not_run`, `undefined`, `degenerate`, `uncalibrated`, `calibrated`,
    /// `planar`, `panoramic`, `planar_or_panoramic`, `watermark`, or
    /// `multiple`).
    pub stage2_model: &'static str,
    /// Winning homography-model inlier count when `stage2_model` is planar,
    /// panoramic, or unresolved planar/panoramic.
    pub stage2_h_inliers: Option<usize>,
    /// Homography-decomposed previous-to-current rotation disagreement from
    /// DPVO's trusted relative rotation, when decomposition succeeded.
    pub stage2_h_rotation_disagreement_deg: Option<f64>,
    /// Acceptance-neutral continuation diagnostic: Umeyama fit scale after
    /// an E-vs-trusted rejection was deliberately carried forward solely
    /// for measurement. Never becomes a `Sim3LoopMeasurement`.
    pub stage2_diagnostic_umeyama_scale: Option<f64>,
    pub stage2_diagnostic_umeyama_inliers: Option<usize>,
    /// Rotation disagreement between the independent 2D-2D E fit and 3D-3D
    /// Umeyama fit in the acceptance-neutral continuation diagnostic.
    pub stage2_umeyama_vs_e_rotation_deg: Option<f64>,
    /// The furthest verification stage this candidate reached this query —
    /// see [`CandidateOutcome`]'s own doc for the full list of values. Kept
    /// as a plain string (not an enum) since this is purely a diagnostic/CSV
    /// field, never matched on downstream.
    pub stage_reached: &'static str,
    /// Alias for `accepted` under this slice's own CSV schema (the task's
    /// own "final accepted" column name) — always equal to `accepted`,
    /// kept as a separate field only so the new CSV columns this slice adds
    /// are self-describing without requiring a reader to already know the
    /// pre-existing `accepted` column's own name/semantics.
    pub final_accepted: bool,
}

/// Per-candidate outcome accumulated while [`DpvoLongLoopIndex::find_and_verify_long_range_loop`]
/// walks its own top-`K` candidate list — the source [`QueryCandidateLogEntry`]'s
/// new A3-stage-2 fields are built from. `stage_reached` is one of (in
/// funnel order): `"not_attempted"` (ranked below an earlier acceptance,
/// never reached at all), `"insufficient_2d2d_matches"`,
/// `"insufficient_2d2d_inliers"`, `"insufficient_2d2d_coverage"`,
/// `"2d2d_rotation_inconsistent"`, `"2d2d_high_residual"`,
/// `"insufficient_bridge"`, `"ransac_rejected"`, `"rotation_inconsistent"`
/// (the existing M12 gate, 3D-3D fit vs trusted pose),
/// `"umeyama_vs_e_rotation_inconsistent"` (A3 stage 2 step (d)), or
/// `"accepted"`.
#[derive(Debug, Clone, Copy)]
struct CandidateOutcome {
    stage2_2d2d_inliers: Option<usize>,
    stage2_e_rotation_disagreement_deg: Option<f64>,
    stage2_e_rotation_wxyz: Option<[f64; 4]>,
    stage2_model: &'static str,
    stage2_h_inliers: Option<usize>,
    stage2_h_rotation_disagreement_deg: Option<f64>,
    stage2_diagnostic_umeyama_scale: Option<f64>,
    stage2_diagnostic_umeyama_inliers: Option<usize>,
    stage2_umeyama_vs_e_rotation_deg: Option<f64>,
    rotation_disagreement_deg: Option<f64>,
    stage_reached: &'static str,
}

impl Default for CandidateOutcome {
    fn default() -> Self {
        Self {
            stage2_2d2d_inliers: None,
            stage2_e_rotation_disagreement_deg: None,
            stage2_e_rotation_wxyz: None,
            stage2_model: "not_run",
            stage2_h_inliers: None,
            stage2_h_rotation_disagreement_deg: None,
            stage2_diagnostic_umeyama_scale: None,
            stage2_diagnostic_umeyama_inliers: None,
            stage2_umeyama_vs_e_rotation_deg: None,
            rotation_disagreement_deg: None,
            stage_reached: "not_attempted",
        }
    }
}

/// One committed frame's retrieval+verification material. Images are never
/// retained (see the module doc); this is the compact per-frame summary
/// computed once, at commit time, from the (borrowed, transient) image.
#[derive(Debug, Clone, PartialEq)]
struct IndexedFrame {
    arrival_index: usize,
    /// The frame's global retrieval descriptor — VLAD or mean-pool,
    /// depending on `DpvoLongLoopConfig::retrieval_scorer` (see
    /// [`RetrievalScorer`]'s own doc); whichever it is, [`query_candidates`]
    /// scores every pair with plain [`cosine_similarity`], so this field's
    /// own name is scorer-agnostic on purpose.
    signature: Vec<f32>,
    /// Patch-grid-coordinate keypoints (already divided by `RES` by the
    /// caller — see [`DpvoLongLoopConfig::patch_pixel_radius`]'s own doc).
    keypoints: Vec<Point2<f64>>,
    descriptors: Vec<Vec<f32>>,
}

fn indexed_frame_bytes(frame: &IndexedFrame) -> usize {
    let descriptor_bytes: usize = frame
        .descriptors
        .iter()
        .map(|d| d.len() * std::mem::size_of::<f32>())
        .sum();
    let keypoint_bytes = frame.keypoints.len() * std::mem::size_of::<Point2<f64>>();
    let signature_bytes = frame.signature.len() * std::mem::size_of::<f32>();
    descriptor_bytes + keypoint_bytes + signature_bytes
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
    /// Milestone M12: every top-`K` candidate ever surfaced by a query, not
    /// just accepted ones — see [`QueryCandidateLogEntry`]'s own doc.
    /// Unbounded (like `dpvo_sim3_backend`'s own retained-pose history, this
    /// index's own regime is hundreds to low thousands of frames per run,
    /// not an unbounded streaming service), so a very long run's own memory
    /// cost here is `queries_attempted * top_k` entries — reported, not
    /// silently bounded, honestly documented rather than pretending it is
    /// free.
    query_log: Vec<QueryCandidateLogEntry>,
    /// A3 stage-1: arrivals where a query was issued (i.e. `due()` fired)
    /// but [`Self::query_candidates`] returned zero candidates — see
    /// [`DpvoLongLoopDiagnostics::queries_with_zero_candidates`]'s own doc
    /// for why this is tracked separately from `query_log` rather than
    /// mixed into it.
    empty_query_arrivals: Vec<usize>,

    diag_queries_attempted: usize,
    diag_queries_with_zero_candidates: usize,
    diag_candidates_considered: usize,
    diag_verification_attempts: usize,
    diag_bridge_sufficient: usize,
    diag_accepted_total: usize,
    diag_rejected_insufficient_bridge: usize,
    diag_rejected_ransac: usize,
    diag_rejected_rotation_inconsistent: usize,
    diag_last_arrival_i: usize,
    diag_last_arrival_j: usize,
    diag_last_gap: usize,
    diag_last_similarity: f32,
    diag_last_scale: f64,
    diag_last_inliers: usize,
    diag_last_mean_residual_ratio: f64,
    diag_total_elapsed_ms: f64,
    diag_stage2_attempts: usize,
    diag_stage2_passed: usize,
    diag_stage2_rejected_matches: usize,
    diag_stage2_rejected_inliers: usize,
    diag_stage2_rejected_coverage: usize,
    diag_stage2_rejected_rotation: usize,
    diag_stage2_rejected_residual: usize,
    diag_stage2_rejected_umeyama_vs_e_rotation: usize,
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
            query_log: Vec::new(),
            empty_query_arrivals: Vec::new(),
            diag_queries_attempted: 0,
            diag_queries_with_zero_candidates: 0,
            diag_candidates_considered: 0,
            diag_verification_attempts: 0,
            diag_bridge_sufficient: 0,
            diag_accepted_total: 0,
            diag_rejected_insufficient_bridge: 0,
            diag_rejected_ransac: 0,
            diag_rejected_rotation_inconsistent: 0,
            diag_last_arrival_i: 0,
            diag_last_arrival_j: 0,
            diag_last_gap: 0,
            diag_last_similarity: 0.0,
            diag_last_scale: 0.0,
            diag_last_inliers: 0,
            diag_last_mean_residual_ratio: 0.0,
            diag_total_elapsed_ms: 0.0,
            diag_stage2_attempts: 0,
            diag_stage2_passed: 0,
            diag_stage2_rejected_matches: 0,
            diag_stage2_rejected_inliers: 0,
            diag_stage2_rejected_coverage: 0,
            diag_stage2_rejected_rotation: 0,
            diag_stage2_rejected_residual: 0,
            diag_stage2_rejected_umeyama_vs_e_rotation: 0,
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
            queries_issued_total: self.diag_queries_attempted,
            queries_with_zero_candidates: self.diag_queries_with_zero_candidates,
            candidates_considered: self.diag_candidates_considered,
            verification_attempts: self.diag_verification_attempts,
            bridge_sufficient_total: self.diag_bridge_sufficient,
            accepted_total: self.diag_accepted_total,
            rejected_insufficient_bridge_total: self.diag_rejected_insufficient_bridge,
            rejected_ransac_total: self.diag_rejected_ransac,
            rejected_rotation_inconsistent_total: self.diag_rejected_rotation_inconsistent,
            last_accepted_arrival_i: self.diag_last_arrival_i,
            last_accepted_arrival_j: self.diag_last_arrival_j,
            last_accepted_gap: self.diag_last_gap,
            last_accepted_similarity: self.diag_last_similarity,
            last_accepted_scale: self.diag_last_scale,
            last_accepted_inliers: self.diag_last_inliers,
            last_accepted_mean_residual_ratio: self.diag_last_mean_residual_ratio,
            total_elapsed_ms: self.diag_total_elapsed_ms,
            stage2_enabled: self.config.stage2_2d2d_geometry,
            stage2_attempts_total: self.diag_stage2_attempts,
            stage2_passed_total: self.diag_stage2_passed,
            stage2_rejected_insufficient_matches_total: self.diag_stage2_rejected_matches,
            stage2_rejected_insufficient_inliers_total: self.diag_stage2_rejected_inliers,
            stage2_rejected_insufficient_coverage_total: self.diag_stage2_rejected_coverage,
            stage2_rejected_rotation_inconsistent_total: self.diag_stage2_rejected_rotation,
            stage2_rejected_high_residual_total: self.diag_stage2_rejected_residual,
            stage2_rejected_umeyama_vs_e_rotation_total: self
                .diag_stage2_rejected_umeyama_vs_e_rotation,
        }
    }

    /// Milestone M12 (open item 2 carried forward from M11): every top-`K`
    /// retrieval candidate ever surfaced by any query, across the whole
    /// run — see [`QueryCandidateLogEntry`]'s own doc for exactly what
    /// `accepted: false` does and does not mean.
    pub fn query_log(&self) -> &[QueryCandidateLogEntry] {
        &self.query_log
    }

    /// A3 stage-1: arrival indices where a query was issued but returned
    /// zero candidates — see [`DpvoLongLoopDiagnostics::queries_with_zero_candidates`]'s
    /// own doc. `examples/euroc_dpvo_vo_demo.rs`'s own CSV export appends one
    /// `rank=-1,candidate_arrival=-1,gap=-1,similarity=0.0,accepted=false`
    /// marker row per entry here.
    pub fn empty_query_arrivals(&self) -> &[usize] {
        &self.empty_query_arrivals
    }

    /// Ingest one committed frame's raw SuperPoint keypoints (already in
    /// patch-grid coordinates) + descriptors — called unconditionally, every
    /// committed frame, per the module doc's "images are never retained"
    /// constraint.
    ///
    /// [`RetrievalScorer::MeanPool`] (A3 ranking slice B): the frame's
    /// [`mean_pool`] signature is computed and indexed IMMEDIATELY — no
    /// vocabulary is ever trained or buffered-toward under this scorer, per
    /// the module doc's own "A3 ranking slice B" section.
    ///
    /// [`RetrievalScorer::Vlad`] (default, every prior milestone
    /// byte-for-byte unchanged): before a vocabulary exists, buffers the
    /// frame (bounded to `3 * vocab_bootstrap_frames`, oldest dropped first,
    /// as a safety valve against a pathological low-keypoint opening segment
    /// that never accumulates enough descriptors to build one — see the
    /// module doc's "Failure modes" section); once buffered, attempts
    /// [`Vocabulary::build`] every subsequent call until it succeeds, then
    /// retroactively VLAD-encodes every buffered frame. After a vocabulary
    /// exists, VLAD-encodes and indexes immediately.
    pub fn ingest_frame(
        &mut self,
        arrival_index: usize,
        keypoints: Vec<Point2<f64>>,
        descriptors: Vec<Vec<f32>>,
    ) {
        if descriptors.is_empty() {
            return;
        }
        if self.config.retrieval_scorer == RetrievalScorer::MeanPool {
            // A3 ranking slice B: no vocabulary involved at all — queryable
            // from the very first ingested frame.
            let signature = mean_pool(&descriptors);
            self.push_indexed(IndexedFrame {
                arrival_index,
                signature,
                keypoints,
                descriptors,
            });
            return;
        }

        let encoded = self.vocab.as_ref().map(|vocab| vlad(&descriptors, vocab));
        if let Some(signature) = encoded {
            self.push_indexed(IndexedFrame {
                arrival_index,
                signature,
                keypoints,
                descriptors,
            });
            return;
        }

        self.bootstrap.push((arrival_index, keypoints, descriptors));
        let safety_cap = self
            .config
            .vocab_bootstrap_frames
            .saturating_mul(3)
            .max(self.config.vocab_bootstrap_frames);
        while self.bootstrap.len() > safety_cap {
            self.bootstrap.remove(0);
        }
        if self.bootstrap.len() >= self.config.vocab_bootstrap_frames {
            self.try_build_vocab();
        }
    }

    fn try_build_vocab(&mut self) {
        let pooled: Vec<&[f32]> = self
            .bootstrap
            .iter()
            .flat_map(|(_, _, descriptors)| descriptors.iter().map(|d| d.as_slice()))
            .collect();
        let Some(vocab) = Vocabulary::build(
            &pooled,
            self.config.vocab_words,
            self.config.vocab_kmeans_iterations,
            self.config.vocab_seed,
        ) else {
            return; // Keep buffering — see `Self::ingest_frame`'s own doc.
        };
        let buffered = std::mem::take(&mut self.bootstrap);
        for (arrival_index, keypoints, descriptors) in buffered {
            let signature = vlad(&descriptors, &vocab);
            self.push_indexed(IndexedFrame {
                arrival_index,
                signature,
                keypoints,
                descriptors,
            });
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
            Some(last) => {
                current_arrival.saturating_sub(last) >= self.config.query_frequency.max(1)
            }
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
        let Some(current) = self
            .frames
            .iter()
            .rev()
            .find(|f| f.arrival_index == current_arrival)
        else {
            return Vec::new();
        };
        let min_gap = self.config.min_temporal_gap;
        let mut scored: Vec<(usize, f32)> = self
            .frames
            .iter()
            .filter(|f| {
                f.arrival_index != current_arrival
                    && current_arrival.saturating_sub(f.arrival_index) >= min_gap
            })
            .map(|f| {
                (
                    f.arrival_index,
                    cosine_similarity(&current.signature, &f.signature),
                )
            })
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
    #[allow(clippy::too_many_arguments)]
    pub fn find_and_verify_long_range_loop(
        &mut self,
        current_arrival: usize,
        current_pose: &SE3,
        current_intrinsics: &DpvoIntrinsics,
        current_patches: &[DpvoPatch],
        // A3 stage 2, first slice: the frame's own patch-grid extent (the
        // SAME stride-`RES` space `DpvoPatch::x`/`y` and this module's own
        // keypoints already live in — `crate::dpvo_vo::DpvoOdometry`'s own
        // `self.config.width/height / RES`), used ONLY by the stage-2
        // coverage gate (`stage2_min_coverage_fraction`) to turn an inlier
        // bounding box into a FRACTION of the image area; unused (any finite
        // value is fine) when `stage2_2d2d_geometry` is `false`.
        grid_width: f64,
        grid_height: f64,
        mut resolve_old: impl FnMut(usize) -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)>,
    ) -> Option<AcceptedLongLoop> {
        let start = Instant::now();
        self.diag_queries_attempted += 1;

        let candidates = self.query_candidates(current_arrival);
        self.diag_candidates_considered += candidates.len();
        if candidates.is_empty() {
            // A3 stage-1: an issued query that surfaced nothing at all is
            // otherwise silent (no `query_log` row, indistinguishable from
            // "never issued") — record it explicitly so densifying
            // `query_frequency` doesn't hide how much of that density lands
            // empty.
            self.diag_queries_with_zero_candidates += 1;
            self.empty_query_arrivals.push(current_arrival);
            self.diag_total_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
            return None;
        }

        let Some(current_idx) = self
            .frames
            .iter()
            .position(|f| f.arrival_index == current_arrival)
        else {
            self.diag_total_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
            return None;
        };
        let current_keypoints = self.frames[current_idx].keypoints.clone();
        let current_descriptors = self.frames[current_idx].descriptors.clone();
        let matcher = CrossCheckMatcher::new(BruteForceMatcher {
            ratio: self.config.match_ratio,
        });

        let mut accepted = None;
        let mut winning_arrival: Option<usize> = None;
        // A3 stage 2 (first slice): per-candidate outcome, keyed by
        // `old_arrival` — see `CandidateOutcome`'s own doc. Replaces M12's
        // narrower `rotation_checks: Vec<(usize, f64)>` (still covers that
        // exact same field, `rotation_disagreement_deg`, plus the new
        // stage-2 fields).
        let mut outcomes: Vec<(usize, CandidateOutcome)> = Vec::new();
        for &(old_arrival, similarity) in &candidates {
            let Some(old_idx) = self
                .frames
                .iter()
                .position(|f| f.arrival_index == old_arrival)
            else {
                continue;
            };
            let old_keypoints = self.frames[old_idx].keypoints.clone();
            let old_descriptors = self.frames[old_idx].descriptors.clone();
            let Some((old_pose, old_intr, old_patches)) = resolve_old(old_arrival) else {
                continue;
            };
            let mut outcome = CandidateOutcome::default();

            // DPVO's own trusted relative rotation — used by the EXISTING
            // M12 3D-3D gate below AND (when stage 2 is on) the NEW 2D-2D
            // gate (c), computed once and shared by both.
            let relative_pose = current_pose.compose(&old_pose.inverse());

            // A3 stage 2 (`docs/visual_slam_sequential_sfm_plan.md`, "A3 —
            // Sound long-range loop closure", stage 2): 2D-2D-FIRST loop
            // geometry, BEFORE the existing 3D-3D bridge ever runs — see
            // `DpvoLongLoopConfig::stage2_2d2d_geometry`'s own doc for the
            // motivation. `stage2_e_rotation` carries the fit's own
            // recovered rotation forward to step (d)'s gate, below the
            // bridge.
            let mut stage2_e_rotation: Option<UnitQuaternion<f64>> = None;
            // True only in the opt-in acceptance-neutral diagnostic: carry
            // an E-vs-trusted rejection through the existing bridge solely
            // to measure independent E-vs-Umeyama agreement. Such a
            // candidate is unconditionally rejected before measurement
            // construction below.
            let mut diagnostic_stage2_rotation_failure = false;
            if self.config.stage2_2d2d_geometry {
                self.diag_stage2_attempts += 1;
                // (a) Match stored SP descriptors 2D-2D: mutual
                // nearest-neighbor + Lowe ratio (`stage2_match_ratio`,
                // default `0.9`) — a SEPARATE cross-check match from the
                // existing 3D-3D bridge's own `matcher` above (different
                // ratio), per this slice's own task brief.
                let stage2_matcher = CrossCheckMatcher::new(BruteForceMatcher {
                    ratio: self.config.stage2_match_ratio,
                });
                let stage2_matches: Vec<DescriptorMatch> =
                    stage2_matcher.match_descriptors(&old_descriptors, &current_descriptors);
                // Keep a parallel `kept` slice so `RelativePose::inliers`
                // (indices into `correspondences`) stay 1:1 addressable back
                // into the ORIGINAL match (`query_index`/`train_index`),
                // mirroring `bridge_matches_to_3d3d`'s own "skip an
                // unresolvable match, keep going" style rather than assuming
                // every raw match index resolves.
                let mut stage2_kept: Vec<&DescriptorMatch> = Vec::new();
                let mut correspondences: Vec<TwoViewCorrespondence> = Vec::new();
                for m in &stage2_matches {
                    let (Some(&kp_o), Some(&kp_n)) = (
                        old_keypoints.get(m.query_index),
                        current_keypoints.get(m.train_index),
                    ) else {
                        continue;
                    };
                    correspondences.push(TwoViewCorrespondence::new(kp_o, kp_n));
                    stage2_kept.push(m);
                }

                // (b) Calibrated relative pose: reuse
                // `visloc_vision::two_view`'s own essential-matrix RANSAC +
                // cheirality-checked decomposition wholesale (this port's
                // existing fundamental/essential machinery already supports
                // the calibrated case directly — no reimplementation
                // needed). `current_intrinsics` is used for BOTH sides: DPVO
                // runs a single static camera for the whole sequence, so
                // `old_intr`/`current_intrinsics` are the same values by
                // construction (both derived once from
                // `DpvoOdometryConfig::intrinsics` at `DpvoOdometry::new`).
                let camera = Camera::pinhole(
                    0,
                    grid_width.max(1.0) as u32,
                    grid_height.max(1.0) as u32,
                    current_intrinsics.fx,
                    current_intrinsics.fy,
                    current_intrinsics.cx,
                    current_intrinsics.cy,
                );
                if self.config.stage2_low_baseline_diagnostic {
                    // Diagnostic-only E/F/H competition. One patch-grid
                    // pixel is eight full-resolution pixels under DPVO's
                    // fixed RES=8 representation: deliberately tighter than
                    // the verifier's generic 4px default in THIS coordinate
                    // space, while still broad enough for real SP matches.
                    let mut options = TwoViewGeometryOptions::for_camera(&camera, 1.0);
                    options.min_num_inliers = self.config.stage2_min_inliers;
                    let report = TwoViewGeometryVerifier::new(options)
                        .classify(&correspondences, &camera);
                    outcome.stage2_model = match report.config {
                        ConfigurationType::Undefined => "undefined",
                        ConfigurationType::Degenerate => "degenerate",
                        ConfigurationType::Uncalibrated => "uncalibrated",
                        ConfigurationType::Calibrated => "calibrated",
                        ConfigurationType::Planar => "planar",
                        ConfigurationType::Panoramic => "panoramic",
                        ConfigurationType::PlanarOrPanoramic => "planar_or_panoramic",
                        ConfigurationType::Watermark => "watermark",
                        ConfigurationType::Multiple => "multiple",
                    };
                    if matches!(
                        report.config,
                        ConfigurationType::Planar
                            | ConfigurationType::Panoramic
                            | ConfigurationType::PlanarOrPanoramic
                    ) {
                        outcome.stage2_h_inliers = Some(report.inliers.len());
                    }
                    if let Some((rotation, _translation)) = report.relative_pose {
                        let h_rotation = UnitQuaternion::from_rotation_matrix(
                            &Rotation3::from_matrix_unchecked(rotation),
                        );
                        outcome.stage2_h_rotation_disagreement_deg = Some(
                            h_rotation.angle_to(&relative_pose.rotation).to_degrees(),
                        );
                    }
                }
                let stage2_result =
                    RelativePoseEstimator::default().estimate(&correspondences, &camera);
                'stage2: {
                    let Some(rel) = stage2_result else {
                        self.diag_stage2_rejected_matches += 1;
                        outcome.stage_reached = "insufficient_2d2d_matches";
                        break 'stage2;
                    };
                    outcome.stage2_2d2d_inliers = Some(rel.inliers.len());
                    let e_quaternion = rel.previous_to_current.rotation.quaternion();
                    outcome.stage2_e_rotation_wxyz = Some([
                        e_quaternion.w,
                        e_quaternion.i,
                        e_quaternion.j,
                        e_quaternion.k,
                    ]);
                    if rel.inliers.len() < self.config.stage2_min_inliers {
                        self.diag_stage2_rejected_inliers += 1;
                        outcome.stage_reached = "insufficient_2d2d_inliers";
                        break 'stage2;
                    }

                    // (c) Gate: inlier spatial coverage — the tighter of the
                    // OLD/NEW side's own inlier bounding box must still
                    // cover >= `stage2_min_coverage_fraction` of the frame's
                    // own patch-grid area.
                    let old_inlier_idx = rel.inliers.iter().map(|&k| stage2_kept[k].query_index);
                    let new_inlier_idx = rel.inliers.iter().map(|&k| stage2_kept[k].train_index);
                    let coverage_old = inlier_bbox_area_fraction(
                        &old_keypoints,
                        old_inlier_idx,
                        grid_width,
                        grid_height,
                    );
                    let coverage_new = inlier_bbox_area_fraction(
                        &current_keypoints,
                        new_inlier_idx,
                        grid_width,
                        grid_height,
                    );
                    if coverage_old.min(coverage_new) < self.config.stage2_min_coverage_fraction {
                        self.diag_stage2_rejected_coverage += 1;
                        outcome.stage_reached = "insufficient_2d2d_coverage";
                        break 'stage2;
                    }

                    // (c) Gate: relative rotation from E must agree with the
                    // EXISTING DPVO-trusted-rotation gate — SAME threshold
                    // (`max_rotation_inconsistency_deg`) and SAME comparison
                    // shape (`UnitQuaternion::angle_to`) the pre-existing
                    // M12 3D-3D gate below already uses; not a new, looser
                    // threshold.
                    let e_rotation = rel.previous_to_current.rotation;
                    let e_rotation_disagreement_deg =
                        e_rotation.angle_to(&relative_pose.rotation).to_degrees();
                    outcome.stage2_e_rotation_disagreement_deg = Some(e_rotation_disagreement_deg);
                    if e_rotation_disagreement_deg > self.config.max_rotation_inconsistency_deg {
                        self.diag_stage2_rejected_rotation += 1;
                        outcome.stage_reached = "2d2d_rotation_inconsistent";
                        if self.config.stage2_low_baseline_diagnostic {
                            diagnostic_stage2_rotation_failure = true;
                            stage2_e_rotation = Some(e_rotation);
                        } else {
                            break 'stage2;
                        }
                    }

                    // (c) Gate: epipolar residual sane.
                    if !rel.mean_sampson_error.is_finite()
                        || rel.mean_sampson_error > self.config.stage2_max_mean_sampson_error
                    {
                        if diagnostic_stage2_rotation_failure {
                            // The diagnostic continuation discovered a
                            // later residual failure after this candidate
                            // had already been provisionally counted in the
                            // rotation bucket. Keep the documented stage-2
                            // funnel partition disjoint.
                            self.diag_stage2_rejected_rotation =
                                self.diag_stage2_rejected_rotation.saturating_sub(1);
                            diagnostic_stage2_rotation_failure = false;
                        }
                        self.diag_stage2_rejected_residual += 1;
                        outcome.stage_reached = "2d2d_high_residual";
                        stage2_e_rotation = None;
                        break 'stage2;
                    }

                    if !diagnostic_stage2_rotation_failure {
                        self.diag_stage2_passed += 1;
                        outcome.stage_reached = "stage2_passed";
                        stage2_e_rotation = Some(e_rotation);
                    }
                }
                if stage2_e_rotation.is_none() {
                    // (a)-(c) failed: per this slice's own task brief, do
                    // NOT proceed to the existing 3D-3D bridge at all.
                    outcomes.push((old_arrival, outcome));
                    continue;
                }
            }

            // (d) Only if stage 2 is off, or on AND (a)-(c) passed: proceed
            // to the EXISTING 3D-3D bridge + Umeyama Sim(3) scale estimate,
            // byte-identical to M11/M12.
            self.diag_verification_attempts += 1;
            let matches = matcher.match_descriptors(&old_descriptors, &current_descriptors);
            let raw_pairs: Vec<(usize, usize)> = matches
                .iter()
                .map(|m| (m.query_index, m.train_index))
                .collect();
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
                outcome.stage_reached = "insufficient_bridge";
                outcomes.push((old_arrival, outcome));
                continue;
            }
            self.diag_bridge_sufficient += 1;

            let Some(fit) = ransac_umeyama_scale(&bridged, &self.config, &mut self.rng) else {
                self.diag_rejected_ransac += 1;
                outcome.stage_reached = "ransac_rejected";
                outcomes.push((old_arrival, outcome));
                continue;
            };

            // Milestone M12 (post-mortem on a real 800f corruption):
            // RANSAC's own inlier-count/residual-ratio gates check only the
            // bridged sample's own internal self-consistency — they cannot
            // tell a genuine revisit from an appearance-similar-but-
            // structurally-different one that happens to produce enough
            // mutually-agreeing (but jointly wrong) correspondences. Cross-
            // check the fit's own recovered ROTATION against DPVO's already-
            // trusted relative rotation (the SAME one `relative_pose` above
            // already carries) as an independent physical consistency gate —
            // see `GeometricFit::rotation`'s and
            // `DpvoLongLoopConfig::max_rotation_inconsistency_deg`'s own doc.
            let fit_rotation_uq = UnitQuaternion::from_rotation_matrix(&fit.rotation);
            if diagnostic_stage2_rotation_failure {
                outcome.stage2_diagnostic_umeyama_scale = Some(fit.scale);
                outcome.stage2_diagnostic_umeyama_inliers = Some(fit.inlier_count);
            }
            let rotation_disagreement_deg = fit_rotation_uq
                .angle_to(&relative_pose.rotation)
                .to_degrees();
            outcome.rotation_disagreement_deg = Some(rotation_disagreement_deg);
            if rotation_disagreement_deg > self.config.max_rotation_inconsistency_deg {
                if !diagnostic_stage2_rotation_failure {
                    self.diag_rejected_rotation_inconsistent += 1;
                    outcome.stage_reached = "rotation_inconsistent";
                    outcomes.push((old_arrival, outcome));
                    continue;
                }
            }

            // A3 stage 2 step (d): ADDITIONALLY require the Umeyama
            // rotation to agree with the 2D-2D E-derived rotation, within a
            // TIGHTER bound (`stage2_umeyama_vs_e_rotation_max_deg`, default
            // `10.0°`) than `max_rotation_inconsistency_deg`'s `20.0°` —
            // only evaluated when stage 2 actually ran (there is no
            // independent E-matrix rotation to compare against otherwise).
            if let Some(e_rotation) = stage2_e_rotation {
                let umeyama_vs_e_deg = fit_rotation_uq.angle_to(&e_rotation).to_degrees();
                outcome.stage2_umeyama_vs_e_rotation_deg = Some(umeyama_vs_e_deg);
                if umeyama_vs_e_deg > self.config.stage2_umeyama_vs_e_rotation_max_deg {
                    if !diagnostic_stage2_rotation_failure {
                        self.diag_stage2_rejected_umeyama_vs_e_rotation += 1;
                        outcome.stage_reached = "umeyama_vs_e_rotation_inconsistent";
                    } else {
                        outcome.stage_reached = "diagnostic_umeyama_vs_e_inconsistent";
                    }
                    outcomes.push((old_arrival, outcome));
                    continue;
                }
            }

            if diagnostic_stage2_rotation_failure {
                // Measurement-only continuation: even an E/Umeyama-consistent
                // candidate is never accepted in this diagnostic slice.
                outcome.stage_reached = "diagnostic_umeyama_vs_e_consistent";
                outcomes.push((old_arrival, outcome));
                continue;
            }

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
            outcome.stage_reached = "accepted";
            outcomes.push((old_arrival, outcome));
            accepted = Some(AcceptedLongLoop {
                arrival_i: old_arrival,
                arrival_j: current_arrival,
                measurement,
            });
            winning_arrival = Some(old_arrival);
            break;
        }

        // Milestone M12 (open item 2 carried forward from M11): log EVERY
        // top-`K` candidate this query surfaced, not just the accepted one —
        // see `QueryCandidateLogEntry`'s own doc for what `accepted: false`
        // does and does not mean (a candidate ranked below the winner is
        // never even attempted, per this function's own "stop at first
        // accepted" design, and is still logged as `accepted: false`,
        // `stage_reached: "not_attempted"`).
        for (rank, &(candidate_arrival, similarity)) in candidates.iter().enumerate() {
            let outcome = outcomes
                .iter()
                .find(|&&(arrival, _)| arrival == candidate_arrival)
                .map(|&(_, o)| o)
                .unwrap_or_default();
            let is_accepted = winning_arrival == Some(candidate_arrival);
            self.query_log.push(QueryCandidateLogEntry {
                query_arrival: current_arrival,
                candidate_arrival,
                gap: current_arrival.saturating_sub(candidate_arrival),
                similarity,
                rank,
                accepted: is_accepted,
                rotation_disagreement_deg: outcome.rotation_disagreement_deg,
                stage2_2d2d_inliers: outcome.stage2_2d2d_inliers,
                stage2_e_rotation_disagreement_deg: outcome.stage2_e_rotation_disagreement_deg,
                stage2_e_rotation_wxyz: outcome.stage2_e_rotation_wxyz,
                stage2_model: outcome.stage2_model,
                stage2_h_inliers: outcome.stage2_h_inliers,
                stage2_h_rotation_disagreement_deg: outcome
                    .stage2_h_rotation_disagreement_deg,
                stage2_diagnostic_umeyama_scale: outcome.stage2_diagnostic_umeyama_scale,
                stage2_diagnostic_umeyama_inliers: outcome.stage2_diagnostic_umeyama_inliers,
                stage2_umeyama_vs_e_rotation_deg: outcome.stage2_umeyama_vs_e_rotation_deg,
                stage_reached: outcome.stage_reached,
                final_accepted: is_accepted,
            });
        }

        self.diag_total_elapsed_ms += start.elapsed().as_secs_f64() * 1000.0;
        accepted
    }
}

/// Milestone M12 (`docs/dpvo_droid_port_plan.md`): choose a frame's
/// `patches_per_frame` patch centers, optionally anchored at SuperPoint
/// keypoint locations instead of pure uniform-random sampling — the "attack
/// the bridge by construction" fix for M11's own honest negative (at
/// `fast.yaml` patch density, a matched appearance keypoint essentially
/// never lands near an existing randomly-placed patch, so
/// [`bridge_matches_to_3d3d`] almost never finds enough 3D-3D
/// correspondences). Anchoring patches AT the SAME keypoints a future
/// long-range match will look for makes the bridge succeed by CONSTRUCTION,
/// not by loosening `patch_pixel_radius` (M11's own module doc + results
/// section: loosening the radius instead of the density is a confirmed
/// corruption mode, never to be reopened).
///
/// # Coordinate mapping
///
/// `sp_keypoints` is `(x, y, score)` in FULL-RESOLUTION image pixels — the
/// SAME raw form `crate::dpvo_vo::DpvoOdometry::process_frame` already
/// extracts once (via SuperPoint) and reuses BOTH here and for
/// [`DpvoLongLoopIndex::ingest_frame`]'s own `/RES` conversion (never
/// extracted twice per frame). Each keypoint is divided by `res` here —
/// identical to `ingest_frame`'s own conversion, using the SAME
/// `visloc_vision::dpvo::RES` value the caller already has (passed in
/// rather than imported directly: this module is deliberately
/// `onnx-inference`-feature-agnostic — see the module doc's own "graph/
/// policy only" placement note — while `visloc_vision::dpvo` itself is
/// gated behind that feature) — to reach the SAME stride-`RES` patch-grid
/// space `DpvoPatch::x`/`y` and this frame's `coords` already live in, then
/// CLAMPED (never rejected) into `[1, ws - 2] x [1, hs - 2]`: the exact
/// integer interior the legacy uniform-random sampler's own
/// `rng.gen_range(1..ws - 1)` (`1 ..= ws - 2` inclusive) already enforces,
/// so an SP-anchored center can never fall outside the border margin every
/// prior milestone's patches already respected. Sub-pixel precision is
/// intentionally PRESERVED (not rounded to the integer lattice):
/// `patchify_cpu`'s own bilinear blend already handles a fractional
/// centroid correctly (the exact same interpolation path every M1-M11 patch
/// already exercises once its depth/pose estimate updates), so there is no
/// reason to discard the keypoint's true sub-pixel location.
///
/// # Ranking, de-duplication, and fallback
///
/// Keypoints are ranked by SCORE, descending, and accepted greedily up to
/// `m` as long as each is at least `min_separation` patch-grid pixels from
/// every already-chosen center (see [`DpvoLongLoopConfig::sp_patch_min_separation`]'s
/// own doc for why) — a simple, deterministic (score-order, not spatial
/// binning) de-duplication, not a full non-max-suppression grid. Any
/// shortfall (`sp_keypoints` empty, `ws`/`hs` too small for any margin at
/// all, or fewer than `m` keypoints survive de-duplication) is filled by the
/// EXACT SAME uniform-random sampler M1-M11 already used
/// (`rng.gen_range(1..ws - 1)`/`rng.gen_range(1..hs - 1)`, same call order),
/// so `sp_keypoints: &[]` reproduces the legacy fully-random sampling
/// byte-for-byte (same RNG call sequence, same values) — the M12 "off =
/// legacy" contract this function alone is responsible for.
#[allow(clippy::too_many_arguments)]
pub fn sp_anchored_patch_centers(
    m: usize,
    ws: usize,
    hs: usize,
    sp_keypoints: &[(f64, f64, f32)],
    res: f64,
    min_separation: f64,
    rng: &mut StdRng,
) -> Vec<(f32, f32)> {
    let mut chosen: Vec<(f32, f32)> = Vec::with_capacity(m);
    if !sp_keypoints.is_empty() && ws > 2 && hs > 2 {
        let lo_x = 1.0_f64;
        let hi_x = (ws as f64) - 2.0;
        let lo_y = 1.0_f64;
        let hi_y = (hs as f64) - 2.0;
        let mut ranked: Vec<(f64, f64, f32)> = sp_keypoints
            .iter()
            .map(|&(x, y, score)| {
                let gx = (x / res).clamp(lo_x, hi_x);
                let gy = (y / res).clamp(lo_y, hi_y);
                (gx, gy, score)
            })
            .collect();
        ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        for (gx, gy, _score) in ranked {
            if chosen.len() >= m {
                break;
            }
            let too_close = chosen.iter().any(|&(cx, cy)| {
                let dx = cx as f64 - gx;
                let dy = cy as f64 - gy;
                (dx * dx + dy * dy).sqrt() < min_separation
            });
            if too_close {
                continue;
            }
            chosen.push((gx as f32, gy as f32));
        }
    }
    // Fallback — byte-identical to the legacy `Patchifier.forward` RANDOM
    // strategy's own call sequence when `chosen` is (still) empty here.
    while chosen.len() < m {
        let x = rng.gen_range(1..ws - 1) as f32;
        let y = rng.gen_range(1..hs - 1) as f32;
        chosen.push((x, y));
    }
    chosen
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
fn patch_to_world_point(
    pose: &SE3,
    intr: &DpvoIntrinsics,
    patch: &DpvoPatch,
) -> Option<Point3<f64>> {
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
fn nearest_patch_within<'a>(
    kp: &Point2<f64>,
    patches: &'a [DpvoPatch],
    radius: f64,
) -> Option<&'a DpvoPatch> {
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

/// A3 stage 2, first slice: bounding-box area of the keypoints at `indices`,
/// as a FRACTION of `grid_w * grid_h` (the frame's own patch-grid area,
/// SAME coordinate space `keypoints` already live in) — the stage-2
/// coverage gate's own core measurement (see
/// [`DpvoLongLoopConfig::stage2_min_coverage_fraction`]'s own doc). Returns
/// `0.0` for fewer than 2 points, or a non-positive grid extent (a
/// degenerate/zero-area box, always fails the gate rather than panicking or
/// dividing by zero).
fn inlier_bbox_area_fraction(
    keypoints: &[Point2<f64>],
    indices: impl Iterator<Item = usize>,
    grid_w: f64,
    grid_h: f64,
) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut count = 0usize;
    for index in indices {
        let Some(p) = keypoints.get(index) else {
            continue;
        };
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
        count += 1;
    }
    if count < 2 || grid_w <= 0.0 || grid_h <= 0.0 {
        return 0.0;
    }
    let area = (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0);
    (area / (grid_w * grid_h)).clamp(0.0, 1.0)
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
        let (Some(kp_o), Some(kp_n)) = (old_keypoints.get(oi), new_keypoints.get(ni)) else {
            continue;
        };
        let (Some(p_o), Some(p_n)) = (
            nearest_patch_within(kp_o, old_patches, radius),
            nearest_patch_within(kp_n, new_patches, radius),
        ) else {
            continue;
        };
        let (Some(w_o), Some(w_n)) = (
            patch_to_world_point(old_pose, old_intr, p_o),
            patch_to_world_point(new_pose, new_intr, p_n),
        ) else {
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

fn count_inliers(
    fit: &TrajectorySimilarityTransform,
    pairs: &[(Point3<f64>, Point3<f64>)],
    threshold: f64,
) -> usize {
    pairs
        .iter()
        .filter(|(source, target)| (predict(fit, source) - target).norm() <= threshold)
        .count()
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
    /// Milestone M12 (measurement-vs-application post-mortem): the refit's
    /// own recovered ROTATION — previously computed and immediately
    /// discarded (the caller always reused DPVO's own trusted
    /// `current_pose.compose(&old_pose.inverse())` rotation for the ordinary
    /// `Sim3LoopMeasurement` edge, per this module's own "DPVO's rotation is
    /// more reliable than its translation scale" design choice). Propagated
    /// out here so [`DpvoLongLoopIndex::find_and_verify_long_range_loop`] can
    /// cross-check it against that SAME trusted rotation
    /// (`DpvoLongLoopConfig::max_rotation_inconsistency_deg`) — RANSAC's own
    /// inlier-count/residual-ratio gates check only the bridged SAMPLE's own
    /// internal self-consistency, never agreement with an independent,
    /// generally-reliable trajectory-level rotation estimate; a large
    /// disagreement is strong evidence the correspondence set does not
    /// describe the same physical revisit even when internally coherent.
    rotation: Rotation3<f64>,
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
fn ransac_umeyama_scale(
    pairs: &[(Point3<f64>, Point3<f64>)],
    cfg: &DpvoLongLoopConfig,
    rng: &mut StdRng,
) -> Option<GeometricFit> {
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
        let Some(sample) = sample_three_distinct(n, rng, 50) else {
            break;
        };
        let source: Vec<Point3<f64>> = sample.iter().map(|&i| pairs[i].0).collect();
        let target: Vec<Point3<f64>> = sample.iter().map(|&i| pairs[i].1).collect();
        let Some(fit) = umeyama_similarity_transform(&source, &target, true) else {
            continue;
        };
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

    let inlier_pairs: Vec<(Point3<f64>, Point3<f64>)> = pairs
        .iter()
        .copied()
        .filter(|(source, target)| (predict(&fit0, source) - target).norm() <= inlier_threshold)
        .collect();
    if inlier_pairs.len() < cfg.min_ransac_inliers {
        return None;
    }
    let source: Vec<Point3<f64>> = inlier_pairs.iter().map(|(s, _)| *s).collect();
    let target: Vec<Point3<f64>> = inlier_pairs.iter().map(|(_, t)| *t).collect();
    let refit = umeyama_similarity_transform(&source, &target, true)?;
    if !(refit.scale.is_finite() && refit.scale >= cfg.min_scale && refit.scale <= cfg.max_scale) {
        return None;
    }

    let mean_residual: f64 = inlier_pairs
        .iter()
        .map(|(source, target)| (predict(&refit, source) - target).norm())
        .sum::<f64>()
        / inlier_pairs.len() as f64;
    let mean_residual_ratio = mean_residual / scene_scale;
    if mean_residual_ratio > cfg.max_mean_residual_ratio {
        return None;
    }

    Some(GeometricFit {
        scale: refit.scale,
        rotation: refit.rotation,
        inlier_count: inlier_pairs.len(),
        mean_residual_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intr() -> DpvoIntrinsics {
        DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 64.0,
            cy: 48.0,
        }
    }

    fn cfg() -> DpvoLongLoopConfig {
        DpvoLongLoopConfig::default()
    }

    // ---- sp_anchored_patch_centers (Milestone M12) ----

    /// Mirrors `visloc_vision::dpvo::RES` (`4`) — this module cannot import
    /// that constant directly (it lives behind the `onnx-inference` feature;
    /// see [`sp_anchored_patch_centers`]'s own doc), so its tests hardcode
    /// the same value the real caller (`crate::dpvo_vo::DpvoOdometry::process_frame`)
    /// passes in.
    const TEST_RES: f64 = 4.0;

    #[test]
    fn sp_anchored_patch_centers_with_no_keypoints_matches_legacy_random_sampling_exactly() {
        // `sp_keypoints: &[]` must reproduce the EXACT legacy
        // `Patchifier.forward` RANDOM sampling call sequence: same number of
        // `gen_range` calls, in the same (x, y) order, so a seeded RNG
        // produces byte-identical coordinates either way — the M12 "off =
        // legacy" contract.
        let (ws, hs, m) = (188usize, 120usize, 48usize);
        let mut rng_a = StdRng::seed_from_u64(42);
        let legacy: Vec<(f32, f32)> = (0..m)
            .map(|_| {
                let x = rng_a.gen_range(1..ws - 1) as f32;
                let y = rng_a.gen_range(1..hs - 1) as f32;
                (x, y)
            })
            .collect();
        let mut rng_b = StdRng::seed_from_u64(42);
        let via_helper = sp_anchored_patch_centers(m, ws, hs, &[], TEST_RES, 2.0, &mut rng_b);
        assert_eq!(legacy, via_helper);
    }

    #[test]
    fn sp_anchored_patch_centers_places_centers_at_given_keypoints() {
        // Keypoints are given in FULL-RESOLUTION pixels; `RES` (4) divides
        // them down to patch-grid space — pick values that land safely away
        // from the border after that division so no clamping kicks in,
        // making the expected output hand-checkable.
        let kps = [
            (40.0 * TEST_RES, 30.0 * TEST_RES, 0.9_f32),
            (50.0 * TEST_RES, 35.0 * TEST_RES, 0.8_f32),
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let centers = sp_anchored_patch_centers(2, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 2);
        assert!(
            (centers[0].0 - 40.0).abs() < 1e-6 && (centers[0].1 - 30.0).abs() < 1e-6,
            "{centers:?}"
        );
        assert!(
            (centers[1].0 - 50.0).abs() < 1e-6 && (centers[1].1 - 35.0).abs() < 1e-6,
            "{centers:?}"
        );
    }

    #[test]
    fn sp_anchored_patch_centers_ranks_by_score_descending() {
        // Three keypoints, deliberately given LOWEST-score-first, spaced far
        // enough apart that de-duplication never removes any of them.
        let kps = [
            (20.0 * TEST_RES, 20.0 * TEST_RES, 0.1_f32),
            (80.0 * TEST_RES, 20.0 * TEST_RES, 0.9_f32),
            (140.0 * TEST_RES, 20.0 * TEST_RES, 0.5_f32),
        ];
        let mut rng = StdRng::seed_from_u64(2);
        // Only room for the top 2 by score: (80,20) then (140,20).
        let centers = sp_anchored_patch_centers(2, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 2);
        assert!((centers[0].0 - 80.0).abs() < 1e-6, "{centers:?}");
        assert!((centers[1].0 - 140.0).abs() < 1e-6, "{centers:?}");
    }

    #[test]
    fn sp_anchored_patch_centers_clamps_into_the_legacy_border_margin() {
        // A keypoint at the full-res image origin maps to patch-grid (0, 0)
        // — outside the legacy `[1, ws-2] x [1, hs-2]` interior — and must be
        // clamped to (1, 1), never rejected outright.
        let kps = [(0.0, 0.0, 1.0_f32)];
        let mut rng = StdRng::seed_from_u64(3);
        let centers = sp_anchored_patch_centers(1, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 1);
        assert!(
            (centers[0].0 - 1.0).abs() < 1e-6 && (centers[0].1 - 1.0).abs() < 1e-6,
            "{centers:?}"
        );
    }

    #[test]
    fn sp_anchored_patch_centers_deduplicates_within_min_separation() {
        // Two keypoints mapping to (almost) the same patch-grid cell — only
        // the higher-scored one should be kept; the second slot falls back
        // to random sampling rather than a near-duplicate center.
        let kps = [
            (40.0 * TEST_RES, 30.0 * TEST_RES, 0.9_f32),
            (40.5 * TEST_RES, 30.5 * TEST_RES, 0.95_f32),
        ];
        let mut rng = StdRng::seed_from_u64(4);
        let centers = sp_anchored_patch_centers(2, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 2);
        // The higher-scored keypoint (40.5, 30.5) wins the first slot.
        assert!(
            (centers[0].0 - 40.5).abs() < 1e-6 && (centers[0].1 - 30.5).abs() < 1e-6,
            "{centers:?}"
        );
        // The second slot is NOT the near-duplicate (40.0, 30.0) — it fell
        // back to random sampling since that candidate was within
        // `min_separation` of an already-chosen center.
        assert!(
            (centers[1].0 - 40.0).abs() > 1e-6 || (centers[1].1 - 30.0).abs() > 1e-6,
            "expected the near-duplicate to be rejected, not chosen: {centers:?}"
        );
    }

    #[test]
    fn sp_anchored_patch_centers_fills_remainder_with_random_fallback() {
        // Only 1 real keypoint but `m = 4` requested — the other 3 must come
        // from the SAME random fallback the legacy sampler uses (verified by
        // checking they fall within the legacy border margin, since an exact
        // value match would require replicating the interleaved RNG state
        // exactly, which the "off = legacy" test above already covers for
        // the zero-keypoint case).
        let kps = [(40.0 * TEST_RES, 30.0 * TEST_RES, 0.9_f32)];
        let mut rng = StdRng::seed_from_u64(5);
        let centers = sp_anchored_patch_centers(4, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 4);
        assert!(
            (centers[0].0 - 40.0).abs() < 1e-6 && (centers[0].1 - 30.0).abs() < 1e-6,
            "{centers:?}"
        );
        for &(x, y) in &centers[1..] {
            assert!(
                (1.0..187.0).contains(&x) && (1.0..119.0).contains(&y),
                "fallback center out of legacy margin: ({x},{y})"
            );
        }
    }

    // ---- patch_to_world_point / nearest_patch_within / bridge ----

    #[test]
    fn patch_to_world_point_matches_hand_derivation_for_identity_pose() {
        let pose = SE3::identity();
        let intrinsics = intr();
        let patch = DpvoPatch {
            x: 64.0,
            y: 48.0,
            inverse_depth: 0.5,
        }; // depth = 2, at the principal point.
        let world = patch_to_world_point(&pose, &intrinsics, &patch).expect("positive depth");
        assert!(
            (world.coords - Vector3::new(0.0, 0.0, 2.0)).norm() < 1e-9,
            "{world:?}"
        );
    }

    #[test]
    fn patch_to_world_point_rejects_non_positive_inverse_depth() {
        let pose = SE3::identity();
        let intrinsics = intr();
        assert!(patch_to_world_point(
            &pose,
            &intrinsics,
            &DpvoPatch {
                x: 64.0,
                y: 48.0,
                inverse_depth: 0.0
            }
        )
        .is_none());
        assert!(patch_to_world_point(
            &pose,
            &intrinsics,
            &DpvoPatch {
                x: 64.0,
                y: 48.0,
                inverse_depth: -0.5
            }
        )
        .is_none());
    }

    #[test]
    fn nearest_patch_within_picks_the_closest_and_respects_radius() {
        let patches = vec![
            DpvoPatch {
                x: 10.0,
                y: 10.0,
                inverse_depth: 0.5,
            },
            DpvoPatch {
                x: 10.5,
                y: 10.5,
                inverse_depth: 0.5,
            },
            DpvoPatch {
                x: 50.0,
                y: 50.0,
                inverse_depth: 0.5,
            },
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
            old_patches.push(DpvoPatch {
                x,
                y,
                inverse_depth: 1.0 / depth,
            });
            old_keypoints.push(Point2::new(x, y));
            // Same pixel anchor (identity pose on both sides, so a physical
            // point at old-side depth `depth` reprojects to new-side pixel
            // `(x, y)` scaled radially only if the pose changed — with both
            // poses identity, the ONLY thing differing is depth itself,
            // scaled by `drift_scale`).
            new_patches.push(DpvoPatch {
                x,
                y,
                inverse_depth: 1.0 / (depth * drift_scale),
            });
            new_keypoints.push(Point2::new(x, y));
        }
        DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        }
    }

    #[test]
    fn bridge_matches_to_3d3d_recovers_expected_world_points() {
        let DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        } = synthetic_drifted_pair(10, 4.0);
        let intrinsics = intr();
        let raw_pairs: Vec<(usize, usize)> = (0..10).map(|i| (i, i)).collect();
        let bridged = bridge_matches_to_3d3d(
            &raw_pairs,
            &old_keypoints,
            &old_pose,
            &intrinsics,
            &old_patches,
            &new_keypoints,
            &new_pose,
            &intrinsics,
            &new_patches,
            1.0,
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
        let DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        } = synthetic_drifted_pair(30, 6.5);
        let intrinsics = intr();
        let raw_pairs: Vec<(usize, usize)> = (0..30).map(|i| (i, i)).collect();
        let bridged = bridge_matches_to_3d3d(
            &raw_pairs,
            &old_keypoints,
            &old_pose,
            &intrinsics,
            &old_patches,
            &new_keypoints,
            &new_pose,
            &intrinsics,
            &new_patches,
            1.0,
        );
        assert_eq!(bridged.len(), 30);
        let mut rng = StdRng::seed_from_u64(7);
        let fit = ransac_umeyama_scale(&bridged, &cfg(), &mut rng)
            .expect("well-posed synthetic fixture should fit");
        assert!(
            (fit.scale - 6.5).abs() / 6.5 < 0.05,
            "expected scale close to 6.5, got {} (inliers={} residual_ratio={})",
            fit.scale,
            fit.inlier_count,
            fit.mean_residual_ratio
        );
        assert_eq!(
            fit.inlier_count, 30,
            "a noise-free fixture should have every pair as an inlier"
        );
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
        let new_points: Vec<Point3<f64>> = old_points
            .iter()
            .map(|p| Point3::from(true_scale * (true_rotation * p.coords) + true_translation))
            .collect();
        let pairs: Vec<(Point3<f64>, Point3<f64>)> =
            old_points.into_iter().zip(new_points).collect();

        let mut rng = StdRng::seed_from_u64(42);
        let fit = ransac_umeyama_scale(&pairs, &cfg(), &mut rng)
            .expect("well-posed rigid+scale fixture should fit");
        assert!(
            (fit.scale - true_scale).abs() / true_scale < 0.02,
            "expected scale near {true_scale}, got {}",
            fit.scale
        );
        assert_eq!(fit.inlier_count, 20);
    }

    #[test]
    fn ransac_umeyama_scale_rejects_below_min_bridge_correspondences() {
        let DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        } = synthetic_drifted_pair(3, 2.0);
        let intrinsics = intr();
        let raw_pairs: Vec<(usize, usize)> = (0..3).map(|i| (i, i)).collect();
        let bridged = bridge_matches_to_3d3d(
            &raw_pairs,
            &old_keypoints,
            &old_pose,
            &intrinsics,
            &old_patches,
            &new_keypoints,
            &new_pose,
            &intrinsics,
            &new_patches,
            1.0,
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
                let source = Point3::new(
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(1.0..10.0),
                );
                let target = Point3::new(
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(-5.0..5.0),
                    rng.gen_range(1.0..10.0),
                );
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
        (0..count)
            .map(|k| synthetic_descriptor(base_seed.wrapping_add(k as u64 * 7919), dim))
            .collect()
    }

    #[test]
    fn index_stays_unbuilt_until_bootstrap_threshold_then_builds_and_backfills() {
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 5,
            vocab_words: 4,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..4 {
            index.ingest_frame(arrival, vec![], frame_descriptors(arrival as u64, 20, 16));
            assert!(
                !index.diagnostics().vocab_built,
                "should not build before the bootstrap threshold"
            );
            assert_eq!(
                index.diagnostics().frames_indexed,
                0,
                "nothing indexed until the vocabulary exists"
            );
        }
        index.ingest_frame(4, vec![], frame_descriptors(4, 20, 16));
        assert!(
            index.diagnostics().vocab_built,
            "vocabulary should build once the threshold is reached"
        );
        assert_eq!(
            index.diagnostics().frames_indexed,
            5,
            "every buffered frame should be backfilled"
        );
    }

    #[test]
    fn due_throttles_by_query_frequency() {
        let cfg = DpvoLongLoopConfig {
            query_frequency: 10,
            ..DpvoLongLoopConfig::default()
        };
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
        assert!(
            arrivals.contains(&10),
            "arrival 10 (gap=200) should be a candidate: {arrivals:?}"
        );
        assert!(
            !arrivals.contains(&190),
            "arrival 190 (gap=20 < min_temporal_gap=50) must be excluded: {arrivals:?}"
        );
        assert!(
            !arrivals.contains(&210),
            "the query frame itself must never be its own candidate"
        );
    }

    // ---- A3 ranking slice B: `RetrievalScorer::MeanPool` ----

    #[test]
    fn default_retrieval_scorer_is_vlad() {
        // Explicit pin for the "default-vlad bit-compat" contract: every
        // prior milestone's config construction (`..DpvoLongLoopConfig::default()`,
        // used throughout this test module and `examples/euroc_dpvo_vo_demo.rs`)
        // must keep reproducing the VLAD path byte-for-byte with zero
        // changes required at any of those call sites.
        assert_eq!(
            DpvoLongLoopConfig::default().retrieval_scorer,
            RetrievalScorer::Vlad
        );
    }

    #[test]
    fn mean_pool_signature_matches_hand_computed_value() {
        // Two orthonormal-basis descriptors: mean = [0.5, 0.5], and
        // normalizing [0.5, 0.5] gives [1/sqrt(2), 1/sqrt(2)] — a fully
        // hand-checkable case (the classic `normalize([1, 1])`).
        let descriptors = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let signature = mean_pool(&descriptors);
        let expected = 1.0f32 / std::f32::consts::SQRT_2;
        assert_eq!(signature.len(), 2);
        assert!(
            (signature[0] - expected).abs() < 1e-6 && (signature[1] - expected).abs() < 1e-6,
            "{signature:?}"
        );
        // Already unit-norm, by construction.
        let norm: f32 = signature.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    }

    #[test]
    fn mean_pool_signature_ignores_descriptor_order() {
        // The mean is order-independent — a basic sanity check that this is
        // really a plain arithmetic mean, not something ordering-sensitive.
        let a = mean_pool(&[vec![1.0f32, 2.0, 3.0], vec![3.0, 2.0, 1.0], vec![0.0, 0.0, 5.0]]);
        let b = mean_pool(&[vec![3.0f32, 2.0, 1.0], vec![0.0, 0.0, 5.0], vec![1.0, 2.0, 3.0]]);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_pool_of_empty_descriptors_returns_an_empty_signature() {
        // Unlike `vlad` (whose vocabulary fixes an output length ahead of
        // time), there is no vocabulary here to fall back on — see
        // `mean_pool`'s own doc for why this returns empty, not a
        // zero-length-guessed vector.
        let signature = mean_pool(&[]);
        assert!(signature.is_empty(), "{signature:?}");
    }

    #[test]
    fn mean_pool_scorer_query_candidates_ranks_by_cosine_similarity() {
        let cfg = DpvoLongLoopConfig {
            retrieval_scorer: RetrievalScorer::MeanPool,
            min_temporal_gap: 50,
            min_similarity: -1.0, // accept everything the gap gate allows, for this ranking test.
            top_k: 5,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        // Arrival 10 shares appearance "A" (base_seed=1) with the query;
        // arrival 20 is a different appearance "B" (base_seed=99).
        index.ingest_frame(10, vec![], frame_descriptors(1, 20, 16));
        index.ingest_frame(20, vec![], frame_descriptors(99, 20, 16));
        // The query frame itself, appearance "A".
        index.ingest_frame(210, vec![], frame_descriptors(1, 20, 16));

        let candidates = index.query_candidates(210);
        assert_eq!(candidates.len(), 2, "{candidates:?}");
        assert_eq!(
            candidates[0].0, 10,
            "the appearance-matching arrival must rank first: {candidates:?}"
        );
        assert_eq!(candidates[1].0, 20);
        assert!(
            candidates[0].1 > candidates[1].1,
            "cosine similarity must actually order them: {candidates:?}"
        );
    }

    #[test]
    fn mean_pool_scorer_never_trains_a_vocabulary_and_indexes_from_the_first_frame() {
        let cfg = DpvoLongLoopConfig {
            retrieval_scorer: RetrievalScorer::MeanPool,
            // The VLAD default — must be entirely irrelevant under MeanPool.
            vocab_bootstrap_frames: 40,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        // A single frame — far fewer than `vocab_bootstrap_frames` — is
        // immediately indexed and queryable, unlike VLAD's own bootstrap
        // wait (contrast
        // `index_stays_unbuilt_until_bootstrap_threshold_then_builds_and_backfills`,
        // below).
        index.ingest_frame(0, vec![], frame_descriptors(1, 20, 16));
        let diag = index.diagnostics();
        assert_eq!(
            diag.frames_indexed, 1,
            "MeanPool must index on the very first ingest call, no bootstrap wait"
        );
        assert!(!diag.vocab_built, "MeanPool must never build a vocabulary");

        // Ingest far more frames than the bootstrap threshold would have
        // required under VLAD — the vocabulary must STILL never build.
        for arrival in 1..60 {
            index.ingest_frame(arrival, vec![], frame_descriptors(arrival as u64, 20, 16));
        }
        assert!(
            !index.diagnostics().vocab_built,
            "MeanPool must never build a vocabulary, however many frames are ingested"
        );
        assert_eq!(index.diagnostics().frames_indexed, 60);
    }

    // ---- End-to-end: find_and_verify_long_range_loop ----

    /// Forward-project a world point into `pose`'s own camera frame (the
    /// exact inverse of [`patch_to_world_point`]) and re-derive the
    /// `DpvoPatch` that would have produced it — used to build fixtures
    /// where the OLD-side and NEW-side worlds are related by a KNOWN,
    /// hand-chosen Sim(3) that need not match `pose`'s own rotation (see
    /// [`find_and_verify_long_range_loop_rejects_a_rotation_inconsistent_candidate`]).
    fn patch_from_world_point(pose: &SE3, intr: &DpvoIntrinsics, world: Point3<f64>) -> DpvoPatch {
        let cam = pose.transform_point(&world);
        let x = intr.cx + intr.fx * cam.x / cam.z;
        let y = intr.cy + intr.fy * cam.y / cam.z;
        DpvoPatch {
            x,
            y,
            inverse_depth: 1.0 / cam.z,
        }
    }

    #[test]
    fn find_and_verify_long_range_loop_rejects_a_rotation_inconsistent_candidate() {
        // Milestone M12 (post-mortem on a real 800f corruption): construct a
        // candidate whose bridged 3D-3D correspondences are a perfect,
        // noise-free PURE-SCALE relationship (no rotation at all between the
        // two independently-reconstructed point sets) — RANSAC/residual
        // gates alone would accept this unconditionally (this is otherwise
        // exactly `synthetic_drifted_pair`'s own fixture shape) — but
        // `new_pose` itself carries a genuine 90-degree rotation relative to
        // `old_pose` (both identity translation), so DPVO's own trusted
        // relative rotation (`current_pose.compose(&old_pose.inverse())`)
        // disagrees with the fit's own recovered (near-identity) rotation by
        // ~90 degrees — must be rejected by `max_rotation_inconsistency_deg`
        // (default 20 degrees) even though every OTHER gate is satisfied.
        let intrinsics = intr();
        let old_pose = SE3::identity();
        let new_pose = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2),
            Vector3::zeros(),
        );
        let drift_scale = 8.0_f64;
        let n = 12;
        let mut old_patches = Vec::with_capacity(n);
        let mut old_keypoints = Vec::with_capacity(n);
        let mut new_patches = Vec::with_capacity(n);
        let mut new_keypoints = Vec::with_capacity(n);
        for i in 0..n {
            let old_world = Point3::new(
                1.0 + i as f64 * 0.3,
                ((i * 3) % 5) as f64 * 0.4 - 0.8,
                4.0 + (i as f64) * 0.5,
            );
            let new_world = Point3::from(old_world.coords * drift_scale); // pure scale, NO rotation.
            old_patches.push(patch_from_world_point(&old_pose, &intrinsics, old_world));
            new_patches.push(patch_from_world_point(&new_pose, &intrinsics, new_world));
            old_keypoints.push(Point2::new(old_patches[i].x, old_patches[i].y));
            new_keypoints.push(Point2::new(new_patches[i].x, new_patches[i].y));
        }

        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            patch_pixel_radius: 0.5,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        let old_descriptors = frame_descriptors(1, n, 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);

        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, old_patches.clone()))
            } else {
                None
            }
        };
        let result = index.find_and_verify_long_range_loop(
            300,
            &new_pose,
            &intrinsics,
            &new_patches,
            188.0,
            120.0,
            resolve_old,
        );
        assert!(
            result.is_none(),
            "a rotation-inconsistent (even if otherwise noise-free) candidate must be rejected"
        );

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(
            diagnostics.bridge_sufficient_total, 1,
            "bridging and RANSAC should both succeed on this noise-free fixture"
        );
        assert_eq!(
            diagnostics.rejected_ransac_total, 0,
            "RANSAC itself should find a fit; the NEW rotation gate is what rejects it"
        );
        assert_eq!(diagnostics.rejected_rotation_inconsistent_total, 1);

        // Milestone M12: the query log must carry the CONCRETE disagreement,
        // not just the pass/fail outcome — should be close to 90 degrees
        // (the fixture's own hand-chosen relative rotation).
        let log = index.query_log();
        let entry = log
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be logged");
        let deg = entry
            .rotation_disagreement_deg
            .expect("a candidate that reached the rotation check must log its own disagreement");
        assert!(
            (deg - 90.0).abs() < 1.0,
            "expected ~90 degrees of disagreement, got {deg}"
        );
    }

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
        let DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        } = synthetic_drifted_pair(30, 8.0);

        // Bootstrap filler frames (distinct appearance, far from both real
        // frames in arrival index and appearance) so the vocabulary builds
        // without becoming a trivial 1-frame-vs-1-frame retrieval.
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
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
            .find_and_verify_long_range_loop(
                300,
                &new_pose,
                &intrinsics,
                &new_patches,
                188.0,
                120.0,
                resolve_old,
            )
            .expect("a genuine, well-posed long-range revisit should be accepted");
        assert_eq!(accepted.arrival_i, 20);
        assert_eq!(accepted.arrival_j, 300);
        assert_eq!(accepted.measurement.arrival_i, 20);
        assert_eq!(accepted.measurement.arrival_j, 300);
        let measured_scale = accepted
            .measurement
            .measured_scale
            .expect("M11 acceptance must carry a measured scale");
        assert!(
            (measured_scale - 8.0).abs() / 8.0 < 0.05,
            "expected measured_scale near 8.0, got {measured_scale}"
        );
        // The ordinary rotation+translation edge reuses DPVO's own current
        // pose composition (both poses are identity here, so this is trivially identity too).
        assert_eq!(
            accepted.measurement.relative_pose,
            new_pose.compose(&old_pose.inverse())
        );

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 1);
        assert_eq!(diagnostics.last_accepted_gap, 280);
        // Milestone M12: the funnel step between "bridge attempted" and
        // "accepted" — a well-posed fixture should reach RANSAC and pass it.
        assert_eq!(diagnostics.bridge_sufficient_total, 1);

        // Milestone M12 (open item 2): the query log must contain this
        // candidate, marked accepted — alongside the OTHER (lower-similarity,
        // lower-ranked) filler-frame candidates the default `top_k = 3` also
        // surfaced for this query, each logged as `accepted: false`.
        let log = index.query_log();
        assert_eq!(
            log.len(),
            3,
            "top_k=3 should surface exactly 3 candidates: {log:?}"
        );
        assert!(log.iter().all(|e| e.query_arrival == 300));
        let winner = log
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be among the logged candidates");
        assert_eq!(winner.gap, 280);
        assert_eq!(
            winner.rank, 0,
            "the accepted candidate is the top-similarity one"
        );
        assert!(winner.accepted);
        assert_eq!(
            log.iter().filter(|e| e.accepted).count(),
            1,
            "at most one candidate per query is ever marked accepted"
        );
    }

    #[test]
    fn find_and_verify_long_range_loop_logs_rejected_candidates_as_not_accepted() {
        // A candidate that clears retrieval (gap + similarity) but fails
        // bridging (too few correspondences) must still appear in the query
        // log, marked `accepted: false` — the whole point of open item 2 is
        // that REJECTED candidates are logged too, not just accepted ones.
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        let intrinsics = intr();
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        // OLD frame: 3 keypoints only, matched 1:1 with the new frame below —
        // far fewer than `min_bridge_correspondences` (default 8), and no
        // patches are supplied at all, so bridging fails outright.
        let old_descriptors = frame_descriptors(1, 3, 16);
        index.ingest_frame(
            20,
            vec![
                Point2::new(10.0, 10.0),
                Point2::new(20.0, 20.0),
                Point2::new(30.0, 30.0),
            ],
            old_descriptors.clone(),
        );
        index.ingest_frame(
            300,
            vec![
                Point2::new(10.0, 10.0),
                Point2::new(20.0, 20.0),
                Point2::new(30.0, 30.0),
            ],
            old_descriptors,
        );
        let pose = SE3::identity();
        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((pose.clone(), intrinsics, vec![]))
            } else {
                None
            }
        };
        let result =
            index.find_and_verify_long_range_loop(300, &pose, &intrinsics, &[], 188.0, 120.0, resolve_old);
        assert!(
            result.is_none(),
            "no owned patches on either side => bridging must fail => no acceptance"
        );

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(diagnostics.rejected_insufficient_bridge_total, 1);
        assert_eq!(diagnostics.bridge_sufficient_total, 0);

        let log = index.query_log();
        assert!(!log.is_empty());
        let entry = log
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be among the logged candidates");
        assert!(
            !entry.accepted,
            "a bridging-rejected candidate must be logged as accepted=false"
        );
        assert!(
            log.iter().all(|e| !e.accepted),
            "no candidate was ever accepted this query"
        );
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
        let result =
            index.find_and_verify_long_range_loop(4, &pose, &intrinsics, &[], 188.0, 120.0, |_| None);
        assert!(result.is_none());
        // A3 stage-1 (`docs/visual_slam_sequential_sfm_plan.md`, "densify
        // query cadence" slice): this IS the zero-candidate case — the
        // query was issued (`due()` was never even consulted here, but the
        // call itself counts, per `queries_issued_total`'s own doc) and
        // `query_candidates` returned nothing (the gap gate rejects
        // everything), so both new counters must fire exactly once and the
        // arrival must be recorded in `empty_query_arrivals`.
        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(diagnostics.queries_issued_total, 1);
        assert_eq!(diagnostics.queries_with_zero_candidates, 1);
        assert_eq!(index.empty_query_arrivals(), &[4]);
    }

    #[test]
    fn queries_issued_total_and_zero_candidates_counter_default_config_unchanged() {
        // A3 stage-1: pins that the two NEW counters do not perturb the
        // existing, already-pinned acceptance path — a query that finds and
        // accepts a genuine candidate must count as issued (1) and NOT as
        // zero-candidate (0), and `empty_query_arrivals` must stay empty,
        // under `DpvoLongLoopConfig::default()`'s own `query_frequency`
        // (only `min_temporal_gap`/`min_similarity`/bootstrap knobs are
        // overridden here, exactly as the pre-existing acceptance test
        // already does, to keep the fixture small).
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            patch_pixel_radius: 0.5,
            ..DpvoLongLoopConfig::default()
        };
        assert_eq!(
            cfg.query_frequency,
            DpvoLongLoopConfig::default().query_frequency,
            "this test intentionally leaves query_frequency at its committed default (40)"
        );
        let mut index = DpvoLongLoopIndex::new(cfg);
        let intrinsics = intr();
        let DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        } = synthetic_drifted_pair(30, 8.0);
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        let old_descriptors = frame_descriptors(1, 30, 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);
        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, old_patches.clone()))
            } else {
                None
            }
        };
        let accepted = index
            .find_and_verify_long_range_loop(
                300,
                &new_pose,
                &intrinsics,
                &new_patches,
                188.0,
                120.0,
                resolve_old,
            )
            .expect("a genuine, well-posed long-range revisit should be accepted");
        assert_eq!(accepted.arrival_i, 20);

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 1);
        assert_eq!(
            diagnostics.queries_issued_total, 1,
            "one call to find_and_verify_long_range_loop => one issued query"
        );
        assert_eq!(
            diagnostics.queries_with_zero_candidates, 0,
            "candidates were found (and one accepted), so this must NOT count as zero-candidate"
        );
        assert!(
            index.empty_query_arrivals().is_empty(),
            "a non-empty query must never appear in empty_query_arrivals"
        );
    }

    #[test]
    fn query_frequency_one_makes_every_eligible_arrival_issue_a_query() {
        // A3 stage-1's own densification slice: `--ll-query-frequency 1`
        // (vs. the committed default `40`) must make `due()` fire on EVERY
        // committed arrival, and each firing must correspond to exactly one
        // `find_and_verify_long_range_loop` call counted in
        // `queries_issued_total` — `min_temporal_gap` is set unreachably
        // high so every one of these queries lands in the zero-candidate
        // bucket, isolating the CADENCE claim from candidate-outcome noise
        // (the sibling test above already covers a query that DOES find
        // candidates).
        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 1000, // unreachable within this fixture's arrival range.
            query_frequency: 1,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        let intrinsics = intr();
        let pose = SE3::identity();
        for arrival in 0..3 {
            index.ingest_frame(arrival, vec![], frame_descriptors(arrival as u64, 20, 16));
        }
        let mut issued = 0usize;
        let mut expected_empty_arrivals = Vec::new();
        for arrival in 3..13 {
            index.ingest_frame(arrival, vec![], frame_descriptors(arrival as u64, 20, 16));
            assert!(
                index.due(arrival),
                "query_frequency=1 must make every arrival due, arrival={arrival}"
            );
            let result =
                index.find_and_verify_long_range_loop(
                    arrival,
                    &pose,
                    &intrinsics,
                    &[],
                    188.0,
                    120.0,
                    |_| None,
                );
            assert!(
                result.is_none(),
                "min_temporal_gap=1000 => no candidate ever clears the gap"
            );
            issued += 1;
            expected_empty_arrivals.push(arrival);
        }
        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.queries_issued_total, issued);
        assert_eq!(diagnostics.queries_with_zero_candidates, issued);
        assert_eq!(index.empty_query_arrivals(), &expected_empty_arrivals[..]);
    }

    // ---- A3 stage 2, first slice: 2D-2D-first loop geometry ----

    /// A3 stage-2 test helper: build a genuinely RIGID 2D-2D correspondence
    /// set (`old_keypoints[i]`/`new_keypoints[i]` are TRUE projections of
    /// the SAME world point `world_points[i]` through `old_pose`/`new_pose`
    /// — so essential-matrix RANSAC recovers `new_pose`'s own actual
    /// relative rotation, up to noise-free numerical precision) PLUS a
    /// bridgeable, INDEPENDENTLY-scaled 3D-3D patch pair:
    /// `old_patches[i]` decodes to `world_points[i]` itself (a "real",
    /// trustworthy old-side reconstruction), while `new_patches[i]` decodes
    /// to `new_pose`'s own optical center plus `alpha` times the vector from
    /// that center to `world_points[i]` — i.e. a PURE, ROTATION-FREE scale
    /// of `world_points[i]` about `new_pose`'s own camera center. This is
    /// GEOMETRICALLY EXACT, not an approximation: scaling a point along its
    /// own camera ray never changes which pixel it projects to, so
    /// `new_patches[i]`'s own pixel is EXACTLY `new_keypoints[i]` — no
    /// `patch_pixel_radius` fudging required. The resulting Umeyama fit
    /// relating `world_points` to the decoded new-side points is therefore
    /// EXACTLY `scale=alpha, rotation=IDENTITY` (zero residual),
    /// INDEPENDENT of `new_pose`'s own rotation — letting the tests below
    /// choose `new_pose`'s rotation to control whether that IDENTITY
    /// Umeyama rotation agrees or disagrees with the (real, accurate)
    /// E-matrix rotation.
    #[allow(clippy::type_complexity)]
    fn rigid_2d2d_with_pure_scale_bridge(
        old_pose: &SE3,
        new_pose: &SE3,
        intr: &DpvoIntrinsics,
        world_points: &[Point3<f64>],
        alpha: f64,
    ) -> (Vec<Point2<f64>>, Vec<DpvoPatch>, Vec<Point2<f64>>, Vec<DpvoPatch>) {
        let c_new = new_pose.inverse().transform_point(&Point3::origin());
        let mut old_keypoints = Vec::with_capacity(world_points.len());
        let mut old_patches = Vec::with_capacity(world_points.len());
        let mut new_keypoints = Vec::with_capacity(world_points.len());
        let mut new_patches = Vec::with_capacity(world_points.len());
        for &world in world_points {
            let op = patch_from_world_point(old_pose, intr, world);
            old_keypoints.push(Point2::new(op.x, op.y));
            old_patches.push(op);

            let decoded_new_world =
                Point3::from(c_new.coords + alpha * (world.coords - c_new.coords));
            let np = patch_from_world_point(new_pose, intr, decoded_new_world);
            new_keypoints.push(Point2::new(np.x, np.y));
            new_patches.push(np);
        }
        (old_keypoints, old_patches, new_keypoints, new_patches)
    }

    /// A spread of 15 world points in front of both `old_pose` (identity)
    /// and any `new_pose` with a modest rotation/translation — wide enough
    /// in `(x, y)` to clear `stage2_min_coverage_fraction` comfortably at
    /// the `188x120` test grid used throughout this module's fixtures.
    fn spread_world_points(n: usize) -> Vec<Point3<f64>> {
        // Explicit, independently-varied (x, y, z) spread — mirrors
        // `crates/vision/src/two_view/mod.rs`'s own `synthetic_world_points`
        // fixture shape (already proven to work with `EssentialRansac`
        // there), scaled to this module's own smaller `intr()` test
        // intrinsics/depth range. A low-discrepancy formula (golden-ratio
        // jitter) was tried first here and produced an occasionally
        // ill-conditioned 8-point sample for large (~60 degree) synthetic
        // rotations; this explicit list is deliberately non-collinear,
        // non-coplanar, and reused (not re-derived) across every stage-2
        // test in this module for that reason.
        let base = [
            (-1.5, -1.0, 5.0),
            (1.5, -1.0, 5.2),
            (-1.5, 1.0, 5.4),
            (1.5, 1.0, 5.6),
            (0.0, 0.0, 5.8),
            (0.7, -0.3, 6.0),
            (-0.9, 0.5, 5.1),
            (0.8, 0.9, 6.3),
            (-0.4, -0.7, 5.5),
            (0.3, 0.2, 6.6),
            (-1.2, 0.2, 5.9),
            (1.1, -0.8, 6.2),
            (-0.6, 1.0, 5.3),
            (0.9, -1.0, 5.7),
            (-1.0, -0.4, 6.4),
            (0.2, 0.8, 5.2),
            (1.3, 0.4, 6.1),
            (-0.3, -1.0, 5.6),
            (0.5, 0.6, 6.5),
            (-1.4, -0.6, 5.0),
            (1.0, 1.0, 6.7),
            (-0.8, -0.9, 5.8),
            (0.4, -0.5, 6.9),
            (-0.1, 0.9, 5.4),
            (1.4, -0.2, 6.0),
        ];
        base.into_iter()
            .cycle()
            .take(n.max(1))
            .map(|(x, y, z)| Point3::new(x, y, z))
            .collect()
    }

    #[test]
    fn find_and_verify_long_range_loop_stage2_default_off_matches_existing_behavior() {
        // Default-off contract: `stage2_2d2d_geometry` defaults to `false`,
        // and running the EXISTING (pre-A3-stage-2) acceptance fixture with
        // it explicitly left off must reproduce the exact M11/M12 outcome —
        // accepted, with every new stage-2 diagnostic/log field at its
        // "never ran" zero/`None` value.
        assert!(!DpvoLongLoopConfig::default().stage2_2d2d_geometry);

        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            patch_pixel_radius: 0.5,
            stage2_2d2d_geometry: false,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        let intrinsics = intr();
        let DriftedPairFixture {
            old_pose,
            old_patches,
            old_keypoints,
            new_pose,
            new_patches,
            new_keypoints,
        } = synthetic_drifted_pair(30, 8.0);
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        let old_descriptors = frame_descriptors(1, 30, 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);
        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, old_patches.clone()))
            } else {
                None
            }
        };

        let accepted = index
            .find_and_verify_long_range_loop(
                300,
                &new_pose,
                &intrinsics,
                &new_patches,
                188.0,
                120.0,
                resolve_old,
            )
            .expect("stage2 off must reproduce the existing M11/M12 acceptance");
        assert_eq!(accepted.arrival_i, 20);
        let measured_scale = accepted.measurement.measured_scale.expect("scale carried");
        assert!((measured_scale - 8.0).abs() / 8.0 < 0.05);

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 1);
        assert!(!diagnostics.stage2_enabled);
        assert_eq!(diagnostics.stage2_attempts_total, 0);
        assert_eq!(diagnostics.stage2_passed_total, 0);
        assert_eq!(diagnostics.stage2_rejected_insufficient_matches_total, 0);
        assert_eq!(diagnostics.stage2_rejected_insufficient_inliers_total, 0);
        assert_eq!(diagnostics.stage2_rejected_insufficient_coverage_total, 0);
        assert_eq!(diagnostics.stage2_rejected_rotation_inconsistent_total, 0);
        assert_eq!(diagnostics.stage2_rejected_high_residual_total, 0);
        assert_eq!(diagnostics.stage2_rejected_umeyama_vs_e_rotation_total, 0);

        let winner = index
            .query_log()
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be logged");
        assert!(winner.accepted);
        assert!(winner.final_accepted);
        assert_eq!(winner.stage_reached, "accepted");
        assert!(winner.stage2_2d2d_inliers.is_none());
        assert!(winner.stage2_e_rotation_disagreement_deg.is_none());
    }

    #[test]
    fn find_and_verify_long_range_loop_stage2_accepts_a_correct_loop_through_all_gates() {
        // A synthetic two-frame geometry where a CORRECT loop passes EVERY
        // stage-2 gate (a)-(c) AND the existing 3D-3D bridge/RANSAC/rotation
        // gates AND the new step-(d) Umeyama-vs-E rotation gate: `new_pose`
        // carries ONLY a translation (zero rotation), so the trusted pose,
        // the 2D-2D essential-matrix fit, and the 3D-3D Umeyama fit (which
        // is exactly IDENTITY rotation by `rigid_2d2d_with_pure_scale_bridge`'s
        // own construction) all agree at ~0 degrees.
        let intrinsics = intr();
        let old_pose = SE3::identity();
        let new_pose = SE3::new(UnitQuaternion::identity(), Vector3::new(0.6, 0.0, 0.2));
        let world_points = spread_world_points(15);
        let alpha = 2.0_f64;
        let (old_keypoints, old_patches, new_keypoints, new_patches) =
            rigid_2d2d_with_pure_scale_bridge(&old_pose, &new_pose, &intrinsics, &world_points, alpha);

        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            patch_pixel_radius: 0.5,
            stage2_2d2d_geometry: true,
            // Kept small so a 15-point fixture can clear it while still
            // exercising the real inlier-count/coverage gate CODE PATHS —
            // not the shipped production defaults (30 / 0.25), mirroring
            // this module's own established style of overriding non-
            // load-bearing thresholds to keep fixtures small (e.g.
            // `patch_pixel_radius` above).
            stage2_min_inliers: 10,
            stage2_min_coverage_fraction: 0.05,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        let old_descriptors = frame_descriptors(1, world_points.len(), 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);

        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, old_patches.clone()))
            } else {
                None
            }
        };
        let accepted = index
            .find_and_verify_long_range_loop(
                300,
                &new_pose,
                &intrinsics,
                &new_patches,
                188.0,
                120.0,
                resolve_old,
            )
            .expect("a correct loop should pass every stage-2 AND existing gate");
        assert_eq!(accepted.arrival_i, 20);
        let measured_scale = accepted.measurement.measured_scale.expect("scale carried");
        assert!(
            (measured_scale - alpha).abs() / alpha < 0.05,
            "expected measured_scale near {alpha}, got {measured_scale}"
        );

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 1);
        assert!(diagnostics.stage2_enabled);
        assert_eq!(diagnostics.stage2_attempts_total, 1);
        assert_eq!(diagnostics.stage2_passed_total, 1);
        assert_eq!(diagnostics.stage2_rejected_insufficient_matches_total, 0);
        assert_eq!(diagnostics.stage2_rejected_insufficient_inliers_total, 0);
        assert_eq!(diagnostics.stage2_rejected_insufficient_coverage_total, 0);
        assert_eq!(diagnostics.stage2_rejected_rotation_inconsistent_total, 0);
        assert_eq!(diagnostics.stage2_rejected_high_residual_total, 0);
        assert_eq!(diagnostics.stage2_rejected_umeyama_vs_e_rotation_total, 0);
        assert_eq!(diagnostics.bridge_sufficient_total, 1);
        assert_eq!(diagnostics.rejected_rotation_inconsistent_total, 0);

        let winner = index
            .query_log()
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be logged");
        assert!(winner.final_accepted);
        assert_eq!(winner.stage_reached, "accepted");
        let stage2_inliers = winner
            .stage2_2d2d_inliers
            .expect("a stage-2-passing candidate must log its own inlier count");
        assert!(
            stage2_inliers >= 10,
            "expected >= stage2_min_inliers, got {stage2_inliers}"
        );
        let e_rot_deg = winner
            .stage2_e_rotation_disagreement_deg
            .expect("a stage-2-passing candidate must log its own E-vs-trusted disagreement");
        assert!(
            e_rot_deg < 5.0,
            "expected near-zero E-vs-trusted disagreement (zero true rotation), got {e_rot_deg}"
        );
    }

    #[test]
    fn find_and_verify_long_range_loop_stage2_rejects_rotation_inconsistent_2d2d_candidate() {
        // A3 stage-2 gate (c): the 2D-2D correspondences are generated from
        // a GENUINE ~15-degree relative rotation (`gen_new_pose` — the SAME
        // rotation/translation magnitude the sibling "accepts"/"dies at (d)"
        // tests below already prove `EssentialRansac` recovers reliably;
        // much larger synthetic rotations were tried first here and made
        // the noise-free 8-point DLT solve numerically unstable, an
        // unrelated pre-existing property of the reused library code this
        // task does not touch), but the pose actually passed to the
        // function as "current" (hence DPVO's own trusted relative
        // rotation) carries a DIFFERENT, much larger rotation — a ~35-degree
        // disagreement, past `max_rotation_inconsistency_deg`'s default
        // `20.0`. Must be rejected at stage 2 BEFORE the existing 3D-3D
        // bridge ever runs (patches are intentionally empty/unused).
        let intrinsics = intr();
        let old_pose = SE3::identity();
        let gen_new_pose = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 15f64.to_radians()),
            Vector3::new(0.5, 0.0, 0.1),
        );
        // Deliberately WRONG vs `gen_new_pose` (a ~35-degree rotation gap).
        let trusted_new_pose = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 50f64.to_radians()),
            Vector3::new(0.5, 0.0, 0.1),
        );
        let world_points = spread_world_points(15);

        let mut old_keypoints = Vec::with_capacity(world_points.len());
        let mut new_keypoints = Vec::with_capacity(world_points.len());
        for &world in &world_points {
            let op = patch_from_world_point(&old_pose, &intrinsics, world);
            old_keypoints.push(Point2::new(op.x, op.y));
            let np = patch_from_world_point(&gen_new_pose, &intrinsics, world);
            new_keypoints.push(Point2::new(np.x, np.y));
        }

        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            stage2_2d2d_geometry: true,
            stage2_min_inliers: 10,
            stage2_min_coverage_fraction: 0.05,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        let old_descriptors = frame_descriptors(1, world_points.len(), 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);

        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, vec![]))
            } else {
                None
            }
        };
        let result = index.find_and_verify_long_range_loop(
            300,
            &trusted_new_pose,
            &intrinsics,
            &[],
            188.0,
            120.0,
            resolve_old,
        );
        assert!(
            result.is_none(),
            "a 2D-2D-vs-trusted rotation-inconsistent candidate must be rejected at stage 2"
        );

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(diagnostics.stage2_attempts_total, 1);
        assert_eq!(diagnostics.stage2_passed_total, 0);
        assert_eq!(diagnostics.stage2_rejected_rotation_inconsistent_total, 1);
        assert_eq!(
            diagnostics.verification_attempts, 0,
            "must be rejected BEFORE the existing 3D-3D bridge is ever attempted"
        );
        assert_eq!(diagnostics.bridge_sufficient_total, 0);

        let entry = index
            .query_log()
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be logged");
        assert_eq!(entry.stage_reached, "2d2d_rotation_inconsistent");
        assert!(!entry.final_accepted);
        let e_rot_deg = entry
            .stage2_e_rotation_disagreement_deg
            .expect("a candidate that reached the rotation check must log its disagreement");
        assert!(
            e_rot_deg > 25.0,
            "expected a large (~35 degree) disagreement, got {e_rot_deg}"
        );
    }

    #[test]
    fn find_and_verify_long_range_loop_stage2_dies_at_umeyama_vs_e_rotation_gate() {
        // A3 stage-2 step (d): construct a candidate that passes EVERY prior
        // gate — stage-2 (a)-(c) (the 2D-2D essential-matrix fit's own
        // rotation agrees with the TRUSTED pose, since both come from the
        // SAME genuine ~15-degree `new_pose` rotation) AND the existing
        // 3D-3D bridge/RANSAC gates AND the existing rotation-vs-trusted-pose
        // gate (the bridge's own Umeyama fit is EXACTLY IDENTITY rotation by
        // `rigid_2d2d_with_pure_scale_bridge`'s own construction, and
        // `0 - 15 = 15 degrees < 20.0` still clears
        // `max_rotation_inconsistency_deg`) — but FAILS the NEW, tighter
        // step-(d) gate: Umeyama (IDENTITY, ~0 degrees) disagrees with the
        // 2D-2D E-matrix fit (~15 degrees, matching the genuine rotation)
        // by ~15 degrees, which exceeds `stage2_umeyama_vs_e_rotation_max_deg`'s
        // default `10.0`. This is precisely the "bridge and trusted pose are
        // both wrong in a mutually consistent way" failure mode this gate
        // exists to catch — a scale-corrupt 3D-3D correspondence set that
        // nonetheless agrees with the (also-drifting) trusted pose closely
        // enough to fool the EXISTING M12 gate alone.
        let intrinsics = intr();
        let old_pose = SE3::identity();
        let new_pose = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 15f64.to_radians()),
            Vector3::new(0.5, 0.0, 0.1),
        );
        let world_points = spread_world_points(15);
        let alpha = 2.0_f64; // pure-scale "drift" — see the helper's own doc.
        let (old_keypoints, old_patches, new_keypoints, new_patches) =
            rigid_2d2d_with_pure_scale_bridge(&old_pose, &new_pose, &intrinsics, &world_points, alpha);

        let cfg = DpvoLongLoopConfig {
            vocab_bootstrap_frames: 3,
            vocab_words: 4,
            min_temporal_gap: 50,
            min_similarity: -1.0,
            patch_pixel_radius: 0.5,
            stage2_2d2d_geometry: true,
            stage2_min_inliers: 10,
            stage2_min_coverage_fraction: 0.05,
            ..DpvoLongLoopConfig::default()
        };
        let mut index = DpvoLongLoopIndex::new(cfg);
        for arrival in 0..3 {
            index.ingest_frame(
                arrival,
                vec![],
                frame_descriptors(1000 + arrival as u64, 20, 16),
            );
        }
        let old_descriptors = frame_descriptors(1, world_points.len(), 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);

        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 {
                Some((old_pose.clone(), intrinsics, old_patches.clone()))
            } else {
                None
            }
        };
        let result = index.find_and_verify_long_range_loop(
            300,
            &new_pose,
            &intrinsics,
            &new_patches,
            188.0,
            120.0,
            resolve_old,
        );
        assert!(
            result.is_none(),
            "a scale-corrupt 3D-3D set (Umeyama rotation disagreeing with the independent \
             2D-2D E-matrix rotation) must be rejected even though it agrees with the \
             ALSO-drifting trusted pose closely enough to pass the existing M12 gate"
        );

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(
            diagnostics.stage2_passed_total, 1,
            "stage 2 (a)-(c) must all pass — the E fit itself is accurate"
        );
        assert_eq!(
            diagnostics.bridge_sufficient_total, 1,
            "the 3D-3D bridge must succeed (patches sit exactly at the matched keypoints)"
        );
        assert_eq!(
            diagnostics.rejected_ransac_total, 0,
            "the pure-scale correspondence set is a perfect, zero-residual fit"
        );
        assert_eq!(
            diagnostics.rejected_rotation_inconsistent_total, 0,
            "the EXISTING gate (vs the also-~15-degree trusted pose) must still pass"
        );
        assert_eq!(
            diagnostics.stage2_rejected_umeyama_vs_e_rotation_total, 1,
            "the NEW step-(d) gate (vs the independent E rotation) must be what rejects it"
        );

        let entry = index
            .query_log()
            .iter()
            .find(|e| e.candidate_arrival == 20)
            .expect("arrival 20 must be logged");
        assert_eq!(entry.stage_reached, "umeyama_vs_e_rotation_inconsistent");
        assert!(!entry.final_accepted);
        let e_rot_vs_trusted = entry
            .stage2_e_rotation_disagreement_deg
            .expect("a stage-2-passing candidate must log its own E-vs-trusted disagreement");
        assert!(
            e_rot_vs_trusted < 5.0,
            "the E fit itself should closely match the genuine ~15-degree trusted rotation, \
             got {e_rot_vs_trusted}"
        );
        let umeyama_vs_trusted = entry
            .rotation_disagreement_deg
            .expect("a bridge+RANSAC-passing candidate must log its own Umeyama-vs-trusted disagreement");
        assert!(
            (5.0..20.0).contains(&umeyama_vs_trusted),
            "expected Umeyama (~0 deg) vs trusted (~15 deg) disagreement inside the OLD gate's \
             own tolerance, got {umeyama_vs_trusted}"
        );
    }
}
