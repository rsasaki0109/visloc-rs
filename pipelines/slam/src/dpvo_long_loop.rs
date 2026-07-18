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

use std::collections::VecDeque;
use std::time::Instant;

use nalgebra::{Point2, Point3, Rotation3, UnitQuaternion, Vector3};
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
            sp_anchored_patches: false,
            sp_patch_min_separation: 2.0,
            max_rotation_inconsistency_deg: 20.0,
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
    /// Milestone M12: every top-`K` candidate ever surfaced by a query, not
    /// just accepted ones — see [`QueryCandidateLogEntry`]'s own doc.
    /// Unbounded (like `dpvo_sim3_backend`'s own retained-pose history, this
    /// index's own regime is hundreds to low thousands of frames per run,
    /// not an unbounded streaming service), so a very long run's own memory
    /// cost here is `queries_attempted * top_k` entries — reported, not
    /// silently bounded, honestly documented rather than pretending it is
    /// free.
    query_log: Vec<QueryCandidateLogEntry>,

    diag_queries_attempted: usize,
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
            diag_queries_attempted: 0,
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
        }
    }

    /// Milestone M12 (open item 2 carried forward from M11): every top-`K`
    /// retrieval candidate ever surfaced by any query, across the whole
    /// run — see [`QueryCandidateLogEntry`]'s own doc for exactly what
    /// `accepted: false` does and does not mean.
    pub fn query_log(&self) -> &[QueryCandidateLogEntry] {
        &self.query_log
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
        let mut winning_arrival: Option<usize> = None;
        // Milestone M12: every candidate that reached the rotation-consistency
        // check, and its own measured disagreement in degrees — logged
        // unconditionally (pass or fail) so `QueryCandidateLogEntry::rotation_disagreement_deg`
        // can report the CONCRETE number, not just the pass/fail outcome.
        let mut rotation_checks: Vec<(usize, f64)> = Vec::new();
        for &(old_arrival, similarity) in &candidates {
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
            self.diag_bridge_sufficient += 1;

            let Some(fit) = ransac_umeyama_scale(&bridged, &self.config, &mut self.rng) else {
                self.diag_rejected_ransac += 1;
                continue;
            };

            let relative_pose = current_pose.compose(&old_pose.inverse());
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
            let rotation_disagreement_deg = fit_rotation_uq.angle_to(&relative_pose.rotation).to_degrees();
            rotation_checks.push((old_arrival, rotation_disagreement_deg));
            if rotation_disagreement_deg > self.config.max_rotation_inconsistency_deg {
                self.diag_rejected_rotation_inconsistent += 1;
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
            accepted = Some(AcceptedLongLoop { arrival_i: old_arrival, arrival_j: current_arrival, measurement });
            winning_arrival = Some(old_arrival);
            break;
        }

        // Milestone M12 (open item 2 carried forward from M11): log EVERY
        // top-`K` candidate this query surfaced, not just the accepted one —
        // see `QueryCandidateLogEntry`'s own doc for what `accepted: false`
        // does and does not mean (a candidate ranked below the winner is
        // never even attempted, per this function's own "stop at first
        // accepted" design, and is still logged as `accepted: false`).
        for (rank, &(candidate_arrival, similarity)) in candidates.iter().enumerate() {
            let rotation_disagreement_deg =
                rotation_checks.iter().find(|&&(arrival, _)| arrival == candidate_arrival).map(|&(_, deg)| deg);
            self.query_log.push(QueryCandidateLogEntry {
                query_arrival: current_arrival,
                candidate_arrival,
                gap: current_arrival.saturating_sub(candidate_arrival),
                similarity,
                rank,
                accepted: winning_arrival == Some(candidate_arrival),
                rotation_disagreement_deg,
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

    Some(GeometricFit { scale: refit.scale, rotation: refit.rotation, inlier_count: inlier_pairs.len(), mean_residual_ratio })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intr() -> DpvoIntrinsics {
        DpvoIntrinsics { fx: 200.0, fy: 200.0, cx: 64.0, cy: 48.0 }
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
        let kps = [(40.0 * TEST_RES, 30.0 * TEST_RES, 0.9_f32), (50.0 * TEST_RES, 35.0 * TEST_RES, 0.8_f32)];
        let mut rng = StdRng::seed_from_u64(1);
        let centers = sp_anchored_patch_centers(2, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 2);
        assert!((centers[0].0 - 40.0).abs() < 1e-6 && (centers[0].1 - 30.0).abs() < 1e-6, "{centers:?}");
        assert!((centers[1].0 - 50.0).abs() < 1e-6 && (centers[1].1 - 35.0).abs() < 1e-6, "{centers:?}");
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
        assert!((centers[0].0 - 1.0).abs() < 1e-6 && (centers[0].1 - 1.0).abs() < 1e-6, "{centers:?}");
    }

    #[test]
    fn sp_anchored_patch_centers_deduplicates_within_min_separation() {
        // Two keypoints mapping to (almost) the same patch-grid cell — only
        // the higher-scored one should be kept; the second slot falls back
        // to random sampling rather than a near-duplicate center.
        let kps = [(40.0 * TEST_RES, 30.0 * TEST_RES, 0.9_f32), (40.5 * TEST_RES, 30.5 * TEST_RES, 0.95_f32)];
        let mut rng = StdRng::seed_from_u64(4);
        let centers = sp_anchored_patch_centers(2, 188, 120, &kps, TEST_RES, 2.0, &mut rng);
        assert_eq!(centers.len(), 2);
        // The higher-scored keypoint (40.5, 30.5) wins the first slot.
        assert!((centers[0].0 - 40.5).abs() < 1e-6 && (centers[0].1 - 30.5).abs() < 1e-6, "{centers:?}");
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
        assert!((centers[0].0 - 40.0).abs() < 1e-6 && (centers[0].1 - 30.0).abs() < 1e-6, "{centers:?}");
        for &(x, y) in &centers[1..] {
            assert!((1.0..187.0).contains(&x) && (1.0..119.0).contains(&y), "fallback center out of legacy margin: ({x},{y})");
        }
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
        DpvoPatch { x, y, inverse_depth: 1.0 / cam.z }
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
        let new_pose = SE3::new(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2), Vector3::zeros());
        let drift_scale = 8.0_f64;
        let n = 12;
        let mut old_patches = Vec::with_capacity(n);
        let mut old_keypoints = Vec::with_capacity(n);
        let mut new_patches = Vec::with_capacity(n);
        let mut new_keypoints = Vec::with_capacity(n);
        for i in 0..n {
            let old_world = Point3::new(1.0 + i as f64 * 0.3, ((i * 3) % 5) as f64 * 0.4 - 0.8, 4.0 + (i as f64) * 0.5);
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
            index.ingest_frame(arrival, vec![], frame_descriptors(1000 + arrival as u64, 20, 16));
        }
        let old_descriptors = frame_descriptors(1, n, 16);
        index.ingest_frame(20, old_keypoints, old_descriptors.clone());
        index.ingest_frame(300, new_keypoints, old_descriptors);

        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 { Some((old_pose.clone(), intrinsics, old_patches.clone())) } else { None }
        };
        let result = index.find_and_verify_long_range_loop(300, &new_pose, &intrinsics, &new_patches, resolve_old);
        assert!(result.is_none(), "a rotation-inconsistent (even if otherwise noise-free) candidate must be rejected");

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(diagnostics.bridge_sufficient_total, 1, "bridging and RANSAC should both succeed on this noise-free fixture");
        assert_eq!(diagnostics.rejected_ransac_total, 0, "RANSAC itself should find a fit; the NEW rotation gate is what rejects it");
        assert_eq!(diagnostics.rejected_rotation_inconsistent_total, 1);

        // Milestone M12: the query log must carry the CONCRETE disagreement,
        // not just the pass/fail outcome — should be close to 90 degrees
        // (the fixture's own hand-chosen relative rotation).
        let log = index.query_log();
        let entry = log.iter().find(|e| e.candidate_arrival == 20).expect("arrival 20 must be logged");
        let deg = entry.rotation_disagreement_deg.expect("a candidate that reached the rotation check must log its own disagreement");
        assert!((deg - 90.0).abs() < 1.0, "expected ~90 degrees of disagreement, got {deg}");
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
        // Milestone M12: the funnel step between "bridge attempted" and
        // "accepted" — a well-posed fixture should reach RANSAC and pass it.
        assert_eq!(diagnostics.bridge_sufficient_total, 1);

        // Milestone M12 (open item 2): the query log must contain this
        // candidate, marked accepted — alongside the OTHER (lower-similarity,
        // lower-ranked) filler-frame candidates the default `top_k = 3` also
        // surfaced for this query, each logged as `accepted: false`.
        let log = index.query_log();
        assert_eq!(log.len(), 3, "top_k=3 should surface exactly 3 candidates: {log:?}");
        assert!(log.iter().all(|e| e.query_arrival == 300));
        let winner = log.iter().find(|e| e.candidate_arrival == 20).expect("arrival 20 must be among the logged candidates");
        assert_eq!(winner.gap, 280);
        assert_eq!(winner.rank, 0, "the accepted candidate is the top-similarity one");
        assert!(winner.accepted);
        assert_eq!(log.iter().filter(|e| e.accepted).count(), 1, "at most one candidate per query is ever marked accepted");
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
            index.ingest_frame(arrival, vec![], frame_descriptors(1000 + arrival as u64, 20, 16));
        }
        // OLD frame: 3 keypoints only, matched 1:1 with the new frame below —
        // far fewer than `min_bridge_correspondences` (default 8), and no
        // patches are supplied at all, so bridging fails outright.
        let old_descriptors = frame_descriptors(1, 3, 16);
        index.ingest_frame(20, vec![Point2::new(10.0, 10.0), Point2::new(20.0, 20.0), Point2::new(30.0, 30.0)], old_descriptors.clone());
        index.ingest_frame(
            300,
            vec![Point2::new(10.0, 10.0), Point2::new(20.0, 20.0), Point2::new(30.0, 30.0)],
            old_descriptors,
        );
        let pose = SE3::identity();
        let resolve_old = |arrival: usize| -> Option<(SE3, DpvoIntrinsics, Vec<DpvoPatch>)> {
            if arrival == 20 { Some((pose.clone(), intrinsics, vec![])) } else { None }
        };
        let result = index.find_and_verify_long_range_loop(300, &pose, &intrinsics, &[], resolve_old);
        assert!(result.is_none(), "no owned patches on either side => bridging must fail => no acceptance");

        let diagnostics = index.diagnostics();
        assert_eq!(diagnostics.accepted_total, 0);
        assert_eq!(diagnostics.rejected_insufficient_bridge_total, 1);
        assert_eq!(diagnostics.bridge_sufficient_total, 0);

        let log = index.query_log();
        assert!(!log.is_empty());
        let entry = log.iter().find(|e| e.candidate_arrival == 20).expect("arrival 20 must be among the logged candidates");
        assert!(!entry.accepted, "a bridging-rejected candidate must be logged as accepted=false");
        assert!(log.iter().all(|e| !e.accepted), "no candidate was ever accepted this query");
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
