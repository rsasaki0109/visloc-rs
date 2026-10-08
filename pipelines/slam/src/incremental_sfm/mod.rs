//! Incremental structure-from-motion from an **unordered** image set.
//!
//! The stereo-VO SfM path ([`crate::stereo_vo_ba`]) assumes an *ordered* video
//! stream: temporal frame→frame matches give forward feature tracks, and stereo
//! gives metric scale for free. That is the wrong shape for a photo collection,
//! where the images have no temporal order, no known overlap graph, and (in the
//! monocular case) no metric scale. This module is the COLMAP-style answer: it
//! takes per-image features plus a set of **geometrically verified pairwise
//! matches** (any source — VLAD-retrieved candidate pairs filtered by an
//! essential-matrix RANSAC) and grows one consistent reconstruction.
//!
//! Pipeline:
//! 1. **Tracks.** Union-find over every `(image, keypoint)` node joined by a
//!    pairwise match. Each connected component is a feature track — one 3D
//!    point seen by many images. Tracks with two keypoints in the *same* image
//!    are inconsistent and dropped.
//! 2. **Seed.** Candidate pairs (most matches first, enough parallax) bootstrap
//!    the reconstruction via two-view relative pose ([`visloc_vision::two_view`]);
//!    the candidate that grows the most images is kept, so a repetitive scene
//!    whose strongest pair is an isolated cluster of adjacent frames is not
//!    trapped. This fixes the gauge (seed image at the origin) and the arbitrary
//!    monocular scale.
//! 3. **Grow.** Repeatedly register the unregistered image that observes the
//!    most already-triangulated tracks, by PnP RANSAC
//!    ([`visloc_vision::ransac`]); then triangulate every track that two
//!    registered views now share with sufficient parallax.
//! 4. **Bundle-adjust.** Periodically and at the end, refine all registered
//!    poses and triangulated points jointly with the Schur-complement BA
//!    ([`crate::bundle`]). Monocular has a 7-DoF gauge (6 rigid + scale), so two
//!    poses are fixed — the anchor and the longest-baseline pose — to pin scale
//!    as well as the frame.
//! 5. **Filter (+ optional re-triangulate).** Post-BA, strip observations that
//!    reproject past the gate (a contaminated union-find track) and drop tracks
//!    whose re-measured parallax is below the gate (depth-ambiguous far-flung
//!    points); optionally also **re-triangulate** against the BA-refined poses
//!    (`retriangulate`, off by default) — completing tracks the narrow seed-time
//!    baseline could not triangulate and re-seeding noisy points (guarded so an
//!    already-better point is never regressed), a density lever for downstream
//!    3DGS/NeRF — then re-optimise, a few rounds. No image is ever un-posed, so
//!    registration is invariant.
//!
//! The output ([`IncrementalSfmResult`]) carries per-image poses (`None` for
//! images that never registered) and merged multi-view tracks, ready for a
//! COLMAP `points3D.txt` export and downstream 3DGS / NeRF training.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::io::Write;

use nalgebra::{Matrix2x3, Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert, PoseRefiner};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_vision::two_view::{
    recover_relative_pose_with_options, CheiralityOptions, ConfigurationType, CorrespondenceGraph,
    RelativePoseEstimator, TwoViewCorrespondence,
};

use crate::process_memory;
use crate::{BaConfig, BaError, BaObservation, BaResult, BundleAdjustment, RobustKernel};

mod debug;
mod driver;
mod growth;
mod pose_guided;
mod refinement;
mod sequence_fallback;
mod structureless;
mod tracks;
mod triangulation;

pub use debug::*;
pub use driver::*;
pub(crate) use growth::*;
use pose_guided::*;
pub use refinement::*;
use sequence_fallback::*;
use structureless::*;
pub use tracks::*;
pub(crate) use triangulation::*;

/// Geometrically verified matches between two images of the set. The match
/// indices are keypoint indices into `features[image_i]` / `features[image_j]`,
/// and are assumed to have already survived an essential-matrix RANSAC (i.e.
/// they are inliers, not raw descriptor nearest neighbours).
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseMatches {
    /// Index of the first image into the `features` slice.
    pub image_i: usize,
    /// Index of the second image into the `features` slice.
    pub image_j: usize,
    /// Verified `(keypoint_in_i, keypoint_in_j)` correspondences.
    pub matches: Vec<(usize, usize)>,
    /// COLMAP two-view configuration when known (full E/F/H verifier). Used by
    /// global SfM to drop planar/panoramic pairs whose essential translation is
    /// ill-conditioned. `None` preserves legacy behaviour for essential-only
    /// verification paths.
    pub two_view_config: Option<ConfigurationType>,
    /// Essential-matrix inliers when the full verifier estimated E (may differ
    /// from [`Self::matches`] when F/H won the COLMAP inlier selection). Used
    /// by opt-in global edge construction so tracks stay dense while bearings
    /// come from E.
    pub essential_matches: Option<Vec<(usize, usize)>>,
    /// Essential matrix from full two-view verification (when estimated).
    /// Global SfM can decompose this directly for prefer-E edges instead of
    /// re-running E RANSAC on the inlier subset (which can flip chirality).
    pub essential_matrix: Option<Matrix3<f64>>,
}

impl PairwiseMatches {
    /// Construct matches without a known two-view configuration.
    pub const fn new(image_i: usize, image_j: usize, matches: Vec<(usize, usize)>) -> Self {
        Self {
            image_i,
            image_j,
            matches,
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        }
    }
}

/// Tunable knobs for [`incremental_sfm`].
#[derive(Debug, Clone, PartialEq)]
pub struct IncrementalSfmConfig {
    /// A pair must contribute at least this many verified matches to be a
    /// candidate seed pair. (Track building still uses *all* pairs.)
    pub min_seed_matches: usize,
    /// COLMAP `Mapper.init_min_tri_angle`: a seed pair is accepted only when
    /// the median triangulation angle of its well-triangulated inliers is at
    /// least this many degrees. `None` (default) keeps the count-only gate.
    pub seed_min_median_tri_angle_deg: Option<f64>,
    /// How many candidate seeds to grow before committing. The highest-match
    /// pair is not always a good seed: on repetitive structure (a building with
    /// near-identical façades) the most-overlapping pair can be an isolated local
    /// cluster of a few adjacent frames that the reconstruction cannot grow out
    /// of. So up to `seed_trials` candidate pairs are each grown and the one that
    /// registers the most images is kept — the COLMAP-style robust-initialisation
    /// pattern — committing early as soon as a seed reaches most of its connected
    /// component (so a well-connected scene still grows exactly one). Pairs that
    /// fail the two-view baseline gate place nothing and don't count against the
    /// budget. `1` restores the old first-qualifying-seed behaviour.
    pub seed_trials: usize,
    /// Maximum number of *growth attempts* in the seed search. Independent of
    /// [`Self::seed_trials`], which counts successful grows: a weak success
    /// (a temporally adjacent seed that only reaches a small fraction of the
    /// connected component) keeps consuming `seed_trials` under the old rule
    /// and exhausts the budget before the wide-baseline seeds later in the
    /// order are tried — the failure mode behind the OpenLORIS 10k collapse.
    /// `0` falls back to [`Self::seed_trials`] so existing configs are
    /// unchanged.
    pub seed_attempts: usize,
    /// Optional diagnostic restriction to one normalized `(image_i, image_j)`
    /// seed pair. `None` preserves the normal descending-match candidate list;
    /// this is intentionally opt-in so controlled seed replays do not alter
    /// ordinary reconstruction behavior.
    pub seed_pair: Option<(usize, usize)>,
    /// Minimum triangulation (parallax) angle in degrees for a point to be
    /// accepted. Small-angle triangulations are depth-unstable and dropped.
    pub min_triangulation_angle_deg: f64,
    /// Maximum reprojection error (px) for a triangulated point in each of the
    /// two views used to triangulate it, and the PnP inlier threshold.
    pub max_reprojection_error_px: f64,
    /// A track must span at least this many distinct images to be kept.
    pub min_track_length: usize,
    /// Optional final-only minimum track length.  When set, tracks shorter
    /// than this value are removed only after registration and all configured
    /// pose-guided splitting/recovery passes have completed, then the
    /// remaining support is re-triangulated and bundle-adjusted.  `None`
    /// preserves the historical growth/PnP/final-support behavior exactly.
    /// The example CLI currently exposes only `Some(3)` as its first guarded
    /// experiment; keeping this separate from `min_track_length` is what makes
    /// the diagnostic unable to change registration history.
    pub final_min_track_length: Option<usize>,
    /// Minimum PnP inliers to accept a new image registration.
    pub min_pnp_inliers: usize,
    /// Run a global bundle adjustment after every `ba_every` registrations.
    /// `0` disables the periodic BA (only the final BA runs).
    pub ba_every: usize,
    /// Defer the plain-growth periodic BA until at least this many cameras are
    /// registered. `0` preserves the historical `ba_every` schedule. This is
    /// intentionally scoped to the simple periodic path; COLMAP-style growth
    /// uses its own local/global schedule, and this knob never suppresses the
    /// configured final BA.
    pub periodic_ba_min_registered_images: usize,
    /// Run a final global bundle adjustment over the whole reconstruction.
    pub final_global_ba: bool,
    /// Bundle-adjustment configuration shared by the periodic and final solves.
    pub ba_config: BaConfig,
    /// Optional final fixed-support least-squares polish. When non-zero, one
    /// additional BA solve runs after all registration/refinement passes with
    /// the exact existing pose/track/observation support, no retriangulation or
    /// filtering, fixed intrinsics, and a pure L2 objective. A failed or
    /// cost-increasing solve is rolled back. `0` preserves the historical
    /// schedule exactly.
    pub final_ba_polish_iterations: usize,
    /// Minimal solver the PnP RANSAC uses to register each new image.
    pub pnp_solver: PnpSolver,
    /// Maximum absolute-pose RANSAC iterations. The dynamic termination
    /// (confidence 0.999) usually exits far earlier; this cap governs
    /// heavily contaminated correspondence sets.
    pub pnp_max_iterations: usize,
    /// Post-BA track-refinement rounds. Each round removes observations that
    /// reproject worse than `max_reprojection_error_px` after the global BA —
    /// the symptom of a contaminated union-find track whose merged 3D point
    /// fits none of its observations — and re-optimises. Registration is
    /// **invariant** (no image is ever un-posed), so this only cleans structure
    /// and can never drop a registered camera; on a clean reconstruction it is
    /// a near-no-op. `0` disables it.
    pub track_filter_iterations: usize,
    /// Re-triangulate tracks in each post-BA refinement round (COLMAP's
    /// completeness/refinement step the single-pass growth lacks). Once a global
    /// BA has moved the poses, two things change: a track that failed the
    /// parallax gate at growth time (a narrow baseline *then*) can now triangulate
    /// against the BA-refined wide-baseline views, and a point first triangulated
    /// from a narrow seed-time baseline can be re-seeded from the current widest
    /// pair. Completion is unconditional; the re-seed of an existing point is a
    /// **guarded swap** — kept only if it lowers that track's mean reprojection —
    /// so a multi-view point BA already placed better is never regressed. When
    /// enabled, at least one post-BA refinement round always runs (even if
    /// `track_filter_iterations` is `0`).
    ///
    /// **`false` by default.** Growth already triangulates greedily (every
    /// un-triangulated track is retried after *every* registration against all
    /// registered views), so by the end the structure is near-complete and the
    /// post-BA pass only mops up the marginal, gate-grazing tracks. Measured on a
    /// 300-frame EuRoC MH_03 monocular subset it adds ~3 % more tracks / ~1.5 %
    /// more observations — useful **density** for a downstream 3DGS/NeRF model —
    /// but is **ATE-neutral-to-slightly-negative** (Sim(3) 2.13 → 2.27 cm), since
    /// the extra tracks are the weakly-constrained ones. Enable it when you want
    /// the densest possible structure and can spend the extra BA rounds; leave it
    /// off when trajectory accuracy is the goal. See
    /// `docs/sfm_vs_colmap_benchmark.md`.
    pub retriangulate: bool,
    /// Build conflict-free tracks from the verified correspondence stream and
    /// re-triangulate every live point whenever registration adds an
    /// observation.  This is the incremental correspondence/point-map path:
    /// the ordinary union-find track builder remains the default, while this
    /// opt-in mode keeps one observation-to-point owner and refuses a merge
    /// that would create a same-image conflict.  It deliberately uses the
    /// plain seed/growth/PnP schedule; `--colmap-style` is rejected by the
    /// example CLI when this mode is selected.
    pub incremental_correspondence_triangulation: bool,
    /// Use COLMAP's `IncrementalMapper` bundle-adjustment **schedule** instead of
    /// the simple "global BA every `ba_every` registrations + final BA" path.
    /// This is a faithful port of COLMAP's defaults — the lever that closes the
    /// small-scene monocular accuracy gap on COLMAP's home turf:
    ///
    ///  - **Local BA after every registration.** Optimise only the new image and
    ///    its most-covisible neighbours (`local_ba_num_images`) plus the points
    ///    they see, holding the rest of the reconstruction fixed — cheap, and it
    ///    keeps the freshly added geometry tight before drift can compound.
    ///  - **Growth-triggered global refinement.** When the registered-image count
    ///    has grown by `global_ba_images_ratio` since the last global solve, run
    ///    an iterative global refinement: global BA → re-triangulate/complete →
    ///    filter, looped until the changed-observation fraction falls below
    ///    `global_ba_change_rate` (≤ `global_ba_max_refinements` rounds).
    ///  - **Registration retries.** A PnP failure is not permanent; after a
    ///    global refinement adds structure, failed images are retried, up to
    ///    `max_registration_trials` attempts each — COLMAP registers every frame
    ///    where the simple single-attempt path leaves a tail unregistered.
    ///
    /// The final refinement is always the iterative global form when this is on.
    /// `false` by default (preserves the simple schedule and every existing test).
    pub colmap_style_mapper: bool,
    /// After plain (non-`colmap_style_mapper`) growth, run COLMAP's iterative
    /// global refinement (multi-round BA + filter + re-triangulate) instead of
    /// the simple one-shot final BA. Keeps the simple growth schedule (no
    /// per-registration local BA) while borrowing only the final polish pass.
    /// `false` by default.
    pub final_iterative_global_refinement: bool,
    /// COLMAP `Mapper.ba_local_num_images`: how many most-covisible registered
    /// images (besides the newly registered one) the per-registration local BA
    /// optimises. Only used when `colmap_style_mapper` is set.
    pub local_ba_num_images: usize,
    /// Stop a local BA once an accepted LM step lowers the cost by less than
    /// this fraction (COLMAP's function tolerance). `None` keeps the shared
    /// `ba_config` tolerances, under which local windows almost always run
    /// the full iteration budget.
    pub local_ba_relative_cost_tolerance: Option<f64>,
    /// COLMAP `Mapper.ba_global_images_ratio`: trigger a global refinement once
    /// the registered-image count has grown by this factor since the last one.
    /// Only used when `colmap_style_mapper` is set.
    pub global_ba_images_ratio: f64,
    /// COLMAP `Mapper.ba_global_max_refinements`: max global BA → complete →
    /// filter rounds per global refinement. Only used when `colmap_style_mapper`.
    pub global_ba_max_refinements: usize,
    /// COLMAP `Mapper.ba_global_max_refinement_change_rate`: stop the global
    /// refinement loop once `changed_observations / total_observations` drops
    /// below this. Only used when `colmap_style_mapper` is set.
    pub global_ba_change_rate: f64,
    /// COLMAP `Mapper.max_reg_trials`: how many times a single image may be
    /// retried for registration (across global-refinement boundaries) before it
    /// is given up on. Only used when `colmap_style_mapper` is set.
    pub max_registration_trials: usize,
    /// After the final global refinement has tightened/re-triangulated the
    /// committed model, give every still-unregistered image one fresh PnP
    /// attempt against that updated structure. This is a bounded completion
    /// pass: counters are reset exactly once, no retry cycle is possible, and a
    /// second final refinement runs only when at least one image registers.
    /// Experimental and off by default.
    pub post_refinement_registration: bool,
    /// After ordinary 2D-3D PnP completion, try to place still-missing images
    /// from three or more registered neighbours' independently recovered
    /// relative poses. Translation scale is recovered by intersecting the
    /// neighbour-to-missing camera-centre direction lines in the existing
    /// reconstruction frame; a single essential pair is never sufficient.
    /// Experimental and off by default.
    pub structureless_registration: bool,
    /// After a successful PnP, require the absolute pose to agree (same
    /// translation hemisphere) with independent two-view essentials against
    /// already-registered neighbours. Rejects chirality-flipped façade
    /// registrations that still have low local reprojection. Default false.
    pub verify_registration_two_view: bool,
    /// Minimum neighbours with a usable two-view check before the gate may
    /// reject. Only used when `verify_registration_two_view` is set.
    pub verify_registration_min_neighbors: usize,
    /// Minimum fraction of checked neighbours that must agree (dot > 0).
    pub verify_registration_min_agree_fraction: f64,
    /// Maximum ascending-scan rounds of the structure-less completion pass.
    /// A single scan registers an image only when its consensus neighbours are
    /// *already* registered at the moment the scan reaches it, so a chain whose
    /// bridge image has a higher index than its dependent images (an island's
    /// entry point numbered above the images it unlocks — the courtyard-class
    /// second-component failure) is left behind by one pass. Each round feeds
    /// the images it registered back in as neighbours for the next round; the
    /// loop stops as soon as a round registers nothing. One round therefore
    /// reproduces the historical single-pass behaviour exactly.
    pub structureless_max_rounds: usize,
    /// Minimum registered relative-pose neighbours required to propose one
    /// structure-less camera pose.
    pub structureless_min_neighbors: usize,
    /// Minimum independently re-estimated essential inliers per neighbour.
    pub structureless_min_pair_inliers: usize,
    /// Maximum angular disagreement between neighbour-implied missing-camera
    /// rotations.
    pub structureless_max_rotation_disagreement_deg: f64,
    /// Minimum acute angle between any two camera-centre direction lines.
    pub structureless_min_intersection_angle_deg: f64,
    /// Maximum RMS line-intersection residual divided by the registered
    /// neighbour-centre spread.
    pub structureless_max_center_line_error_ratio: f64,
    /// Minimum signed neighbour-line parameter divided by neighbour spread.
    /// A small negative tolerance absorbs noisy intersections at an almost
    /// coincident adjacent frame without accepting a materially reversed
    /// essential translation direction.
    pub structureless_min_forward_ratio: f64,
    /// Minimum triangulated/reprojecting tracks required after tentative pose
    /// insertion and local refinement.
    pub structureless_min_support_tracks: usize,
    /// Maximum independent local-submap tracks synthesized from verified
    /// pairwise edges for one tentative structure-less insertion.
    pub structureless_max_local_tracks: usize,
    /// Minimum views per synthesized local-submap landmark. Two-view points
    /// are allowed because the camera pose itself already requires a separate
    /// multi-neighbour consensus.
    pub structureless_min_local_track_views: usize,
    /// Maximum mean reprojection error over the tentative image's supported
    /// tracks after local refinement.
    pub structureless_max_reprojection_error_px: f64,
    /// Maximum relative increase in the pre-existing model's mean reprojection
    /// error allowed when admitting one structure-less pose.
    pub structureless_max_clean_error_increase_ratio: f64,
    /// Revisit same-image-conflicted union-find components only after the normal
    /// reconstruction has produced trustworthy poses. Candidate landmarks are
    /// triangulated from verified edges, must agree in at least three registered
    /// views with cycle support, and enter one guarded global BA. The recovery is
    /// rolled back if it worsens the clean model's reprojection objective.
    /// Experimental and off by default.
    pub geometry_guided_conflict_recovery: bool,
    /// When enabled, allow a bounded sequence-aware registration fallback
    /// after ordinary PnP cannot place an image.  The example supplies unique
    /// numeric image-stem values through [`Self::sequence_stem_values`]; only
    /// an image whose stem is exactly one greater than an already registered
    /// predecessor is eligible.  The fallback uses a stable essential pose,
    /// the robust recent consecutive-step scale, and the normal triangulation
    /// gates before admitting a provisional pose.  `false` preserves the
    /// ordinary unordered PnP schedule exactly.
    pub sequence_relative_pose_fallback: bool,
    /// Defer sequence-relative fallback until the ordinary growth, conflict
    /// recovery, and post-refinement PnP stage has stalled.  After one
    /// provisional sequence pose is admitted, ordinary post-refinement PnP is
    /// resumed before another fallback is attempted.  This is experimental
    /// and off by default; `false` preserves eager fallback timing.
    pub sequence_fallback_after_post: bool,
    /// Use a constant-velocity projection of recent world-frame consecutive
    /// steps for the sequence fallback scale.  The projection is accepted
    /// only when positive, finite, and inside the existing robust median/MAD
    /// fence.  `false` preserves the historical median-magnitude estimator.
    pub sequence_constant_velocity_scale: bool,
    /// Use the constant-velocity projection without its strict local
    /// median/MAD fence.  Positive finite projections are still constrained
    /// to a broad 0.25x..4x recent-median scale range.  This is a separate
    /// experimental policy; `false` preserves both the strict projected mode
    /// and the historical median-magnitude mode.
    pub sequence_relaxed_constant_velocity_scale: bool,
    /// In the after-post sequence fallback, carry the accepted baseline
    /// magnitude from one consecutive provisional pose to the next.  The
    /// first fallback still uses the selected constant-velocity projection;
    /// a carried value is admitted only inside the broad 0.25x..4x
    /// recent-median sanity bounds.  A normal PnP/post registration clears
    /// the carry chain.  Experimental and off by default.
    pub sequence_fallback_carry_scale: bool,
    /// Trailing numeric stem values indexed like the feature/pose slices.
    /// `None` disables sequence lookup.  This metadata is intentionally kept
    /// outside `FeatureSet`, so library callers can opt in without changing
    /// feature files or the default unordered API.
    pub sequence_stem_values: Option<Vec<u64>>,
    /// Rebuild all legacy union-find components after a complete posed model
    /// exists.  The opt-in pass splits conflicting (and, when necessary,
    /// poorly fitting clean) components into deterministic 3-D hypotheses
    /// using fixed camera poses, then runs the ordinary final BA.  The legacy
    /// track builder and default schedule are unchanged when this is false.
    pub pose_guided_track_splitting: bool,
    /// Number of bounded outer pose-guided split attempts.  The value is
    /// ignored while `pose_guided_track_splitting` is false; its default of
    /// one preserves the original single-pass diagnostic.
    pub pose_guided_track_splitting_iterations: usize,
    /// Optional reprojection gate used only by pose-guided splitting.  `None`
    /// reuses `max_reprojection_error_px`, preserving the prior split exactly;
    /// the ordinary mapper's triangulation/PnP/filter gates are unaffected.
    pub pose_guided_split_max_reprojection_error_px: Option<f64>,
    /// When pose-guided splitting is enabled, require every observation added
    /// beyond its two-view anchor to have direct verified support from at
    /// least two distinct observations/images already in that hypothesis.
    /// Tracks with only the two-view anchor remain valid.  This is a separate
    /// experimental admission rule and is off by default so the original
    /// pose-guided partition remains reproducible.
    pub pose_guided_graph_support: bool,
    /// Before pose-guided splitting, opt into deterministic bridge-cut
    /// refinement of original correspondence components.  Only bridges whose
    /// two sides independently fit posed 3-D points while the combined side
    /// does not fit one point are cut; the legacy/default path is unchanged.
    pub pose_guided_bridge_cuts: bool,
    /// After pose-guided splitting, iteratively merge complementary tracks
    /// only when a verified cross-track edge exists and their union fits one
    /// posed 3-D point under the split reprojection gate.  The image sets of
    /// the two tracks must be disjoint, and one observation per image is
    /// enforced throughout.  `false` preserves the split-only partition.
    pub pose_guided_track_merging: bool,
    /// Optional reprojection gate used only while fitting post-split unions.
    /// `None` inherits `pose_guided_split_max_reprojection_error_px` (or the
    /// ordinary gate when the split override is absent).  The post-BA hard
    /// validation still uses `max_reprojection_error_px`.
    pub pose_guided_merge_max_reprojection_error_px: Option<f64>,
    /// Minimum registered views supporting a geometry-recovered conflict track.
    /// Values below three are clamped to three.
    pub conflict_recovery_min_views: usize,
    /// Maximum verified anchor edges tested per conflicted component, ranked by
    /// descending posed-view parallax. This bounds recovery work on large chains.
    pub conflict_recovery_max_hypotheses: usize,
    /// Per-observation reprojection gate for a geometry-recovered track.
    pub conflict_recovery_max_reprojection_error_px: f64,
    /// Maximum mean reprojection error of a recovered track before guarded BA.
    pub conflict_recovery_max_mean_reprojection_px: f64,
    /// Maximum relative increase allowed in the original clean tracks' mean
    /// reprojection after the single guarded recovery BA.
    pub conflict_recovery_max_clean_error_increase_ratio: f64,
    /// Multi-view exemption to the `min_triangulation_angle_deg` gate. A point on
    /// a forward-flying trajectory often subtends a parallax angle below the gate
    /// yet is **well-constrained** when many views observe it (each view adds a
    /// reprojection constraint on its 3 DoF). `None` keeps the strict angle gate
    /// for every track (the simple path). `Some(n)` keeps — and triangulates — a
    /// track whose widest parallax is between `low_parallax_min_angle_deg` and
    /// `min_triangulation_angle_deg` **if it has ≥ n registered observations**, so
    /// long low-parallax tracks survive while 2-view depth-ambiguous ones (which
    /// would slide freely along their ray and corrupt the poses) are still
    /// rejected. This is the lever that recovers COLMAP-grade structure density on
    /// forward-motion video without the accuracy collapse a blanket low gate
    /// causes. Used by both the simple and COLMAP-style paths when set.
    pub low_parallax_min_observations: Option<usize>,
    /// Lower parallax floor (degrees) for the multi-view exemption above: a track
    /// below this angle is dropped regardless of how many views see it (truly
    /// degenerate). Only consulted when `low_parallax_min_observations` is `Some`.
    pub low_parallax_min_angle_deg: f64,
    /// Refine the shared pinhole intrinsics `(fx, fy, cx, cy)` in the **final**
    /// global refinement (alternating BA ↔ intrinsics, see
    /// [`crate::BaConfig::refine_intrinsics`]). A slightly-off fixed calibration
    /// forces a residual onto the poses; letting the camera absorb it is COLMAP's
    /// lever for the last of the small-scene accuracy gap. Growth keeps the input
    /// intrinsics fixed; the refined camera emerges from the final solve and is
    /// returned in [`IncrementalSfmResult::refined_camera`]. `false` by default.
    pub refine_intrinsics: bool,
    /// COLMAP `Reconstruction::FilterImages`: after each growth global refinement,
    /// **de-register** any image whose count of well-supported 3D-point
    /// observations (triangulated, within `max_reprojection_error_px`) has fallen
    /// below `filter_min_image_observations`. A pose that BA + point filtering
    /// stripped of support is unreliable; dropping it (its trial counter resets, so
    /// it can re-register once the structure around it improves) keeps a bad pose
    /// from dragging the global solve. The two seed images are never filtered (they
    /// anchor the gauge), and the registered count is never taken below 3. Only
    /// used when `colmap_style_mapper` is set. `false` by default.
    pub filter_images: bool,
    /// Minimum well-supported observations an image must keep to stay registered
    /// under `filter_images`. Only consulted when `filter_images` is set.
    pub filter_min_image_observations: usize,
    /// Which algorithm builds step 1's feature tracks from `pairwise`. See
    /// [`TrackSource`]'s doc for the M2 background; `UnionFind` by default.
    pub track_source: TrackSource,
    /// Build tracks by processing verified correspondences in descending
    /// pair-level confidence and rejecting a merge that would introduce two
    /// keypoints from one image into a track. The confidence is deliberately
    /// limited to metadata retained by [`PairwiseMatches`]: verified inlier
    /// count, then essential-inlier count, with deterministic image/keypoint
    /// tie-breaks. When enabled it takes precedence over `track_source`.
    /// `false` preserves the legacy union-find/graph path exactly.
    pub confidence_ordered_tracks: bool,
    /// Build tracks with an opt-in per-correspondence geometric order. For
    /// finite E-supported matches from a `Calibrated` two-view model, the
    /// normalized Sampson residual is ordered first; F/H, degenerate, missing
    /// or invalid models deliberately fall back to the pair-level confidence
    /// order above rather than mixing incomparable residuals. Takes precedence
    /// over [`Self::confidence_ordered_tracks`]. `false` preserves the legacy
    /// path exactly.
    pub geometric_confidence_tracks: bool,
    /// Canonicalize track and observation iteration by stable physical
    /// feature keys (image id, quantized pixel coordinates, then descriptor
    /// contents) instead of input feature indices. This is useful when a
    /// caller has permuted feature rows but kept their coordinates and
    /// descriptors paired; `false` preserves the legacy index order exactly.
    pub stable_track_order: bool,
    /// Build tracks by prioritising accepted correspondences with stronger
    /// multi-view cycle support. For an edge `(i,a)-(j,b)`, a third image
    /// contributes support when both endpoints connect to the same feature in
    /// that image; distinct supporting images are ordered before the exact
    /// number of matching third-image features. Pair-level and, when safely
    /// available, calibrated-E residual confidence break ties. The legacy
    /// union-find path remains unchanged when this is `false`.
    pub cycle_supported_tracks: bool,
    /// Run one opt-in final fixed-support BA with observation weights derived
    /// from the pre-BA triangulation geometry. The proxy is a clamped,
    /// median-normalized `sin²(parallax)` information score; it changes no
    /// track or observation membership and is disabled by default.
    pub geometry_weighted_ba: bool,
    /// Apply a conditioning safeguard to landmark variables whose pre-BA point
    /// block is numerically ill-conditioned and whose current residual is
    /// already outside the reprojection gate. Such landmarks are excluded
    /// from this BA's residual rows (a fixed, high-residual point would exert
    /// a misleading camera pull). The deterministic geometry gate is disabled
    /// by default; well-fitting weak points remain ordinary BA variables.
    pub freeze_ill_conditioned_landmarks: bool,
    /// Run a bounded point-only BA with all currently registered camera poses
    /// and intrinsics fixed immediately before each global/periodic joint BA.
    /// This can absorb a large landmark correction before it enters the joint
    /// camera Schur system. `0` preserves the historical schedule exactly.
    pub landmark_ba_warm_start_iterations: usize,
    /// Minimum registered-camera count at which the landmark warm start is
    /// enabled. `0` applies it to every global/periodic BA; a positive value
    /// permits evidence-scoped experiments such as the first 27-camera BA.
    pub landmark_ba_warm_start_min_registered_images: usize,
    /// Optional COLMAP sparse-model poses used only by the registration-time
    /// `sfm-debug-oracle` transition log. The entries are indexed like
    /// `features` and `poses`; `None` disables the diagnostic completely.
    /// Supplying this field never changes registration, triangulation, or BA.
    pub debug_oracle_poses: Option<Vec<Option<Pose>>>,
    /// How unregistered images are ranked for the next PnP attempt. Raw
    /// correspondence count is the historical default; the visibility pyramid
    /// is an explicit opt-in for experiments that prefer spatial coverage.
    pub next_image_policy: NextImagePolicy,
    /// Seed pairs (`image_i`, `image_j`, normalized to `(min, max)`) that
    /// [`seed_candidate_order`] must skip. Empty by default. This is the
    /// mechanism `LocalSubmapBuilder::build`'s scale-pathology retry (see
    /// `crate::local_submap`, `NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(b)) uses
    /// to force a rebuild onto the *next*-ranked seed candidate after a
    /// previous seed pair (which reached `88/88` registration with
    /// unremarkable per-observation gates) produced an internally
    /// scale-exploded reconstruction: excluding the offending pair and
    /// re-running `incremental_sfm` deterministically walks to the next
    /// candidate in the same descending-match-count order, without
    /// perturbing any other seed-selection behaviour.
    pub excluded_seed_pairs: HashSet<(usize, usize)>,
}

/// Ranking policy for the next image offered to incremental PnP registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NextImagePolicy {
    /// Try [`VisibilityPyramid`] first and, when it leaves any input image
    /// unregistered, rerun from the same immutable inputs with
    /// [`CorrespondenceCount`].  The better reconstruction is chosen
    /// deterministically by registered images, valid observations, tracks,
    /// then lower reprojection error.  Ties retain the visibility result.
    /// This policy is explicit and does not change the library default.
    Auto,
    /// Prefer spatially distributed 2D-3D support, then raw support count.
    VisibilityPyramid,
    /// Prefer the largest raw 2D-3D support count (the historical policy).
    #[default]
    CorrespondenceCount,
}

/// Which algorithm builds step 1's feature tracks from `pairwise` — the M2
/// port in `docs/colmap_port_plan.md` ("Persistent `CorrespondenceGraph`").
/// [`Self::UnionFind`] is the original ad hoc union-find
/// ([`build_tracks`]), kept as the default (see the M2 results section in
/// that doc for the ETH3D A/B that motivated staying opt-in rather than
/// flipping the default). [`Self::CorrespondenceGraph`] instead builds the
/// same tracks by routing through
/// `visloc_vision::two_view::correspondence_graph::CorrespondenceGraph`
/// ([`build_tracks_via_graph`]) — COLMAP's persistent view-graph object,
/// which also exposes `NumObservationsForImage`/`NumCorrespondencesForImage`/
/// `ExtractTransitiveCorrespondences`-style queries the union-find has no way
/// to answer, for future milestones (M4's transitive pairing, in particular).
/// Both paths are proven to produce byte-identical tracks on this crate's
/// existing fixtures (see the `graph_tracks_match_union_find_tracks_*` tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrackSource {
    /// The original ad hoc union-find over `(image, keypoint)` nodes.
    #[default]
    UnionFind,
    /// COLMAP-style persistent [`CorrespondenceGraph`] (M2 port).
    CorrespondenceGraph,
}

/// Minimal PnP solver used to register a new image against the reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PnpSolver {
    /// 6-point Direct Linear Transform. Linear and fast, but **degenerate on
    /// coplanar points** — a flat building façade or planar patch yields a
    /// garbage pose. Kept for parity with the classic path.
    Dlt,
    /// Grunert's Perspective-Three-Point minimal solver. Geometrically
    /// well-posed for any three non-collinear points whether or not the scene
    /// is planar, so it registers planar façades the DLT cannot. The default.
    #[default]
    P3p,
}

impl Default for IncrementalSfmConfig {
    fn default() -> Self {
        Self {
            min_seed_matches: 30,
            seed_min_median_tri_angle_deg: None,
            seed_trials: 12,
            seed_attempts: 0,
            seed_pair: None,
            min_triangulation_angle_deg: 2.0,
            max_reprojection_error_px: 4.0,
            min_track_length: 2,
            final_min_track_length: None,
            min_pnp_inliers: 12,
            ba_every: 5,
            periodic_ba_min_registered_images: 0,
            final_global_ba: true,
            ba_config: BaConfig {
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                ..BaConfig::default()
            },
            final_ba_polish_iterations: 0,
            pnp_solver: PnpSolver::default(),
            // Legacy default: fixed 128-sample PnP with no dynamic
            // termination. Raising this above 128 opts into the COLMAP-style
            // confidence-based adaptive budget for large correspondence
            // sets.
            pnp_max_iterations: 128,
            track_filter_iterations: 2,
            retriangulate: false,
            incremental_correspondence_triangulation: false,
            // COLMAP IncrementalMapper defaults (off unless colmap_style_mapper).
            colmap_style_mapper: false,
            final_iterative_global_refinement: false,
            local_ba_num_images: 8,
            local_ba_relative_cost_tolerance: None,
            global_ba_images_ratio: 1.1,
            global_ba_max_refinements: 5,
            global_ba_change_rate: 0.0005,
            max_registration_trials: 3,
            post_refinement_registration: false,
            structureless_registration: false,
            verify_registration_two_view: false,
            verify_registration_min_neighbors: 2,
            verify_registration_min_agree_fraction: 0.5,
            structureless_max_rounds: 4,
            structureless_min_neighbors: 3,
            structureless_min_pair_inliers: 30,
            structureless_max_rotation_disagreement_deg: 3.0,
            structureless_min_intersection_angle_deg: 2.0,
            structureless_max_center_line_error_ratio: 0.25,
            structureless_min_forward_ratio: -0.005,
            structureless_min_support_tracks: 20,
            structureless_max_local_tracks: 512,
            structureless_min_local_track_views: 2,
            structureless_max_reprojection_error_px: 2.0,
            structureless_max_clean_error_increase_ratio: 0.001,
            geometry_guided_conflict_recovery: false,
            sequence_relative_pose_fallback: false,
            sequence_fallback_after_post: false,
            sequence_constant_velocity_scale: false,
            sequence_relaxed_constant_velocity_scale: false,
            sequence_fallback_carry_scale: false,
            sequence_stem_values: None,
            pose_guided_track_splitting: false,
            pose_guided_track_splitting_iterations: 1,
            pose_guided_split_max_reprojection_error_px: None,
            pose_guided_graph_support: false,
            pose_guided_bridge_cuts: false,
            pose_guided_track_merging: false,
            pose_guided_merge_max_reprojection_error_px: None,
            conflict_recovery_min_views: 3,
            conflict_recovery_max_hypotheses: 32,
            conflict_recovery_max_reprojection_error_px: 2.0,
            conflict_recovery_max_mean_reprojection_px: 1.0,
            conflict_recovery_max_clean_error_increase_ratio: 0.001,
            low_parallax_min_observations: None,
            low_parallax_min_angle_deg: 1.0,
            refine_intrinsics: false,
            filter_images: false,
            filter_min_image_observations: 15,
            track_source: TrackSource::default(),
            confidence_ordered_tracks: false,
            geometric_confidence_tracks: false,
            stable_track_order: false,
            cycle_supported_tracks: false,
            geometry_weighted_ba: false,
            freeze_ill_conditioned_landmarks: false,
            landmark_ba_warm_start_iterations: 0,
            landmark_ba_warm_start_min_registered_images: 0,
            debug_oracle_poses: None,
            next_image_policy: NextImagePolicy::default(),
            excluded_seed_pairs: HashSet::new(),
        }
    }
}

/// One reconstructed 3D point and the image observations that support it.
#[derive(Debug, Clone, PartialEq)]
pub struct SfmTrack {
    /// World-frame position (metres up to the monocular gauge scale).
    pub position: Point3<f64>,
    /// `(image_index, keypoint_index, pixel)` for every registered image that
    /// observes this point.
    pub observations: Vec<(usize, usize, Point2<f64>)>,
}

/// Diagnostics from feature-track construction before triangulation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackBuildStats {
    /// Verified pairwise correspondences offered to the track builder.
    pub input_correspondences: usize,
    /// Connected components formed before the minimum-length gate.
    pub connected_components: usize,
    /// Legacy components discarded because they contain one image twice.
    pub conflicting_components: usize,
    /// Observations contained in those discarded legacy components.
    pub conflicting_observations: usize,
    /// Tracks retained after conflict and minimum-length gates.
    pub retained_tracks: usize,
    /// Observations in retained tracks.
    pub retained_observations: usize,
}

/// Bounded topology diagnostics for the confidence-ordered rig-track policy.
///
/// This preview intentionally reports only the conflict regions induced by
/// rejected confidence-ordered edges. It does not change the normal builder,
/// materialize alternate tracks, or retain any state for a mapper run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PairConfidenceConflictStats {
    /// Number of verified match rows offered to the confidence ordering.
    pub correspondences: usize,
    /// Number of distinct `(image, keypoint)` endpoints touched by a row.
    pub nodes: usize,
    /// Number of edges that joined two previously distinct, image-compatible
    /// DSU components.
    pub accepted_edges: usize,
    /// Number of edges rejected because their two components shared an image.
    pub rejected_edges: usize,
    /// Number of final DSU components after accepted unions.
    pub final_components: usize,
    /// Number of connected regions in the graph of final components induced by
    /// rejected edges.
    pub conflict_regions: usize,
    /// Number of final components involved in at least one conflict region.
    pub involved_components: usize,
    /// Sum of observations in each involved final component, counted once per
    /// disjoint region.
    pub involved_observations: usize,
    /// Largest number of final components in one conflict region.
    pub max_region_components: usize,
    /// Largest number of observations in one conflict region.
    pub max_region_observations: usize,
    /// Largest number of images shared by the two components at a rejected
    /// edge, measured at the rejection point.
    pub max_overlapping_images_per_rejected_edge: usize,
    /// Histogram of conflict-region component counts as deterministic
    /// ascending `(component_count, region_count)` entries.
    pub region_component_count_histogram: Vec<(usize, usize)>,
}

/// Output of [`incremental_sfm`].
#[derive(Debug, Clone)]
pub struct IncrementalSfmResult {
    /// Refined pose per input image; `None` for images that never registered.
    pub poses: Vec<Option<Pose>>,
    /// Reconstructed multi-view tracks (after the final BA, if enabled).
    pub tracks: Vec<SfmTrack>,
    /// Track-construction diagnostics measured before triangulation and BA.
    pub track_build_stats: TrackBuildStats,
    /// Number of images that registered into the reconstruction.
    pub registered_images: usize,
    /// Images added by the optional one-shot post-refinement completion pass.
    pub post_refinement_registered_images: usize,
    /// Images placed by the optional multi-neighbour relative-pose recovery
    /// after the ordinary post-refinement PnP pass.
    pub structureless_registered_images: usize,
    /// Conflict tracks admitted by the optional geometry-guided recovery gate.
    pub geometry_recovered_tracks: usize,
    /// Observations contained in admitted geometry-recovered tracks.
    pub geometry_recovered_observations: usize,
    /// Whether recovery was allowed to update poses through its guarded BA.
    /// Complete models use structure-only recovery and report `false`.
    pub geometry_recovery_pose_ba_applied: bool,
    /// Mean reprojection error (px) over every observation of every track.
    pub mean_reprojection_px: f64,
    /// Result of the final BA solve, if one ran.
    pub ba_result: Option<BaResult>,
    /// Refined camera intrinsics, when `config.refine_intrinsics` was set. The
    /// poses, tracks, and `mean_reprojection_px` are all expressed against *this*
    /// camera, so a COLMAP / 3DGS export must use it rather than the input camera.
    /// `None` when intrinsics refinement was off.
    pub refined_camera: Option<Camera>,
    /// Local index (into this call's `features`/`pairwise`) of the first
    /// image of the seed pair the winning growth trial started from.
    /// Purely observational — does not influence poses/tracks/gates.
    pub seed_image_i: usize,
    /// Local index of the second image of the winning seed pair.
    pub seed_image_j: usize,
    /// Number of verified matches in the winning seed pair (`pairwise[..].matches.len()`).
    pub seed_match_count: usize,
}

/// Why [`incremental_sfm`] could not build a reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncrementalSfmError {
    /// No verified pair met `min_seed_matches` / parallax to bootstrap from.
    NoSeedPair,
    /// The chosen seed pair's relative pose / initial triangulation failed.
    SeedInitFailed,
    /// An externally supplied diagnostic track partition violated the mapper's
    /// one-observation-per-image/index contract.
    InvalidTrackMembership(String),
    /// An opt-in initial-pose model did not satisfy the mapper's input
    /// contract (one pose per image, at least two finite poses).
    InvalidInitialPoses(String),
    /// A bundle-adjustment solve failed.
    Ba(BaError),
}

impl std::fmt::Display for IncrementalSfmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncrementalSfmError::NoSeedPair => {
                write!(f, "no verified pair met the seed criteria")
            }
            IncrementalSfmError::SeedInitFailed => {
                write!(f, "seed pair relative-pose / triangulation failed")
            }
            IncrementalSfmError::InvalidTrackMembership(message) => {
                write!(f, "invalid track membership: {message}")
            }
            IncrementalSfmError::InvalidInitialPoses(message) => {
                write!(f, "invalid initial poses: {message}")
            }
            IncrementalSfmError::Ba(e) => write!(f, "bundle adjustment failed: {e:?}"),
        }
    }
}

impl std::error::Error for IncrementalSfmError {}

#[cfg(test)]
mod tests;
