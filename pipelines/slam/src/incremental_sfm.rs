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

use std::collections::{HashMap, HashSet};

use nalgebra::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_vision::two_view::{
    ConfigurationType, CorrespondenceGraph, RelativePoseEstimator, TwoViewCorrespondence,
};

use crate::{BaConfig, BaError, BaObservation, BaResult, BundleAdjustment, RobustKernel};

/// Gate for the mapper's diagnostic `eprintln!`s (seed-sweep reach, growth
/// stalls/recoveries). Off by default (checking an env var per print site is
/// cheap; this is not a hot inner loop). Added for the M4 path-dependence
/// diagnosis in `docs/colmap_port_plan.md` — set `VISLOC_SFM_DEBUG=1` to see,
/// per seed trial, how far it grew, and, per growth stall, whether it was a
/// genuine correspondence shortfall or a trial-budget exhaustion, and whether
/// the stall-recovery refinement ([`grow_from_seed`]'s `stalled_once`) helped.
fn sfm_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG").is_some()
}

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
}

/// Tunable knobs for [`incremental_sfm`].
#[derive(Debug, Clone, PartialEq)]
pub struct IncrementalSfmConfig {
    /// A pair must contribute at least this many verified matches to be a
    /// candidate seed pair. (Track building still uses *all* pairs.)
    pub min_seed_matches: usize,
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
    /// Minimum triangulation (parallax) angle in degrees for a point to be
    /// accepted. Small-angle triangulations are depth-unstable and dropped.
    pub min_triangulation_angle_deg: f64,
    /// Maximum reprojection error (px) for a triangulated point in each of the
    /// two views used to triangulate it, and the PnP inlier threshold.
    pub max_reprojection_error_px: f64,
    /// A track must span at least this many distinct images to be kept.
    pub min_track_length: usize,
    /// Minimum PnP inliers to accept a new image registration.
    pub min_pnp_inliers: usize,
    /// Run a global bundle adjustment after every `ba_every` registrations.
    /// `0` disables the periodic BA (only the final BA runs).
    pub ba_every: usize,
    /// Run a final global bundle adjustment over the whole reconstruction.
    pub final_global_ba: bool,
    /// Bundle-adjustment configuration shared by the periodic and final solves.
    pub ba_config: BaConfig,
    /// Minimal solver the PnP RANSAC uses to register each new image.
    pub pnp_solver: PnpSolver,
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
    /// COLMAP `Mapper.ba_local_num_images`: how many most-covisible registered
    /// images (besides the newly registered one) the per-registration local BA
    /// optimises. Only used when `colmap_style_mapper` is set.
    pub local_ba_num_images: usize,
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
    /// After the final iterative global refinement has tightened/re-triangulated
    /// the committed model, give every still-unregistered image one fresh PnP
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
    /// How unregistered images are ranked for the next PnP attempt. The
    /// visibility pyramid is the safe default; correspondence count preserves
    /// the pre-M3 ordering for controlled regression experiments.
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
    /// Prefer spatially distributed 2D-3D support, then raw support count.
    #[default]
    VisibilityPyramid,
    /// Prefer the largest raw 2D-3D support count (the pre-`9c35f72` policy).
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
            seed_trials: 12,
            min_triangulation_angle_deg: 2.0,
            max_reprojection_error_px: 4.0,
            min_track_length: 2,
            min_pnp_inliers: 12,
            ba_every: 5,
            final_global_ba: true,
            ba_config: BaConfig {
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                ..BaConfig::default()
            },
            pnp_solver: PnpSolver::default(),
            track_filter_iterations: 2,
            retriangulate: false,
            // COLMAP IncrementalMapper defaults (off unless colmap_style_mapper).
            colmap_style_mapper: false,
            local_ba_num_images: 8,
            global_ba_images_ratio: 1.1,
            global_ba_max_refinements: 5,
            global_ba_change_rate: 0.0005,
            max_registration_trials: 3,
            post_refinement_registration: false,
            structureless_registration: false,
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

#[derive(Debug, Clone, Default)]
struct TrackBuildOutput {
    tracks: Vec<Vec<(usize, usize)>>,
    conflicting_components: Vec<Vec<(usize, usize)>>,
    stats: TrackBuildStats,
}

fn build_track_output(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> TrackBuildOutput {
    match config.track_source {
        TrackSource::UnionFind => {
            build_tracks_detailed(features.len(), pairwise, config.min_track_length)
        }
        TrackSource::CorrespondenceGraph => {
            let tracks = build_tracks_via_graph(features, pairwise, config.min_track_length);
            let stats = TrackBuildStats {
                input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
                connected_components: tracks.len(),
                retained_tracks: tracks.len(),
                retained_observations: tracks.iter().map(Vec::len).sum(),
                ..TrackBuildStats::default()
            };
            TrackBuildOutput {
                tracks,
                conflicting_components: Vec::new(),
                stats,
            }
        }
    }
}

/// Build only the feature-track topology and return its diagnostics, without
/// seed selection, triangulation, registration, or bundle adjustment. This is
/// intended for cheap preflight rejection of a candidate view graph before an
/// expensive independent mapper arm is launched.
pub fn preview_track_build_stats(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> TrackBuildStats {
    build_track_output(features, pairwise, config).stats
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
            IncrementalSfmError::Ba(e) => write!(f, "bundle adjustment failed: {e:?}"),
        }
    }
}

impl std::error::Error for IncrementalSfmError {}

/// A grown reconstruction the seed search compares: how many images it
/// registered, the per-image poses and the per-track points.
type SeedGrowth = (usize, Vec<Option<Pose>>, Vec<Option<Point3<f64>>>, Camera);

/// Run incremental SfM over an unordered image set.
///
/// `features[k]` are the keypoints + descriptors of image `k`; `pairwise` are
/// the geometrically verified matches between image pairs. Returns the refined
/// poses and merged tracks, or an [`IncrementalSfmError`] if no reconstruction
/// could be bootstrapped.
pub fn incremental_sfm(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    let sfm_started = std::time::Instant::now();
    let n_images = features.len();

    // ---- 1. Build feature tracks (M2: union-find or CorrespondenceGraph) ----
    let started = std::time::Instant::now();
    let track_build = build_track_output(features, pairwise, config);
    let TrackBuildOutput {
        mut tracks,
        conflicting_components,
        stats: track_build_stats,
    } = track_build;
    let track_build_seconds = started.elapsed().as_secs_f64();
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: track build source={:?} input={} components={} \
             conflicts={} conflict_obs={} retained_tracks={} retained_obs={}",
            config.track_source,
            track_build_stats.input_correspondences,
            track_build_stats.connected_components,
            track_build_stats.conflicting_components,
            track_build_stats.conflicting_observations,
            track_build_stats.retained_tracks,
            track_build_stats.retained_observations,
        );
    }

    // For each image, which (keypoint, track) pairs it observes — drives both
    // triangulation and next-image selection.
    let mut obs_by_image: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_images];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, kp) in track {
            obs_by_image[image].push((kp, track_id));
        }
    }

    // ---- 2. Seed selection: try several candidate seeds, keep the largest ----
    // The highest-match pair is not always a good seed. On repetitive structure
    // (a building photographed around near-identical façades) the most-overlapping
    // verified pair can be a handful of adjacent frames that triangulate fine but
    // form an isolated local cluster the reconstruction cannot grow out of. So
    // walk verified pairs in descending match order and keep the reconstruction
    // that registers the most images, committing as soon as one is *not trapped*
    // — reaches at least half of its connected component. A well-connected scene
    // (the strongest pair is already central) commits on the first candidate that
    // places, growing exactly one reconstruction, just as the old
    // first-qualifying-seed path did; only a repetitive scene whose strongest
    // pairs are isolated clusters keeps searching, and then takes the
    // farthest-reaching seed found. Each grow runs its periodic BA, so reach is
    // measured on the real (bundle-adjusted) trajectory, not a drifting proxy.
    //
    // `seed_trials` caps how many pairs actually *grow* a reconstruction; pairs
    // that fail the two-view baseline gate placed nothing and are skipped for
    // free, so an orbit whose highest-overlap pairs are all low-parallax adjacent
    // frames still reaches the first wide-baseline pair beyond them.
    let seed_growth_started = std::time::Instant::now();
    let seed_order = seed_candidate_order(pairwise, config);
    let trials = config.seed_trials.max(1);
    let not_trapped = largest_connected_component(pairwise, n_images)
        .div_ceil(2)
        .max(1);
    let mut best: Option<SeedGrowth> = None;
    // Tracks which `pairwise` entry produced `best`, purely for observability
    // (the per-submap build summary log wants to report which image pair was
    // actually chosen as the seed). Always `Some` exactly when `best` is,
    // updated in lockstep below.
    let mut best_pi: Option<usize> = None;
    let mut grows = 0usize;
    for &pi in &seed_order {
        let (trial_poses, trial_points, reach, trial_cam) = grow_from_seed(
            camera,
            features,
            &tracks,
            &obs_by_image,
            config,
            &pairwise[pi],
        )?;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: seed trial {pi} pair=({}, {}) matches={} -> reach={reach}",
                pairwise[pi].image_i,
                pairwise[pi].image_j,
                pairwise[pi].matches.len(),
            );
        }
        if reach == 0 {
            continue; // pair failed the seed gate — nothing placed, no grow ran
        }
        grows += 1;
        if best
            .as_ref()
            .is_none_or(|(best_reach, _, _, _)| reach > *best_reach)
        {
            best = Some((reach, trial_poses, trial_points, trial_cam));
            best_pi = Some(pi);
        }
        if reach >= not_trapped || grows >= trials {
            break;
        }
    }
    let (_, mut poses, mut track_point, grown_cam) = best.ok_or(IncrementalSfmError::NoSeedPair)?;
    let winning_pi = best_pi.expect("set together with `best` on every assignment above");
    let seed_image_i = pairwise[winning_pi].image_i;
    let seed_image_j = pairwise[winning_pi].image_j;
    let seed_match_count = pairwise[winning_pi].matches.len();
    let seed_growth_seconds = seed_growth_started.elapsed().as_secs_f64();

    // ---- 4 + 5. Final refinement ----
    // When intrinsics refinement is on, growth already co-evolved them into
    // `grown_cam` (COLMAP keeps the camera moving with the structure so a wrong
    // focal cannot be silently absorbed). The final solve continues refining from
    // there; `cam` expresses the output poses/tracks/reprojection and is returned
    // to the caller for export.
    let final_refinement_started = std::time::Instant::now();
    let mut cam = grown_cam;
    let mut ba_result = if config.colmap_style_mapper {
        // COLMAP's final pass IS an iterative global refinement (global BA →
        // complete/re-triangulate → filter, to convergence). The grow loop has
        // already run local BAs + growth-triggered refinements throughout.
        Some(
            iterative_global_refinement(
                &mut cam,
                features,
                &mut tracks,
                config,
                &mut poses,
                &mut track_point,
            )
            .map_err(IncrementalSfmError::Ba)?,
        )
    } else {
        // Simple schedule: one final global BA, then a few filter (+ optional
        // re-triangulate) rounds. With re-triangulation on, run at least one
        // round even when the filter budget is zero — the completion/re-seed pass
        // is the point of the round.
        let mut ba_result = if config.final_global_ba {
            let (res, refined) = run_bundle_adjustment(
                &cam,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
                config.refine_intrinsics,
            )
            .map_err(IncrementalSfmError::Ba)?;
            if let Some(c) = refined {
                cam = c;
            }
            Some(res)
        } else {
            None
        };
        let refine_rounds = if config.retriangulate {
            config.track_filter_iterations.max(1)
        } else {
            config.track_filter_iterations
        };
        for _ in 0..refine_rounds {
            let removed = filter_outlier_observations(
                &cam,
                features,
                &mut tracks,
                config,
                &poses,
                &mut track_point,
            );
            let retriangulated = if config.retriangulate {
                retriangulate_tracks(&cam, features, &tracks, config, &poses, &mut track_point)
            } else {
                0
            };
            if removed == 0 && retriangulated == 0 {
                break;
            }
            let (res, refined) = run_bundle_adjustment(
                &cam,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
                config.refine_intrinsics,
            )
            .map_err(IncrementalSfmError::Ba)?;
            if let Some(c) = refined {
                cam = c;
            }
            ba_result = Some(res);
        }
        ba_result
    };

    let mut geometry_recovered_tracks = 0usize;
    let mut geometry_recovered_observations = 0usize;
    let mut geometry_recovery_pose_ba_applied = false;
    let geometry_recovery_started = std::time::Instant::now();
    if config.geometry_guided_conflict_recovery && !conflicting_components.is_empty() {
        let recovered = recover_conflict_tracks_geometry(
            &cam,
            features,
            pairwise,
            &conflicting_components,
            &poses,
            config,
        );
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: geometry conflict recovery proposed {} tracks / {} observations",
                recovered.len(),
                recovered
                    .iter()
                    .map(|track| track.observations.len())
                    .sum::<usize>(),
            );
        }
        if !recovered.is_empty() {
            let clean_track_count = tracks.len();
            let clean_mean_before = mean_reprojection_for_track_range(
                &cam,
                features,
                &tracks,
                &poses,
                &track_point,
                0,
                clean_track_count,
            );
            let poses_before = poses.clone();
            let track_point_before = track_point.clone();
            for candidate in &recovered {
                tracks.push(candidate.observations.clone());
                track_point.push(Some(candidate.point));
            }

            // Once every image is already registered, conflict recovery is a
            // structure-density operation. The held-out MH_01 A/B showed that
            // a residual-improving extra pose BA can still worsen independent
            // GT ATE, so a complete trajectory is immutable here. Incomplete
            // models may use one guarded BA because recovered structure can
            // unlock missing-image PnP and improve the development trajectory.
            let model_complete = poses.iter().all(Option::is_some);
            let mut accepted = model_complete;
            if model_complete {
                geometry_recovered_tracks = recovered.len();
                geometry_recovered_observations =
                    recovered.iter().map(|track| track.observations.len()).sum();
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: geometry conflict recovery accepted structure-only; \
                         complete {}/{} pose model remains byte-identical",
                        poses.len(),
                        poses.len(),
                    );
                }
            } else if let Ok((result, _)) = run_bundle_adjustment(
                &cam,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
                false,
            ) {
                let clean_mean_after = mean_reprojection_for_track_range(
                    &cam,
                    features,
                    &tracks,
                    &poses,
                    &track_point,
                    0,
                    clean_track_count,
                );
                let recovered_mean_after = mean_reprojection_for_track_range(
                    &cam,
                    features,
                    &tracks,
                    &poses,
                    &track_point,
                    clean_track_count,
                    tracks.len(),
                );
                let allowed_clean_mean = clean_mean_before
                    * (1.0
                        + config
                            .conflict_recovery_max_clean_error_increase_ratio
                            .max(0.0));
                accepted = clean_mean_before.is_finite()
                    && clean_mean_after.is_finite()
                    && recovered_mean_after.is_finite()
                    && clean_mean_after <= allowed_clean_mean + 1e-12
                    && recovered_mean_after <= config.conflict_recovery_max_mean_reprojection_px;
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: geometry conflict recovery guard accepted={accepted} \
                         clean_mean={clean_mean_before:.6}->{clean_mean_after:.6} \
                         (allowed {allowed_clean_mean:.6}) recovered_mean={recovered_mean_after:.6}",
                    );
                }
                if accepted {
                    geometry_recovered_tracks = recovered.len();
                    geometry_recovered_observations =
                        recovered.iter().map(|track| track.observations.len()).sum();
                    geometry_recovery_pose_ba_applied = true;
                    ba_result = Some(result);
                }
            } else if sfm_debug_enabled() {
                eprintln!("sfm-debug: geometry conflict recovery BA failed; rolling back");
            }
            if !accepted {
                tracks.truncate(clean_track_count);
                poses = poses_before;
                track_point = track_point_before;
            }
        }
    }
    let geometry_recovery_seconds = geometry_recovery_started.elapsed().as_secs_f64();

    let mut post_refinement_registered_images = 0usize;
    if config.colmap_style_mapper && config.post_refinement_registration {
        post_refinement_registered_images = post_refinement_registration_pass(
            &cam,
            features,
            &tracks,
            config,
            &mut poses,
            &mut track_point,
        )
        .map_err(IncrementalSfmError::Ba)?;
        if post_refinement_registered_images > 0 {
            ba_result = Some(
                iterative_global_refinement(
                    &mut cam,
                    features,
                    &mut tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?,
            );
        }
    }
    let structureless_started = std::time::Instant::now();
    let structureless_registered_images = if config.colmap_style_mapper
        && config.structureless_registration
        && poses.iter().any(Option::is_none)
    {
        structureless_registration_rounds(
            &cam,
            features,
            pairwise,
            &mut tracks,
            config,
            &mut poses,
            &mut track_point,
        )
    } else {
        0
    };
    let structureless_seconds = structureless_started.elapsed().as_secs_f64();
    let final_refinement_seconds = final_refinement_started.elapsed().as_secs_f64();

    // ---- Assemble output tracks (only triangulated, registered observations) ----
    let assembly_started = std::time::Instant::now();
    let mut out_tracks = Vec::new();
    let mut reproj_sum = 0.0;
    let mut reproj_count = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(position) = track_point[track_id] else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, kp) in track {
            let Some(pose) = &poses[image] else { continue };
            let Some(pixel) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            observations.push((image, kp, pixel));
            if let Some(err) = reprojection_error_px(&cam, pose, &position, &pixel) {
                reproj_sum += err;
                reproj_count += 1;
            }
        }
        if observations.len() >= config.min_track_length {
            out_tracks.push(SfmTrack {
                position,
                observations,
            });
        }
    }

    let registered_images = poses.iter().filter(|p| p.is_some()).count();
    let mean_reprojection_px = if reproj_count > 0 {
        reproj_sum / reproj_count as f64
    } else {
        f64::NAN
    };
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-timing: total={:.3}s track_build={track_build_seconds:.3}s \
             seed_growth={seed_growth_seconds:.3}s final_refinement={final_refinement_seconds:.3}s \
             geometry_recovery={geometry_recovery_seconds:.3}s \
             structureless={structureless_seconds:.3}s assembly={:.3}s",
            sfm_started.elapsed().as_secs_f64(),
            assembly_started.elapsed().as_secs_f64(),
        );
    }

    Ok(IncrementalSfmResult {
        poses,
        tracks: out_tracks,
        track_build_stats,
        registered_images,
        post_refinement_registered_images,
        structureless_registered_images,
        geometry_recovered_tracks,
        geometry_recovered_observations,
        geometry_recovery_pose_ba_applied,
        mean_reprojection_px,
        ba_result,
        refined_camera: config.refine_intrinsics.then_some(cam),
        seed_image_i,
        seed_image_j,
        seed_match_count,
    })
}

/// Union-find over `(image, keypoint)` nodes joined by pairwise matches. Returns
/// the consistent tracks (no two keypoints from the same image) spanning at
/// least `min_track_length` distinct images.
#[cfg(test)]
fn build_tracks(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    build_tracks_with_stats(n_images, pairwise, min_track_length).0
}

#[cfg(test)]
fn build_tracks_with_stats(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> (Vec<Vec<(usize, usize)>>, TrackBuildStats) {
    let output = build_tracks_detailed(n_images, pairwise, min_track_length);
    (output.tracks, output.stats)
}

fn build_tracks_detailed(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    let _ = n_images;
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        ..TrackBuildStats::default()
    };
    // Map each observed (image, keypoint) to a dense node id.
    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let node_of = |image: usize,
                   kp: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>|
     -> usize {
        *node_id.entry((image, kp)).or_insert_with(|| {
            nodes.push((image, kp));
            nodes.len() - 1
        })
    };

    let mut parent: Vec<usize> = Vec::new();
    let ensure = |id: usize, parent: &mut Vec<usize>| {
        while parent.len() <= id {
            let next = parent.len();
            parent.push(next);
        }
    };

    for pair in pairwise {
        for &(ki, kj) in &pair.matches {
            let a = node_of(pair.image_i, ki, &mut node_id, &mut nodes);
            let b = node_of(pair.image_j, kj, &mut node_id, &mut nodes);
            ensure(a, &mut parent);
            ensure(b, &mut parent);
            union(&mut parent, a, b);
        }
    }

    // Group nodes by representative root.
    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, kp)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, kp));
    }
    stats.connected_components = groups.len();

    let mut tracks = Vec::new();
    let mut conflicting_components = Vec::new();
    for (_root, mut obs) in groups {
        // Reject tracks with conflicting observations (same image twice): such
        // a component merged two distinct points through a bad match chain.
        let mut images_seen: HashMap<usize, usize> = HashMap::new();
        let mut conflict = false;
        for &(image, _kp) in &obs {
            let count = images_seen.entry(image).or_insert(0);
            *count += 1;
            if *count > 1 {
                conflict = true;
                break;
            }
        }
        if conflict {
            stats.conflicting_components += 1;
            stats.conflicting_observations += obs.len();
            obs.sort_unstable();
            conflicting_components.push(obs);
            continue;
        }
        if images_seen.len() >= min_track_length {
            obs.sort_unstable();
            tracks.push(obs);
        }
    }
    // Deterministic track order (the grouping `HashMap` iterates in a random
    // order per run): a stable order makes landmark ids — and therefore the
    // whole incremental reconstruction — reproducible.
    tracks.sort_unstable();
    conflicting_components.sort_unstable();
    stats.retained_tracks = tracks.len();
    stats.retained_observations = tracks.iter().map(Vec::len).sum();
    TrackBuildOutput {
        tracks,
        conflicting_components,
        stats,
    }
}

/// M2 port: build feature tracks by routing through a
/// `visloc_vision::two_view::CorrespondenceGraph` instead of an ad hoc
/// union-find (COLMAP's own `CorrespondenceGraph`, ported in
/// `crates/vision/src/two_view/correspondence_graph.rs` — see that module's
/// doc for full citations). Every `pairwise` entry is added via
/// `CorrespondenceGraph::add_two_view_geometry`; this call site has no
/// per-pair `ConfigurationType` available (that M1 classification, when it
/// runs at all, is consumed upstream by the caller deciding which pairs make
/// it into `pairwise` in the first place — see
/// `examples/unordered_sfm_demo.rs`'s `verify_pairs` and the
/// `correspondence_graph` module doc's "Degenerate-pair policy" section), so
/// every edge is tagged with a placeholder [`ConfigurationType::Calibrated`]
/// that this function never reads back.
///
/// Tracks are then exactly COLMAP's connected components: for every
/// not-yet-visited `(image, keypoint)` observation, pull its **unbounded**
/// transitive closure (`extract_transitive_correspondences(.., ..,
/// usize::MAX)` — see that method's doc for why `usize::MAX` reproduces a
/// full connected component rather than a `num_transitivity`-bounded
/// neighbourhood) and apply the same same-image-conflict rejection and
/// `min_track_length` gate [`build_tracks_with_stats`] does. Because both algorithms
/// partition the exact same node set by the exact same edge set into
/// equivalence classes, and both sort observations within a track and tracks
/// against each other identically, this produces **byte-identical**
/// `Vec<Vec<(usize, usize)>>` output to [`build_tracks_with_stats`] on any input — the
/// M2 acceptance bar (`docs/colmap_port_plan.md`: "byte-identical tracks — a
/// refactor gate, not an accuracy claim"). See the
/// `graph_tracks_match_union_find_tracks_*` tests below.
fn build_tracks_via_graph(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut graph = CorrespondenceGraph::new();
    for (image_id, feature_set) in features.iter().enumerate() {
        graph.add_image(image_id, feature_set.keypoints.len());
    }

    // `CorrespondenceGraph::add_two_view_geometry` — faithfully to COLMAP's
    // own `THROW_CHECK(inserted)` — accepts a given unordered image pair only
    // *once* (see that method's doc). The legacy union-find track builder
    // has no such restriction: it just unions whatever `(image, keypoint)`
    // pairs every `PairwiseMatches` entry hands it, in either direction,
    // even if the same unordered pair appears more than once (e.g. a
    // pathological/test input, or two independently-verified match sets for
    // the same pair). To keep `build_tracks_via_graph` producing identical
    // tracks on *any* such input, pre-merge every `pairwise` entry into one
    // match list per unordered pair — normalizing direction to the pair's
    // canonical `(min, max)` order — before a single `add_two_view_geometry`
    // call per pair.
    let mut merged: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for pair in pairwise {
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        let entry = merged.entry(key).or_default();
        if pair.image_i <= pair.image_j {
            entry.extend(pair.matches.iter().copied());
        } else {
            entry.extend(pair.matches.iter().map(|&(a, b)| (b, a)));
        }
    }
    for (&(image_id1, image_id2), matches) in &merged {
        // Ignore ingest errors: a self-pair (`image_i == image_j`) is a
        // caller bug the legacy union-find path also has no defence against
        // (it would silently union a node with itself, a no-op); dropping it
        // here preserves the same "garbage in, best-effort out" behaviour
        // rather than panicking.
        let _ = graph.add_two_view_geometry(
            image_id1,
            image_id2,
            matches,
            ConfigurationType::Calibrated,
        );
    }
    graph.finalize();

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut tracks = Vec::new();
    for (image_id, feature_set) in features.iter().enumerate() {
        if !graph.exists_image(image_id) {
            continue; // dropped by finalize: never received a correspondence
        }
        for point2d_idx in 0..feature_set.keypoints.len() {
            if visited.contains(&(image_id, point2d_idx)) {
                continue;
            }
            if !graph.has_correspondences(image_id, point2d_idx) {
                visited.insert((image_id, point2d_idx));
                continue;
            }

            let closure =
                graph.extract_transitive_correspondences(image_id, point2d_idx, usize::MAX);
            let mut obs: Vec<(usize, usize)> = closure
                .iter()
                .map(|c| (c.image_id, c.point2d_idx))
                .collect();
            obs.push((image_id, point2d_idx));
            for &node in &obs {
                visited.insert(node);
            }

            // Same conflict rule as `build_tracks`: two keypoints from the
            // same image in one component means a bad match chain merged two
            // distinct points — drop the whole track.
            let mut images_seen: HashMap<usize, usize> = HashMap::new();
            let mut conflict = false;
            for &(image, _kp) in &obs {
                let count = images_seen.entry(image).or_insert(0);
                *count += 1;
                if *count > 1 {
                    conflict = true;
                    break;
                }
            }
            if conflict {
                continue;
            }
            if images_seen.len() >= min_track_length {
                obs.sort_unstable();
                tracks.push(obs);
            }
        }
    }
    tracks.sort_unstable();
    tracks
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Size of the largest connected component of the view graph — images joined by
/// a verified pair. This bounds how many images any single seed can ever reach,
/// so a seed that reaches a large fraction of it is well-connected rather than an
/// isolated local cluster of a few near-identical frames.
fn largest_connected_component(pairwise: &[PairwiseMatches], n_images: usize) -> usize {
    if n_images == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n_images).collect();
    for p in pairwise {
        union(&mut parent, p.image_i, p.image_j);
    }
    let mut count = vec![0usize; n_images];
    for i in 0..n_images {
        let r = find(&mut parent, i);
        count[r] += 1;
    }
    count.into_iter().max().unwrap_or(0)
}

/// Indices of verified pairs in descending match-count order, restricted to
/// those that clear `min_seed_matches`. These are the candidate seeds, strongest
/// first; [`grow_from_seed`] decides which one actually bootstraps the largest
/// reconstruction.
fn seed_candidate_order(pairwise: &[PairwiseMatches], config: &IncrementalSfmConfig) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pairwise.len())
        .filter(|&i| pairwise[i].matches.len() >= config.min_seed_matches)
        .filter(|&i| {
            let pair = &pairwise[i];
            let key = (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            );
            !config.excluded_seed_pairs.contains(&key)
        })
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pairwise[i].matches.len()));
    order
}

/// Recover one verified pair's two-view relative pose and place both images
/// (seed `i` at the world origin, `j` at the relative pose). Returns `true` only
/// if the pair bootstraps a well-conditioned baseline: enough of its inlier
/// correspondences triangulate under the shared parallax / cheirality /
/// reprojection gate. A low-parallax pair (e.g. two adjacent frames) is rejected
/// and `poses` is left untouched for `i` and `j`.
fn place_seed_pair(
    camera: &Camera,
    features: &[FeatureSet],
    pair: &PairwiseMatches,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
) -> bool {
    let estimator = RelativePoseEstimator::default();

    // Build correspondences, keeping the (kp_i, kp_j) map aligned so a
    // relative-pose inlier index maps back to the right keypoints.
    let mut corrs = Vec::with_capacity(pair.matches.len());
    let mut corr_kp = Vec::with_capacity(pair.matches.len());
    for &(ki, kj) in &pair.matches {
        let (Some(pi_xy), Some(pj_xy)) = (
            features[pair.image_i].keypoints.get(ki),
            features[pair.image_j].keypoints.get(kj),
        ) else {
            continue;
        };
        corrs.push(TwoViewCorrespondence::new(*pi_xy, *pj_xy));
        corr_kp.push((*pi_xy, *pj_xy));
    }
    let Some(relative) = estimator.estimate(&corrs, camera) else {
        return false;
    };
    if relative.inliers.len() < config.min_seed_matches {
        return false;
    }
    // Tentatively place: image i at the origin, image j at the relative.
    poses[pair.image_i] = Some(Pose::from_world_to_camera(
        nalgebra::UnitQuaternion::identity(),
        Vector3::zeros(),
    ));
    poses[pair.image_j] = Some(Pose::from_world_to_camera(
        relative.previous_to_current.rotation,
        relative.previous_to_current.translation,
    ));
    // Count inlier correspondences that triangulate to well-conditioned points.
    let mut well_triangulated = 0usize;
    for &inl in &relative.inliers {
        let (px_i, px_j) = corr_kp[inl];
        let obs = [(pair.image_i, px_i), (pair.image_j, px_j)];
        if triangulate_track(camera, poses, &obs, config).is_some() {
            well_triangulated += 1;
        }
    }
    if well_triangulated >= config.min_seed_matches {
        return true; // good baseline — keep these poses
    }
    // Low parallax: undo.
    poses[pair.image_i] = None;
    poses[pair.image_j] = None;
    false
}

/// Bootstrap from `seed_pair` and grow the reconstruction by repeatedly
/// registering the best next image, running the periodic global bundle
/// adjustment every `ba_every` registrations. Returns the per-image poses,
/// per-track points and the number of registered images — the reach the seed
/// selection compares across candidates. A seed that fails the baseline gate
/// yields zero registered images.
#[allow(clippy::type_complexity)]
fn grow_from_seed(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    obs_by_image: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    seed_pair: &PairwiseMatches,
) -> Result<(Vec<Option<Pose>>, Vec<Option<Point3<f64>>>, usize, Camera), IncrementalSfmError> {
    let grow_started = std::time::Instant::now();
    let mut select_seconds = 0.0;
    let mut pnp_seconds = 0.0;
    let mut triangulation_seconds = 0.0;
    let mut local_ba_seconds = 0.0;
    let mut global_refinement_seconds = 0.0;
    let mut pnp_attempts = 0usize;
    let mut local_ba_calls = 0usize;
    let mut global_refinement_calls = 0usize;
    let n_images = features.len();
    let mut poses: Vec<Option<Pose>> = vec![None; n_images];
    let mut track_point: Vec<Option<Point3<f64>>> = vec![None; tracks.len()];

    // Per-trial camera clone: the seed search grows several reconstructions over
    // the same shared `tracks`, so each trial co-evolves intrinsics on its own
    // copy (no cross-trial contamination). The winning trial's camera is returned.
    let mut cam = camera.clone();
    if !place_seed_pair(&cam, features, seed_pair, config, &mut poses) {
        return Ok((poses, track_point, 0, cam));
    }
    let started = std::time::Instant::now();
    triangulate_pending(&cam, features, tracks, &poses, config, &mut track_point);
    triangulation_seconds += started.elapsed().as_secs_f64();

    // `trials[i]` counts PnP attempts on image `i`. In the simple schedule one
    // failed attempt is permanent (the cap is 1); the COLMAP schedule retries up
    // to `max_registration_trials` across global-refinement boundaries.
    let max_trials = if config.colmap_style_mapper {
        config.max_registration_trials.max(1)
    } else {
        1
    };
    let mut trials: Vec<usize> = vec![0; n_images];
    let mut registrations_since_ba = 0usize;
    // COLMAP triggers a global refinement once the registered-image count has
    // grown by `global_ba_images_ratio` since the last one.
    let mut reg_at_last_global = poses.iter().filter(|p| p.is_some()).count();
    // COLMAP `IncrementalPipeline::ReconstructSubModel`'s do-while loop
    // (`controllers/incremental_pipeline.cc:519-629`) never gives up the first
    // time no image can be registered: when a full round finds nothing
    // (`!reg_next_success`), it runs one more `IterativeGlobalRefinement` and
    // tries again, only stopping once *two consecutive* rounds both find
    // nothing (`while (reg_next_success || prev_reg_next_success)`, line 629).
    // `stalled_once` is that same one-shot recovery. It matters because
    // `select_next_image` returning `None` is not always "structurally done" —
    // a track that lacked the 6th correspondence [`select_next_image`] needs
    // can gain one once [`growth_global_refinement`]'s retriangulation
    // completes a track that had ≥2 registered observers all along, just not
    // at a pair the on-the-fly [`triangulate_pending`] happened to accept
    // (BA can tighten those same views' poses enough, between one
    // registration and the next stall, to flip a marginal parallax/
    // reprojection gate that failed moments before). This is the M4 fix for
    // the path-dependence diagnosed in `docs/colmap_port_plan.md`'s "M3
    // results" (courtyard stuck at 13-14/38 even under exhaustive pair
    // coverage): the growth-ratio-triggered refinement above only fires while
    // registrations keep succeeding, so once growth truly stalls the ratio
    // can never trigger again and this loop broke immediately, leaving
    // whatever a completing refinement might have unlocked untried.
    //
    // Deliberately **not** ported: resetting `trials` on the stall, even
    // though it would let an already-trial-exhausted image be re-offered.
    // COLMAP's own `num_reg_trials` never resets either
    // (`incremental_mapper.cc:229`, incremented unconditionally on *every*
    // `RegisterNextImage` call, success or failure, for the reconstruction's
    // whole lifetime) — and here that persistence is load-bearing, not just
    // an unported nicety: with `filter_images` on, a resetting version can
    // livelock (register a weakly-supported image → `filter_images` demotes
    // it next stall → the reset makes it eligible again → it re-registers
    // identically → demoted again → …, forever, since each re-registration
    // looks like "progress" and would keep re-arming the recovery). Never
    // resetting bounds every image, demoted or not, to
    // `max_registration_trials` total lifetime attempts, so this cannot
    // cycle more than that many times before the image is excluded for good
    // — the same guarantee COLMAP's design gets from never resetting.
    let mut stalled_once = false;
    loop {
        let started = std::time::Instant::now();
        let selection = select_next_image(
            &cam,
            config.next_image_policy,
            features,
            obs_by_image,
            &poses,
            &trials,
            max_trials,
            &track_point,
        );
        select_seconds += started.elapsed().as_secs_f64();
        let Some((next_image, corrs)) = selection else {
            let n_reg = poses.iter().filter(|p| p.is_some()).count();
            // With image filtering disabled, a recovery refinement can only
            // unlock an unregistered image. Once reconstruction is complete it
            // duplicates the final iterative refinement below. Filtering is the
            // exception: even a complete model may need this round to demote a
            // weak pose, so preserve the recovery whenever `filter_images` is on.
            if n_reg == n_images && !config.filter_images {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: growth complete at {n_reg}/{n_images}; \
                         skipping redundant stall-recovery refinement",
                    );
                }
                break;
            }
            if config.colmap_style_mapper && !stalled_once {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: growth stalled at {n_reg}/{n_images} registered — \
                         forcing one stall-recovery refinement and retrying",
                    );
                }
                let started = std::time::Instant::now();
                growth_global_refinement(
                    &mut cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?;
                global_refinement_seconds += started.elapsed().as_secs_f64();
                global_refinement_calls += 1;
                reg_at_last_global = poses.iter().filter(|p| p.is_some()).count();
                if config.filter_images {
                    filter_images(&cam, features, tracks, config, &mut poses, &track_point);
                }
                stalled_once = true;
                continue;
            }
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: growth exhausted at {n_reg}/{n_images} registered \
                     (colmap_style_mapper={}, stalled_once={stalled_once})",
                    config.colmap_style_mapper,
                );
                for line in diagnose_unregistered_images(
                    obs_by_image,
                    &poses,
                    &trials,
                    max_trials,
                    &track_point,
                ) {
                    eprintln!("sfm-debug: {line}");
                }
            }
            break;
        };
        trials[next_image] += 1;

        // P3P (Grunert) is the default minimal solver — well-posed on coplanar
        // façades where the linear DLT degenerates. Both share the Gauss-Newton
        // refiner and the config reprojection gate.
        let started = std::time::Instant::now();
        let report = match config.pnp_solver {
            PnpSolver::P3p => PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: 128,
                reprojection_threshold: config.max_reprojection_error_px,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
            }
            .estimate(&corrs, &cam),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: config.max_reprojection_error_px,
                ..PnPRansac::default()
            }
            .estimate(&corrs, &cam),
        };
        pnp_seconds += started.elapsed().as_secs_f64();
        pnp_attempts += 1;
        let attempt_inliers = report.as_ref().map(|r| r.inliers.len());
        let Some(report) = report.filter(|r| r.inliers.len() >= config.min_pnp_inliers) else {
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: PnP attempt #{} on image {next_image} failed \
                     ({} corrs -> {} inliers, need >={})",
                    trials[next_image],
                    corrs.len(),
                    attempt_inliers.map_or("none".to_string(), |n| n.to_string()),
                    config.min_pnp_inliers,
                );
            }
            continue; // registration failed this attempt (may be retried)
        };
        // Genuine progress — a future stall earns its own one-shot recovery
        // (see `stalled_once`'s module-level doc above).
        stalled_once = false;
        poses[next_image] = Some(report.pose);
        let started = std::time::Instant::now();
        triangulate_pending(&cam, features, tracks, &poses, config, &mut track_point);
        triangulation_seconds += started.elapsed().as_secs_f64();

        if config.colmap_style_mapper {
            // COLMAP `AdjustLocalBundle`: tighten the new image + its covisible
            // neighbourhood after every registration.
            let started = std::time::Instant::now();
            adjust_local_bundle(
                &cam,
                features,
                tracks,
                config,
                &mut poses,
                &mut track_point,
                next_image,
            )
            .map_err(IncrementalSfmError::Ba)?;
            local_ba_seconds += started.elapsed().as_secs_f64();
            local_ba_calls += 1;

            // Growth-ratio global refinement (COLMAP `IterativeGlobalRefinement`).
            // During the seed search `tracks` is shared read-only across trials,
            // so the in-growth refinement only re-triangulates + re-BAs (touching
            // this trial's own poses/points); the track-membership *filter* that
            // would mutate the shared tracks is deferred to the final refinement,
            // after a seed has been committed. The BA's Huber kernel keeps
            // outliers down-weighted in the meantime.
            let n_reg = poses.iter().filter(|p| p.is_some()).count();
            if n_reg as f64 >= reg_at_last_global as f64 * config.global_ba_images_ratio {
                let started = std::time::Instant::now();
                growth_global_refinement(
                    &mut cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?;
                global_refinement_seconds += started.elapsed().as_secs_f64();
                global_refinement_calls += 1;
                reg_at_last_global = n_reg;
                // Structure changed — give previously-failed images a fresh shot
                // by resetting their trial counters (COLMAP retries on change).
                for (i, t) in trials.iter_mut().enumerate() {
                    if poses[i].is_none() {
                        *t = 0;
                    }
                }
                // COLMAP `FilterImages`: de-register images whose pose lost support
                // after the global solve. Done AFTER the retry reset so a filtered
                // image keeps its accumulated trial count (it is re-registered at
                // most `max_registration_trials` times, not indefinitely).
                if config.filter_images {
                    filter_images(&cam, features, tracks, config, &mut poses, &track_point);
                }
            }
        } else {
            registrations_since_ba += 1;
            if config.ba_every > 0 && registrations_since_ba >= config.ba_every {
                // The simple schedule keeps intrinsics fixed during growth (refine
                // is a colmap-style / final-solve concern); refined slot is None.
                run_bundle_adjustment(
                    &cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                    false,
                )
                .map_err(IncrementalSfmError::Ba)?;
                registrations_since_ba = 0;
            }
        }
    }

    let registered = poses.iter().filter(|p| p.is_some()).count();
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-timing: grow total={:.3}s select={select_seconds:.3}s \
             pnp={pnp_seconds:.3}s/{pnp_attempts} triangulate={triangulation_seconds:.3}s \
             local_ba={local_ba_seconds:.3}s/{local_ba_calls} \
             global_refinement={global_refinement_seconds:.3}s/{global_refinement_calls}",
            grow_started.elapsed().as_secs_f64(),
        );
    }
    Ok((poses, track_point, registered, cam))
}

/// One bounded registration sweep after final global refinement. Unlike the
/// growth loop, this cannot cycle: every missing image receives at most one
/// attempt, and the caller invokes the function at most once.
fn post_refinement_registration_pass(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<usize, BaError> {
    let mut obs_by_image: Vec<Vec<(usize, usize)>> = vec![Vec::new(); features.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, kp) in track {
            obs_by_image[image].push((kp, track_id));
        }
    }

    let mut trials = vec![0usize; features.len()];
    let mut registered = 0usize;
    while let Some((image, corrs)) = select_next_image(
        camera,
        config.next_image_policy,
        features,
        &obs_by_image,
        poses,
        &trials,
        1,
        track_point,
    ) {
        trials[image] = 1;
        let report = match config.pnp_solver {
            PnpSolver::P3p => PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: 128,
                reprojection_threshold: config.max_reprojection_error_px,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
            }
            .estimate(&corrs, camera),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: config.max_reprojection_error_px,
                ..PnPRansac::default()
            }
            .estimate(&corrs, camera),
        };
        let attempt_inliers = report.as_ref().map(|r| r.inliers.len());
        let Some(report) = report.filter(|r| r.inliers.len() >= config.min_pnp_inliers) else {
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: post-refinement PnP on image {image} failed \
                     ({} corrs -> {} inliers, need >={})",
                    corrs.len(),
                    attempt_inliers.map_or("none".to_string(), |n| n.to_string()),
                    config.min_pnp_inliers,
                );
            }
            continue;
        };

        poses[image] = Some(report.pose);
        triangulate_pending(camera, features, tracks, poses, config, track_point);
        adjust_local_bundle(camera, features, tracks, config, poses, track_point, image)?;
        registered += 1;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: post-refinement registered image {image} \
                 ({} corrs, {} inliers)",
                corrs.len(),
                report.inliers.len(),
            );
        }
    }
    Ok(registered)
}

#[derive(Debug, Clone)]
struct StructurelessConstraint {
    neighbor: usize,
    neighbor_center: Point3<f64>,
    missing_rotation: UnitQuaternion<f64>,
    center_direction: Vector3<f64>,
    weight: f64,
}

#[derive(Debug, Clone)]
struct StructurelessPoseProposal {
    pose: Pose,
    neighbor_spread: f64,
    line_error_ratio: f64,
    consensus_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum StructurelessRejection {
    TooFewNeighbors {
        found: usize,
        required: usize,
    },
    RotationDisagreement {
        max_deg: f64,
        allowed_deg: f64,
    },
    WeakCenterGeometry {
        max_angle_deg: f64,
        spread: f64,
    },
    NoCenterConsensus {
        rotation_consensus: usize,
    },
    SingularCenterFit,
    DirectionSign {
        neighbor: usize,
        along_ratio: f64,
        allowed: f64,
    },
    CenterLineResidual {
        ratio: f64,
        allowed: f64,
    },
}

/// Fit the missing camera centre to directed lines originating at registered
/// neighbour centres. Each line direction comes from an independently
/// recovered essential pose, while its origin carries the current model's
/// monocular scale. This is the scale-bearing part of structure-less recovery:
/// one line is deliberately under-constrained and is always rejected.
fn solve_structureless_pose(
    constraints: &[StructurelessConstraint],
    config: &IncrementalSfmConfig,
) -> Result<StructurelessPoseProposal, StructurelessRejection> {
    let required_neighbors = config.structureless_min_neighbors.max(2);
    if constraints.len() < required_neighbors {
        return Err(StructurelessRejection::TooFewNeighbors {
            found: constraints.len(),
            required: required_neighbors,
        });
    }

    let max_rotation_rad = config
        .structureless_max_rotation_disagreement_deg
        .max(0.0)
        .to_radians();
    // A single bad essential edge must not veto an otherwise coherent set.
    // Enumerate every rotation as a deterministic consensus centre and keep
    // the largest, then highest-support, <=threshold subset.
    let mut consensus_indices = Vec::new();
    let mut consensus_weight = -1.0f64;
    let mut rotation_reference_index = None;
    for (reference_index, reference) in constraints.iter().enumerate() {
        let candidate: Vec<usize> = constraints
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| {
                ((reference.missing_rotation.inverse() * constraint.missing_rotation).angle()
                    <= max_rotation_rad)
                    .then_some(index)
            })
            .collect();
        let weight: f64 = candidate
            .iter()
            .map(|&index| constraints[index].weight)
            .sum();
        if candidate.len() > consensus_indices.len()
            || (candidate.len() == consensus_indices.len() && weight > consensus_weight)
            || (candidate.len() == consensus_indices.len()
                && weight.to_bits() == consensus_weight.to_bits()
                && candidate.first().copied().unwrap_or(reference_index)
                    < consensus_indices.first().copied().unwrap_or(usize::MAX))
        {
            consensus_indices = candidate;
            consensus_weight = weight;
            rotation_reference_index = Some(reference_index);
        }
    }
    if consensus_indices.len() < required_neighbors {
        let strongest = &constraints[0];
        let max_rotation_disagreement = constraints
            .iter()
            .map(|constraint| {
                (strongest.missing_rotation.inverse() * constraint.missing_rotation).angle()
            })
            .fold(0.0f64, f64::max);
        return Err(StructurelessRejection::RotationDisagreement {
            max_deg: max_rotation_disagreement.to_degrees(),
            allowed_deg: max_rotation_rad.to_degrees(),
        });
    }
    // Preserve the actual consensus centre. Choosing the strongest edge after
    // finding the set is not equivalent: two members can each lie within the
    // threshold of the centre yet be almost 2x the threshold apart.
    let reference_index = rotation_reference_index.expect("rotation consensus has a centre");
    let reference = &constraints[reference_index];

    let min_intersection_angle = config
        .structureless_min_intersection_angle_deg
        .max(0.0)
        .to_radians();
    let mut max_intersection_angle = 0.0f64;
    let mut rotation_consensus_spread = 0.0f64;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        let a = &constraints[a_index];
        for &b_index in consensus_indices.iter().skip(position + 1) {
            let b = &constraints[b_index];
            let cosine = a
                .center_direction
                .dot(&b.center_direction)
                .abs()
                .clamp(0.0, 1.0);
            max_intersection_angle = max_intersection_angle.max(cosine.acos());
            rotation_consensus_spread =
                rotation_consensus_spread.max((a.neighbor_center - b.neighbor_center).norm());
        }
    }
    if max_intersection_angle < min_intersection_angle || rotation_consensus_spread <= 1e-9 {
        return Err(StructurelessRejection::WeakCenterGeometry {
            max_angle_deg: max_intersection_angle.to_degrees(),
            spread: rotation_consensus_spread,
        });
    }

    let identity = Matrix3::identity();
    let fit_center = |indices: &[usize]| -> Option<Point3<f64>> {
        let mut normal = Matrix3::zeros();
        let mut rhs = Vector3::zeros();
        for &index in indices {
            let constraint = &constraints[index];
            let direction = constraint.center_direction.try_normalize(1e-12)?;
            let weight = constraint.weight.max(1.0);
            let projector = identity - direction * direction.transpose();
            normal += projector * weight;
            rhs += projector * constraint.neighbor_center.coords * weight;
        }
        Some(Point3::from(normal.try_inverse()? * rhs))
    };

    // Translation directions need their own robust consensus: agreeing
    // rotations do not imply that every essential decomposition has a reliable
    // baseline direction. Seed from every sufficiently non-parallel line pair,
    // score all rotation-consensus lines, then refit the largest 3+ set.
    let max_line_ratio = config.structureless_max_center_line_error_ratio.max(0.0);
    let mut center_consensus = Vec::new();
    let mut center_consensus_weight = -1.0f64;
    let mut center_consensus_error = f64::INFINITY;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        for &b_index in consensus_indices.iter().skip(position + 1) {
            let a = &constraints[a_index];
            let b = &constraints[b_index];
            let angle = a
                .center_direction
                .dot(&b.center_direction)
                .abs()
                .clamp(0.0, 1.0)
                .acos();
            if angle < min_intersection_angle {
                continue;
            }
            let Some(candidate_center) = fit_center(&[a_index, b_index]) else {
                continue;
            };
            let mut inliers = Vec::new();
            let mut squared_error = 0.0;
            let mut weight = 0.0;
            for &index in &consensus_indices {
                let constraint = &constraints[index];
                let displacement = candidate_center - constraint.neighbor_center;
                let along_ratio =
                    displacement.dot(&constraint.center_direction) / rotation_consensus_spread;
                let perpendicular = displacement
                    - constraint.center_direction * displacement.dot(&constraint.center_direction);
                let line_ratio = perpendicular.norm() / rotation_consensus_spread;
                if along_ratio >= config.structureless_min_forward_ratio
                    && line_ratio <= max_line_ratio
                {
                    inliers.push(index);
                    let edge_weight = constraint.weight.max(1.0);
                    squared_error += edge_weight * line_ratio * line_ratio;
                    weight += edge_weight;
                }
            }
            if inliers.len() < required_neighbors {
                continue;
            }
            let rms_error = (squared_error / weight.max(1.0)).sqrt();
            if inliers.len() > center_consensus.len()
                || (inliers.len() == center_consensus.len() && weight > center_consensus_weight)
                || (inliers.len() == center_consensus.len()
                    && weight.to_bits() == center_consensus_weight.to_bits()
                    && rms_error < center_consensus_error)
            {
                center_consensus = inliers;
                center_consensus_weight = weight;
                center_consensus_error = rms_error;
            }
        }
    }
    if center_consensus.len() < required_neighbors {
        return Err(StructurelessRejection::NoCenterConsensus {
            rotation_consensus: consensus_indices.len(),
        });
    }
    consensus_indices = center_consensus;
    // A weighted least-squares refit can move slightly outside the inlier set
    // that generated the winning two-line hypothesis. Reclassify after every
    // refit and discard only the inconsistent lines instead of allowing one
    // marginal edge to veto an otherwise valid 3+ neighbour consensus.
    // Removal is monotonic, so this converges in at most N iterations.
    let center = loop {
        let fitted =
            fit_center(&consensus_indices).ok_or(StructurelessRejection::SingularCenterFit)?;
        let retained: Vec<usize> = consensus_indices
            .iter()
            .copied()
            .filter(|&index| {
                let constraint = &constraints[index];
                let displacement = fitted - constraint.neighbor_center;
                let along_ratio =
                    displacement.dot(&constraint.center_direction) / rotation_consensus_spread;
                let perpendicular = displacement
                    - constraint.center_direction * displacement.dot(&constraint.center_direction);
                let line_ratio = perpendicular.norm() / rotation_consensus_spread;
                along_ratio >= config.structureless_min_forward_ratio
                    && line_ratio <= max_line_ratio
            })
            .collect();
        if retained.len() < required_neighbors {
            return Err(StructurelessRejection::NoCenterConsensus {
                rotation_consensus: consensus_indices.len(),
            });
        }
        if retained.len() == consensus_indices.len() {
            break fitted;
        }
        consensus_indices = retained;
    };
    let mut selected_neighbor_spread = 0.0f64;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        for &b_index in consensus_indices.iter().skip(position + 1) {
            selected_neighbor_spread = selected_neighbor_spread.max(
                (constraints[a_index].neighbor_center - constraints[b_index].neighbor_center)
                    .norm(),
            );
        }
    }
    if selected_neighbor_spread <= 1e-9 {
        return Err(StructurelessRejection::SingularCenterFit);
    }
    // Use the same rotation-consensus span used while scoring RANSAC centre
    // hypotheses. Switching to the smaller selected-subset span after refit
    // would make an inlier fail a stricter, inconsistent normalized gate.
    let neighbor_spread = rotation_consensus_spread;

    let mut weighted_squared_error = 0.0;
    let mut weight_sum = 0.0;
    for &index in &consensus_indices {
        let constraint = &constraints[index];
        let displacement = center - constraint.neighbor_center;
        // Essential decomposition resolves the sign through cheirality. A
        // negative line parameter means the multi-neighbour fit contradicts
        // that independent two-view geometry.
        let along = displacement.dot(&constraint.center_direction);
        let along_ratio = along / neighbor_spread;
        if along_ratio < config.structureless_min_forward_ratio {
            return Err(StructurelessRejection::DirectionSign {
                neighbor: constraint.neighbor,
                along_ratio,
                allowed: config.structureless_min_forward_ratio,
            });
        }
        let perpendicular = displacement
            - constraint.center_direction * displacement.dot(&constraint.center_direction);
        let weight = constraint.weight.max(1.0);
        weighted_squared_error += weight * perpendicular.norm_squared();
        weight_sum += weight;
    }
    let rms_line_error = (weighted_squared_error / weight_sum.max(1.0)).sqrt();
    let line_error_ratio = rms_line_error / neighbor_spread;
    if !line_error_ratio.is_finite()
        || line_error_ratio > config.structureless_max_center_line_error_ratio.max(0.0)
    {
        return Err(StructurelessRejection::CenterLineResidual {
            ratio: line_error_ratio,
            allowed: config.structureless_max_center_line_error_ratio.max(0.0),
        });
    }

    let rotation = reference.missing_rotation;
    let translation = -rotation.transform_vector(&center.coords);
    Ok(StructurelessPoseProposal {
        pose: Pose::from_world_to_camera(rotation, translation),
        neighbor_spread,
        line_error_ratio,
        consensus_indices,
    })
}

fn estimate_structureless_constraints(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    missing: usize,
    config: &IncrementalSfmConfig,
) -> Vec<StructurelessConstraint> {
    let estimator = RelativePoseEstimator::default();
    let mut constraints = Vec::new();
    for pair in pairwise {
        let (neighbor, invert) = if pair.image_j == missing && poses[pair.image_i].is_some() {
            (pair.image_i, false)
        } else if pair.image_i == missing && poses[pair.image_j].is_some() {
            (pair.image_j, true)
        } else {
            continue;
        };
        let Some(neighbor_pose) = poses[neighbor].as_ref() else {
            continue;
        };
        let mut correspondences = Vec::with_capacity(pair.matches.len());
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (Some(pixel_i), Some(pixel_j)) = (
                features[pair.image_i].keypoints.get(keypoint_i),
                features[pair.image_j].keypoints.get(keypoint_j),
            ) else {
                continue;
            };
            correspondences.push(TwoViewCorrespondence::new(*pixel_i, *pixel_j));
        }
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            continue;
        };
        if relative.inliers.len() < config.structureless_min_pair_inliers {
            continue;
        }
        let neighbor_to_missing = if invert {
            relative.previous_to_current.inverse()
        } else {
            relative.previous_to_current
        };
        let missing_rotation =
            neighbor_to_missing.rotation * neighbor_pose.world_to_camera.rotation;
        let Some(center_direction) = (-missing_rotation
            .inverse()
            .transform_vector(&neighbor_to_missing.translation))
        .try_normalize(1e-12) else {
            continue;
        };
        constraints.push(StructurelessConstraint {
            neighbor,
            neighbor_center: neighbor_pose.camera_center_world(),
            missing_rotation,
            center_direction,
            weight: relative.inliers.len() as f64,
        });
    }
    constraints.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.neighbor.cmp(&b.neighbor))
    });
    constraints
}

fn mean_reprojection_for_registered_mask(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    registered_mask: &[bool],
    point_mask: &[bool],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        if !point_mask.get(track_id).copied().unwrap_or(false) {
            continue;
        }
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        for &(image, keypoint) in track {
            if !registered_mask.get(image).copied().unwrap_or(false) {
                continue;
            }
            let (Some(pose), Some(pixel)) = (
                poses.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint)),
            ) else {
                continue;
            };
            if let Some(error) = reprojection_error_px(camera, pose, point, pixel) {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn supported_tracks_for_image(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    image: usize,
    max_error: f64,
) -> (usize, f64) {
    let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
        return (0, f64::NAN);
    };
    let mut count = 0usize;
    let mut sum = 0.0;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        let Some((_, keypoint)) = track.iter().find(|(track_image, _)| *track_image == image)
        else {
            continue;
        };
        let Some(pixel) = features[image].keypoints.get(*keypoint) else {
            continue;
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            continue;
        };
        if error <= max_error {
            count += 1;
            sum += error;
        }
    }
    if count == 0 {
        (0, f64::NAN)
    } else {
        (count, sum / count as f64)
    }
}

#[derive(Debug, Clone, Copy)]
struct StructurelessPoseConsistency {
    accepted: bool,
    max_rotation_deg: f64,
    min_forward_ratio: f64,
    line_error_ratio: f64,
}

#[cfg(test)]
fn interpolate_structureless_pose(from: &Pose, to: &Pose, alpha: f64) -> Pose {
    interpolate_structureless_pose_components(from, to, alpha, alpha)
}

fn interpolate_structureless_pose_components(
    from: &Pose,
    to: &Pose,
    rotation_alpha: f64,
    center_alpha: f64,
) -> Pose {
    let rotation_alpha = rotation_alpha.clamp(0.0, 1.0);
    let center_alpha = center_alpha.clamp(0.0, 1.0);
    if rotation_alpha <= 0.0 && center_alpha <= 0.0 {
        return from.clone();
    }
    if rotation_alpha >= 1.0 && center_alpha >= 1.0 {
        return to.clone();
    }
    let rotation = from
        .world_to_camera
        .rotation
        .slerp(&to.world_to_camera.rotation, rotation_alpha);
    let from_center = from.camera_center_world();
    let to_center = to.camera_center_world();
    let center =
        Point3::from(from_center.coords * (1.0 - center_alpha) + to_center.coords * center_alpha);
    let translation = -rotation.transform_vector(&center.coords);
    Pose::from_world_to_camera(rotation, translation)
}

fn structureless_pose_consistency(
    pose: &Pose,
    constraints: &[StructurelessConstraint],
    proposal: &StructurelessPoseProposal,
    config: &IncrementalSfmConfig,
) -> StructurelessPoseConsistency {
    let center = pose.camera_center_world();
    let max_rotation = config
        .structureless_max_rotation_disagreement_deg
        .max(0.0)
        .to_radians();
    let mut weighted_squared_error = 0.0;
    let mut weight_sum = 0.0;
    let mut max_rotation_seen = 0.0f64;
    let mut min_forward_seen = f64::INFINITY;
    for &index in &proposal.consensus_indices {
        let constraint = &constraints[index];
        let rotation_error =
            (constraint.missing_rotation.inverse() * pose.world_to_camera.rotation).angle();
        max_rotation_seen = max_rotation_seen.max(rotation_error);
        let displacement = center - constraint.neighbor_center;
        let forward_ratio =
            displacement.dot(&constraint.center_direction) / proposal.neighbor_spread;
        min_forward_seen = min_forward_seen.min(forward_ratio);
        let perpendicular = displacement
            - constraint.center_direction * displacement.dot(&constraint.center_direction);
        let weight = constraint.weight.max(1.0);
        weighted_squared_error += weight * perpendicular.norm_squared();
        weight_sum += weight;
    }
    let ratio =
        (weighted_squared_error / weight_sum.max(1.0)).sqrt() / proposal.neighbor_spread.max(1e-12);
    StructurelessPoseConsistency {
        accepted: max_rotation_seen <= max_rotation
            && min_forward_seen >= config.structureless_min_forward_ratio
            && ratio.is_finite()
            && ratio <= config.structureless_max_center_line_error_ratio.max(0.0),
        max_rotation_deg: max_rotation_seen.to_degrees(),
        min_forward_ratio: min_forward_seen,
        line_error_ratio: ratio,
    }
}

/// Build an independent local submap from verified pairwise edges that were
/// not retained by the global union-find tracks. Observations already owned by
/// a global 3D track are never duplicated. Each new point must be seen by the
/// missing image and the configured number of registered consensus neighbours,
/// triangulate with sufficient parallax, and reproject within the initialization
/// gate in every contributing view.
fn build_structureless_local_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    poses: &[Option<Pose>],
    missing: usize,
    constraints: &[StructurelessConstraint],
    proposal: &StructurelessPoseProposal,
    config: &IncrementalSfmConfig,
) -> Vec<(Vec<(usize, usize)>, Point3<f64>)> {
    let allowed_neighbors: HashSet<usize> = proposal
        .consensus_indices
        .iter()
        .map(|&index| constraints[index].neighbor)
        .collect();
    let occupied: HashSet<(usize, usize)> = tracks
        .iter()
        .enumerate()
        .filter(|(track_id, _)| track_point.get(*track_id).is_some_and(Option::is_some))
        .flat_map(|(_, track)| track.iter().copied())
        .collect();
    let mut by_missing_keypoint: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for pair in pairwise {
        let (neighbor, missing_first) = if pair.image_i == missing
            && allowed_neighbors.contains(&pair.image_j)
            && poses[pair.image_j].is_some()
        {
            (pair.image_j, true)
        } else if pair.image_j == missing
            && allowed_neighbors.contains(&pair.image_i)
            && poses[pair.image_i].is_some()
        {
            (pair.image_i, false)
        } else {
            continue;
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (missing_keypoint, neighbor_keypoint) = if missing_first {
                (keypoint_i, keypoint_j)
            } else {
                (keypoint_j, keypoint_i)
            };
            if features[missing].keypoints.get(missing_keypoint).is_none()
                || features[neighbor]
                    .keypoints
                    .get(neighbor_keypoint)
                    .is_none()
                || occupied.contains(&(missing, missing_keypoint))
                || occupied.contains(&(neighbor, neighbor_keypoint))
            {
                continue;
            }
            by_missing_keypoint
                .entry(missing_keypoint)
                .or_default()
                .push((neighbor, neighbor_keypoint));
        }
    }

    let mut missing_keypoints: Vec<usize> = by_missing_keypoint.keys().copied().collect();
    missing_keypoints.sort_unstable();
    let mut local_tracks = Vec::new();
    let mut claimed_observations = HashSet::new();
    for missing_keypoint in missing_keypoints {
        let mut neighbors = by_missing_keypoint.remove(&missing_keypoint).unwrap();
        neighbors.sort_unstable();
        neighbors.dedup_by_key(|observation| observation.0);
        let required_registered_views = config
            .structureless_min_local_track_views
            .max(2)
            .saturating_sub(1);
        if neighbors.len() < required_registered_views {
            continue;
        }
        let mut observations = vec![(missing, missing_keypoint)];
        observations.extend(neighbors);
        observations.sort_unstable();
        if observations
            .iter()
            .any(|observation| claimed_observations.contains(observation))
        {
            continue;
        }
        let pixels: Vec<(usize, Point2<f64>)> = observations
            .iter()
            .map(|&(image, keypoint)| (image, features[image].keypoints[keypoint]))
            .collect();
        let Some(point) = triangulate_track(camera, poses, &pixels, config) else {
            continue;
        };
        let mut valid = true;
        for &(image, keypoint) in &observations {
            let Some(error) = reprojection_error_px(
                camera,
                poses[image].as_ref().unwrap(),
                &point,
                &features[image].keypoints[keypoint],
            ) else {
                valid = false;
                break;
            };
            // This is an initialization gate only. The point is subsequently
            // refined in the fixed-pose local submap and must still clear the
            // stricter structure-less admission error below.
            if error > config.max_reprojection_error_px {
                valid = false;
                break;
            }
        }
        if valid {
            claimed_observations.extend(observations.iter().copied());
            local_tracks.push((observations, point));
            if local_tracks.len() >= config.structureless_max_local_tracks.max(1) {
                break;
            }
        }
    }
    local_tracks
}

/// Run [`structureless_registration_pass`] repeatedly until a round registers
/// nothing, the budget [`IncrementalSfmConfig::structureless_max_rounds`] is
/// spent, or every image is registered. Each round scans in the same fixed
/// ascending image order, so the loop is deterministic; images registered by
/// earlier rounds act as neighbours for later ones, which is what lets an
/// island chain inward through its bridge even when the bridge's index is
/// higher than the images it unlocks.
fn structureless_registration_rounds(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut Vec<Option<Point3<f64>>>,
) -> usize {
    let mut total = 0usize;
    let max_rounds = config.structureless_max_rounds.max(1);
    for round in 0..max_rounds {
        if !poses.iter().any(Option::is_none) {
            break;
        }
        let registered = structureless_registration_pass(
            camera,
            features,
            pairwise,
            tracks,
            config,
            poses,
            track_point,
        );
        if sfm_debug_enabled() && registered > 0 {
            eprintln!("sfm-debug: structure-less round {round} registered {registered} image(s)");
        }
        total += registered;
        if registered == 0 {
            break;
        }
    }
    total
}

/// One bounded multi-neighbour recovery sweep. Each missing image is attempted
/// at most once. Failed geometry, local BA, or admission gates restore the
/// complete pose/point state byte-for-byte before moving on.
fn structureless_registration_pass(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut Vec<Option<Point3<f64>>>,
) -> usize {
    let missing_images: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.is_none().then_some(image))
        .collect();
    let mut registered = 0usize;
    for image in missing_images {
        let constraints =
            estimate_structureless_constraints(camera, features, pairwise, poses, image, config);
        let proposal = match solve_structureless_pose(&constraints, config) {
            Ok(proposal) => proposal,
            Err(reason) => {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: structure-less image {image} rejected before insertion \
                         ({} registered relative neighbours): {reason:?}",
                        constraints.len(),
                    );
                }
                continue;
            }
        };
        let tracks_before = tracks.clone();
        let tracks_before_len = tracks_before.len();
        let poses_before = poses.to_vec();
        let points_before = track_point.to_vec();
        let registered_mask: Vec<bool> = poses_before.iter().map(Option::is_some).collect();
        let clean_point_mask: Vec<bool> = points_before.iter().map(Option::is_some).collect();
        let clean_mean_before = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            &poses_before,
            &points_before,
            &registered_mask,
            &clean_point_mask,
        );
        poses[image] = Some(proposal.pose.clone());
        let local_tracks = build_structureless_local_tracks(
            camera,
            features,
            pairwise,
            tracks,
            track_point,
            poses,
            image,
            &constraints,
            &proposal,
            config,
        );
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: structure-less image {image} synthesized {} independent local tracks",
                local_tracks.len()
            );
        }
        let local_observations: HashSet<(usize, usize)> = local_tracks
            .iter()
            .flat_map(|(track, _)| track.iter().copied())
            .collect();
        for (track_id, track) in tracks.iter_mut().enumerate().take(tracks_before_len) {
            if track_point[track_id].is_none() {
                track.retain(|observation| !local_observations.contains(observation));
            }
        }
        for (track, point) in local_tracks {
            tracks.push(track);
            track_point.push(Some(point));
        }
        triangulate_pending(camera, features, tracks, poses, config, track_point);
        let proposal_poses = poses.to_vec();
        let proposal_points = track_point.to_vec();
        let (proposal_support, proposal_image_mean) = supported_tracks_for_image(
            camera,
            features,
            tracks,
            &proposal_poses,
            &proposal_points,
            image,
            config.max_reprojection_error_px,
        );
        let proposal_clean_mean = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            &proposal_poses,
            &proposal_points,
            &registered_mask,
            &clean_point_mask,
        );
        let proposal_consistency = proposal_poses[image]
            .as_ref()
            .map(|pose| structureless_pose_consistency(pose, &constraints, &proposal, config));
        // A structure-less proposal is already tied to the registered map by
        // several independently estimated relative poses.  Moving its
        // neighbours in the ordinary growth local-BA window can trade that
        // scale-bearing consensus for a lower pixel residual (MH_05 image 86
        // exposed exactly that failure).  Refine only the recovered pose and
        // its incident landmarks; every previously registered observer stays
        // fixed and therefore acts as the local-submap alignment boundary.
        let mut structureless_variable = HashSet::new();
        structureless_variable.insert(image);
        let local_result = bundle_adjust_local(
            camera,
            features,
            tracks,
            config,
            poses,
            track_point,
            &structureless_variable,
        );
        let refined_poses = poses.to_vec();
        let allowed_clean_mean = clean_mean_before
            * (1.0 + config.structureless_max_clean_error_increase_ratio.max(0.0));
        let (mut support, mut image_mean) = supported_tracks_for_image(
            camera,
            features,
            tracks,
            poses,
            track_point,
            image,
            config.max_reprojection_error_px,
        );
        let mut clean_mean_after = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            poses,
            track_point,
            &registered_mask,
            &clean_point_mask,
        );
        let mut pose_consistency = poses[image]
            .as_ref()
            .map(|pose| structureless_pose_consistency(pose, &constraints, &proposal, config));
        let local_ok = local_result.is_ok();
        let mut support_ok = support >= config.structureless_min_support_tracks;
        let mut image_error_ok =
            image_mean.is_finite() && image_mean <= config.structureless_max_reprojection_error_px;
        let mut clean_ok = clean_mean_before.is_finite()
            && clean_mean_after.is_finite()
            && clean_mean_after <= allowed_clean_mean + 1e-12;
        let mut geometry_ok = pose_consistency.is_some_and(|diagnostic| diagnostic.accepted);
        let mut accepted = local_ok && support_ok && image_error_ok && clean_ok && geometry_ok;
        let mut trust_region_alpha = None;

        // The unconstrained local BA can cross the independently measured
        // relative-geometry boundary while greatly improving reprojection.
        // Search back along the camera part of that BA update. For each pose
        // inside the relative-geometry feasible region, re-solve only the new
        // landmarks against fixed cameras, then commit the largest step that
        // satisfies every admission gate. This is a bounded deterministic
        // local-submap projection, not a relaxed threshold.
        if local_ok && !accepted {
            let proposal_pose = proposal_poses[image].as_ref().unwrap();
            let refined_pose = refined_poses[image].as_ref().unwrap();
            let mut trust_candidates = Vec::with_capacity(400);
            for rotation_step in (0..20).rev() {
                for center_step in (0..20).rev() {
                    trust_candidates.push((rotation_step as f64 / 20.0, center_step as f64 / 20.0));
                }
            }
            let mut candidate_index = 0usize;
            let mut best_near_candidate: Option<(f64, f64, f64)> = None;
            let mut fine_candidates_enqueued = false;
            'trust_region: while candidate_index < trust_candidates.len() {
                let (rotation_alpha, center_alpha) = trust_candidates[candidate_index];
                candidate_index += 1;
                let mut candidate_poses = proposal_poses.clone();
                candidate_poses[image] = Some(interpolate_structureless_pose_components(
                    proposal_pose,
                    refined_pose,
                    rotation_alpha,
                    center_alpha,
                ));
                let candidate_consistency = structureless_pose_consistency(
                    candidate_poses[image].as_ref().unwrap(),
                    &constraints,
                    &proposal,
                    config,
                );
                // Local tracks triangulated at the unconstrained proposal may
                // be invalid at the projected pose (and vice versa). Rebuild
                // the bounded submap at each geometry-feasible trust-region
                // pose from the pre-insertion state. This keeps landmark
                // synthesis consistent with the camera pose being admitted.
                let mut candidate_tracks = tracks_before.clone();
                let mut candidate_points = points_before.clone();
                let candidate_local_tracks = if candidate_consistency.accepted {
                    build_structureless_local_tracks(
                        camera,
                        features,
                        pairwise,
                        &candidate_tracks,
                        &candidate_points,
                        &candidate_poses,
                        image,
                        &constraints,
                        &proposal,
                        config,
                    )
                } else {
                    Vec::new()
                };
                let candidate_local_observations: HashSet<(usize, usize)> = candidate_local_tracks
                    .iter()
                    .flat_map(|(track, _)| track.iter().copied())
                    .collect();
                for (track_id, track) in candidate_tracks.iter_mut().enumerate() {
                    if candidate_points[track_id].is_none() {
                        track.retain(|observation| {
                            !candidate_local_observations.contains(observation)
                        });
                    }
                }
                for (track, point) in candidate_local_tracks {
                    candidate_tracks.push(track);
                    candidate_points.push(Some(point));
                }
                if candidate_consistency.accepted {
                    triangulate_pending(
                        camera,
                        features,
                        &candidate_tracks,
                        &candidate_poses,
                        config,
                        &mut candidate_points,
                    );
                }
                let submap_ok = candidate_consistency.accepted
                    && refine_structureless_new_landmarks(
                        camera,
                        features,
                        &candidate_tracks,
                        config,
                        &candidate_poses,
                        &mut candidate_points,
                        image,
                        &clean_point_mask,
                    )
                    .is_ok();
                let (candidate_support, candidate_image_mean) = supported_tracks_for_image(
                    camera,
                    features,
                    &candidate_tracks,
                    &candidate_poses,
                    &candidate_points,
                    image,
                    config.max_reprojection_error_px,
                );
                let candidate_clean_mean = mean_reprojection_for_registered_mask(
                    camera,
                    features,
                    &candidate_tracks,
                    &candidate_poses,
                    &candidate_points,
                    &registered_mask,
                    &clean_point_mask,
                );
                let candidate_ok = submap_ok
                    && candidate_support >= config.structureless_min_support_tracks
                    && candidate_image_mean.is_finite()
                    && candidate_image_mean <= config.structureless_max_reprojection_error_px
                    && clean_mean_before.is_finite()
                    && candidate_clean_mean.is_finite()
                    && candidate_clean_mean <= allowed_clean_mean + 1e-12
                    && candidate_consistency.accepted;
                let near_candidate = submap_ok
                    && candidate_support >= config.structureless_min_support_tracks
                    && candidate_image_mean.is_finite()
                    && clean_mean_before.is_finite()
                    && candidate_clean_mean.is_finite()
                    && candidate_clean_mean <= allowed_clean_mean + 1e-12
                    && candidate_consistency.accepted;
                if near_candidate
                    && best_near_candidate
                        .is_none_or(|(_, _, best_mean)| candidate_image_mean < best_mean)
                {
                    best_near_candidate =
                        Some((rotation_alpha, center_alpha, candidate_image_mean));
                }
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: structure-less image {image} trust rotation-alpha={rotation_alpha:.2} \
                         center-alpha={center_alpha:.2} \
                         tracks={} support={candidate_support} mean={candidate_image_mean:.3}px \
                         clean={candidate_clean_mean:.6} rot={:.3}deg forward={:.4} \
                         line={:.4} submap-ok={submap_ok} accepted={candidate_ok}",
                        candidate_tracks.len().saturating_sub(tracks_before_len),
                        candidate_consistency.max_rotation_deg,
                        candidate_consistency.min_forward_ratio,
                        candidate_consistency.line_error_ratio,
                    );
                }
                if candidate_ok {
                    poses.clone_from_slice(&candidate_poses);
                    *tracks = candidate_tracks;
                    *track_point = candidate_points;
                    support = candidate_support;
                    image_mean = candidate_image_mean;
                    clean_mean_after = candidate_clean_mean;
                    pose_consistency = Some(candidate_consistency);
                    support_ok = true;
                    image_error_ok = true;
                    clean_ok = true;
                    geometry_ok = true;
                    accepted = true;
                    trust_region_alpha = Some((rotation_alpha, center_alpha));
                    break 'trust_region;
                }
                if candidate_index == trust_candidates.len() && !fine_candidates_enqueued {
                    fine_candidates_enqueued = true;
                    if let Some((best_rotation, best_center, _)) = best_near_candidate {
                        let rotation_percent = (best_rotation * 100.0).round() as i32;
                        let center_percent = (best_center * 100.0).round() as i32;
                        for fine_rotation in
                            ((rotation_percent - 5).max(0)..=(rotation_percent + 5).min(100)).rev()
                        {
                            for fine_center in
                                ((center_percent - 5).max(0)..=(center_percent + 5).min(100)).rev()
                            {
                                if fine_rotation % 5 != 0 || fine_center % 5 != 0 {
                                    trust_candidates.push((
                                        fine_rotation as f64 / 100.0,
                                        fine_center as f64 / 100.0,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        let proposal_accepted = proposal_support >= config.structureless_min_support_tracks
            && proposal_image_mean.is_finite()
            && proposal_image_mean <= config.structureless_max_reprojection_error_px
            && clean_mean_before.is_finite()
            && proposal_clean_mean.is_finite()
            && proposal_clean_mean <= allowed_clean_mean + 1e-12
            && proposal_consistency.is_some_and(|diagnostic| diagnostic.accepted);
        if accepted {
            registered += 1;
            if sfm_debug_enabled() {
                if let Some((rotation_alpha, center_alpha)) = trust_region_alpha {
                    eprintln!(
                        "sfm-debug: structure-less image {image} projected BA step \
                         to trust-region rotation-alpha={rotation_alpha:.2} \
                         center-alpha={center_alpha:.2}"
                    );
                }
                eprintln!(
                    "sfm-debug: structure-less registered image {image} \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6})",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    support,
                    image_mean,
                    clean_mean_before,
                    clean_mean_after,
                );
            }
        } else if proposal_accepted {
            // Local BA is optional for admission: if it leaves the independent
            // relative-pose consensus, retain the already-gated scale-bearing
            // proposal and its newly triangulated structure, not the BA drift.
            poses.clone_from_slice(&proposal_poses);
            track_point.clone_from_slice(&proposal_points);
            registered += 1;
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: structure-less registered image {image} pose-only \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6}; local BA rejected)",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    proposal_support,
                    proposal_image_mean,
                    clean_mean_before,
                    proposal_clean_mean,
                );
            }
        } else {
            *tracks = tracks_before;
            track_point.truncate(points_before.len());
            poses.clone_from_slice(&poses_before);
            track_point.clone_from_slice(&points_before);
            if sfm_debug_enabled() {
                let (pose_rotation_deg, pose_forward_ratio, pose_line_ratio) = pose_consistency
                    .map(|diagnostic| {
                        (
                            diagnostic.max_rotation_deg,
                            diagnostic.min_forward_ratio,
                            diagnostic.line_error_ratio,
                        )
                    })
                    .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
                eprintln!(
                    "sfm-debug: structure-less image {image} rolled back \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6} allowed={:.6} local-ok={} support-ok={} \
                     image-ok={} clean-ok={} geometry-ok={} pose-rot={:.3}deg \
                     pose-forward={:.4} pose-line={:.4}; proposal-support={} \
                     proposal-mean={:.3}px proposal-clean={:.6} proposal-ok={})",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    support,
                    image_mean,
                    clean_mean_before,
                    clean_mean_after,
                    allowed_clean_mean,
                    local_ok,
                    support_ok,
                    image_error_ok,
                    clean_ok,
                    geometry_ok,
                    pose_rotation_deg,
                    pose_forward_ratio,
                    pose_line_ratio,
                    proposal_support,
                    proposal_image_mean,
                    proposal_clean_mean,
                    proposal_accepted,
                );
            }
        }
    }
    registered
}

/// Triangulate every track that has ≥2 registered observations and is not yet
/// triangulated, accepting only well-conditioned (parallax + reprojection) points.
fn triangulate_pending(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_some() {
            continue;
        }
        // Registered observations of this track: (image, pixel, world ray).
        let mut obs: Vec<(usize, Point2<f64>)> = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push((image, px));
            }
        }
        if obs.len() < 2 {
            continue;
        }
        if let Some(point) = triangulate_track(camera, poses, &obs, config) {
            track_point[track_id] = Some(point);
        }
    }
}

/// Whether a track's widest parallax `angle` (radians) clears the triangulation
/// gate: the strict `min_triangulation_angle_deg`, or — with the multi-view
/// exemption (`low_parallax_min_observations`) configured — the relaxed
/// `low_parallax_min_angle_deg` floor once at least that many views observe it.
fn parallax_angle_ok(angle: f64, num_obs: usize, config: &IncrementalSfmConfig) -> bool {
    if angle >= config.min_triangulation_angle_deg.to_radians() {
        return true;
    }
    match config.low_parallax_min_observations {
        Some(min_obs) => {
            num_obs >= min_obs && angle >= config.low_parallax_min_angle_deg.to_radians()
        }
        None => false,
    }
}

/// Triangulate one track from its registered observations: choose the
/// widest-parallax view pair, DLT-triangulate, and validate cheirality,
/// parallax, and reprojection in both views.
pub(crate) fn triangulate_track(
    camera: &Camera,
    poses: &[Option<Pose>],
    obs: &[(usize, Point2<f64>)],
    config: &IncrementalSfmConfig,
) -> Option<Point3<f64>> {
    let max_reproj = config.max_reprojection_error_px;
    // Precompute world-frame bearing rays for each observation.
    let mut rays: Vec<Vector3<f64>> = Vec::with_capacity(obs.len());
    for &(image, px) in obs {
        let pose = poses[image].as_ref()?;
        let n = camera.normalize_pixel(&px)?;
        let bearing = Vector3::new(n.x, n.y, 1.0).normalize();
        rays.push(pose.camera_to_world().rotation * bearing);
    }

    // Pick the observation pair with the smallest |cos| (widest parallax).
    let mut best: Option<(usize, usize, f64)> = None;
    for a in 0..obs.len() {
        for b in (a + 1)..obs.len() {
            let cos = rays[a].dot(&rays[b]).clamp(-1.0, 1.0).abs();
            if best.is_none_or(|(_, _, c)| cos < c) {
                best = Some((a, b, cos));
            }
        }
    }
    let (a, b, cos) = best?;
    // Widest-pair parallax angle; accept on the strict gate or the multi-view
    // exemption (a long low-parallax track is well-constrained by its many views).
    if !parallax_angle_ok(cos.acos(), obs.len(), config) {
        return None; // insufficient parallax
    }

    let (image_a, px_a) = obs[a];
    let (image_b, px_b) = obs[b];
    let pose_a = poses[image_a].as_ref()?;
    let pose_b = poses[image_b].as_ref()?;

    // Relative transform mapping camera-a frame to camera-b frame.
    let a_to_b = pose_b.world_to_camera.compose(&pose_a.camera_to_world());
    let point_cam_a = triangulate_two_view_left_frame(camera, camera, &a_to_b, &px_a, &px_b)?;
    if !point_cam_a.z.is_finite() || point_cam_a.z <= 0.0 {
        return None;
    }
    let point_world = pose_a.camera_to_world().transform_point(&point_cam_a);

    // Validate reprojection in both anchor views.
    for (image, px) in [(image_a, px_a), (image_b, px_b)] {
        let pose = poses[image].as_ref()?;
        let err = reprojection_error_px(camera, pose, &point_world, &px)?;
        if err > max_reproj {
            return None;
        }
    }
    Some(point_world)
}

#[derive(Debug, Clone)]
struct RecoveredConflictTrack {
    observations: Vec<(usize, usize)>,
    point: Point3<f64>,
    registered_observations: usize,
    mean_reprojection_px: f64,
}

/// Split dropped union-find conflict components against an already-posed model.
///
/// This deliberately does not trust descriptor distance or image-pair support
/// as a global ordering (both were catastrophic on MH_03). A verified edge is
/// only an anchor proposal. The resulting 3D hypothesis must explain a unique
/// observation in at least three registered images, and those selected
/// observations must contain a cycle in the verified correspondence graph.
fn recover_conflict_tracks_geometry(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    conflicting_components: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
) -> Vec<RecoveredConflictTrack> {
    if conflicting_components.is_empty() || config.conflict_recovery_max_hypotheses == 0 {
        return Vec::new();
    }

    let mut component_of = HashMap::new();
    for (component_id, component) in conflicting_components.iter().enumerate() {
        for &observation in component {
            component_of.insert(observation, component_id);
        }
    }

    type Observation = (usize, usize);
    let mut edges_by_component: Vec<Vec<(Observation, Observation)>> =
        vec![Vec::new(); conflicting_components.len()];
    for pair in pairwise {
        for &(kp_i, kp_j) in &pair.matches {
            let a = (pair.image_i, kp_i);
            let b = (pair.image_j, kp_j);
            let Some(&component_id) = component_of.get(&a) else {
                continue;
            };
            if component_of.get(&b) != Some(&component_id) {
                continue;
            }
            let edge = if a <= b { (a, b) } else { (b, a) };
            edges_by_component[component_id].push(edge);
        }
    }
    for edges in &mut edges_by_component {
        edges.sort_unstable();
        edges.dedup();
    }

    let mut triangulation_config = config.clone();
    triangulation_config.max_reprojection_error_px =
        config.conflict_recovery_max_reprojection_error_px;
    triangulation_config.low_parallax_min_observations = None;
    let min_views = config.conflict_recovery_min_views.max(3);
    let observation_pixel = |&(image, kp): &Observation| {
        features
            .get(image)
            .and_then(|feature_set| feature_set.keypoints.get(kp))
            .copied()
    };

    let mut recovered = Vec::new();
    for (component, edges) in conflicting_components.iter().zip(edges_by_component.iter()) {
        let mut adjacency: HashMap<Observation, Vec<Observation>> = HashMap::new();
        for &(a, b) in edges {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
        let mut ranked_anchors = Vec::new();
        for &(a, b) in edges {
            let (Some(pose_a), Some(pose_b), Some(px_a), Some(px_b)) = (
                poses.get(a.0).and_then(Option::as_ref),
                poses.get(b.0).and_then(Option::as_ref),
                observation_pixel(&a),
                observation_pixel(&b),
            ) else {
                continue;
            };
            let Some(n_a) = camera.normalize_pixel(&px_a) else {
                continue;
            };
            let Some(n_b) = camera.normalize_pixel(&px_b) else {
                continue;
            };
            let ray_a =
                pose_a.camera_to_world().rotation * Vector3::new(n_a.x, n_a.y, 1.0).normalize();
            let ray_b =
                pose_b.camera_to_world().rotation * Vector3::new(n_b.x, n_b.y, 1.0).normalize();
            let angle = ray_a.dot(&ray_b).clamp(-1.0, 1.0).abs().acos();
            if angle.is_finite() {
                ranked_anchors.push((angle, a, b, px_a, px_b));
            }
        }
        ranked_anchors.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        let mut best: Option<RecoveredConflictTrack> = None;
        for &(_angle, anchor_a, anchor_b, px_a, px_b) in ranked_anchors
            .iter()
            .take(config.conflict_recovery_max_hypotheses)
        {
            let anchor_observations = [(anchor_a.0, px_a), (anchor_b.0, px_b)];
            let Some(point) =
                triangulate_track(camera, poses, &anchor_observations, &triangulation_config)
            else {
                continue;
            };

            // First retain every registered observation consistent with the 3D
            // hypothesis. A component can contain several keypoints from one
            // image, so keep only that image's lowest-residual observation.
            let mut valid_errors: HashMap<Observation, f64> = HashMap::new();
            for &observation in component {
                let Some(pose) = poses.get(observation.0).and_then(Option::as_ref) else {
                    continue;
                };
                let Some(pixel) = observation_pixel(&observation) else {
                    continue;
                };
                let Some(error) = reprojection_error_px(camera, pose, &point, &pixel) else {
                    continue;
                };
                if error <= config.conflict_recovery_max_reprojection_error_px {
                    valid_errors.insert(observation, error);
                }
            }
            if !valid_errors.contains_key(&anchor_a) || !valid_errors.contains_key(&anchor_b) {
                continue;
            }

            // Restrict evidence to the verified-edge component containing the
            // anchor, then enforce one observation per image.
            let mut reachable = HashSet::from([anchor_a]);
            let mut frontier = vec![anchor_a];
            while let Some(node) = frontier.pop() {
                for &neighbour in adjacency.get(&node).into_iter().flatten() {
                    if valid_errors.contains_key(&neighbour) && reachable.insert(neighbour) {
                        frontier.push(neighbour);
                    }
                }
            }
            let mut best_by_image: HashMap<usize, (usize, f64)> = HashMap::new();
            for observation in reachable {
                let error = valid_errors[&observation];
                let entry = best_by_image
                    .entry(observation.0)
                    .or_insert((observation.1, error));
                if error < entry.1 || (error == entry.1 && observation.1 < entry.0) {
                    *entry = (observation.1, error);
                }
            }
            let mut selected: Vec<Observation> = best_by_image
                .iter()
                .map(|(&image, &(kp, _))| (image, kp))
                .collect();
            selected.sort_unstable();
            if selected.len() < min_views {
                continue;
            }
            let selected_set: HashSet<_> = selected.iter().copied().collect();
            let cycle_edges = edges
                .iter()
                .filter(|(a, b)| selected_set.contains(a) && selected_set.contains(b))
                .count();
            // A connected N-view tree has N-1 edges. Requiring N edges means
            // at least one independent cycle supports the hypothesis.
            if cycle_edges < selected.len() {
                continue;
            }
            let mean_reprojection_px = selected
                .iter()
                .map(|observation| valid_errors[observation])
                .sum::<f64>()
                / selected.len() as f64;
            if mean_reprojection_px > config.conflict_recovery_max_mean_reprojection_px {
                continue;
            }

            let registered_observations = selected.len();
            // An unregistered observation cannot be reprojection-checked yet.
            // Keep one only when it has at least two verified edges into the
            // accepted registered cycle; PnP RANSAC remains the final guard.
            let mut unregistered_support: HashMap<Observation, usize> = HashMap::new();
            for &(edge_a, edge_b) in edges {
                for (candidate, supported) in [(edge_a, edge_b), (edge_b, edge_a)] {
                    if poses.get(candidate.0).is_some_and(|pose| pose.is_none())
                        && selected_set.contains(&supported)
                    {
                        *unregistered_support.entry(candidate).or_insert(0) += 1;
                    }
                }
            }
            let mut inferred_by_image: HashMap<usize, (usize, usize)> = HashMap::new();
            for (observation, support) in unregistered_support {
                if support < 2 {
                    continue;
                }
                let entry = inferred_by_image
                    .entry(observation.0)
                    .or_insert((observation.1, support));
                if support > entry.1 || (support == entry.1 && observation.1 < entry.0) {
                    *entry = (observation.1, support);
                }
            }
            selected.extend(
                inferred_by_image
                    .into_iter()
                    .map(|(image, (kp, _))| (image, kp)),
            );
            selected.sort_unstable();

            let candidate = RecoveredConflictTrack {
                observations: selected,
                point,
                registered_observations,
                mean_reprojection_px,
            };
            let replace = best.as_ref().is_none_or(|current| {
                candidate.registered_observations > current.registered_observations
                    || (candidate.registered_observations == current.registered_observations
                        && (candidate.observations.len() > current.observations.len()
                            || (candidate.observations.len() == current.observations.len()
                                && candidate.mean_reprojection_px < current.mean_reprojection_px)))
            });
            if replace {
                best = Some(candidate);
            }
        }
        if let Some(track) = best {
            recovered.push(track);
        }
    }
    recovered
}

fn mean_reprojection_for_track_range(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    start: usize,
    end: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (track_id, track) in tracks.iter().enumerate().take(end).skip(start) {
        let Some(point) = track_point.get(track_id).and_then(|point| *point) else {
            continue;
        };
        for &(image, kp) in track {
            let (Some(pose), Some(pixel)) = (
                poses.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|feature_set| feature_set.keypoints.get(kp)),
            ) else {
                continue;
            };
            if let Some(error) = reprojection_error_px(camera, pose, &point, pixel) {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

/// Among unregistered images still under the per-image trial cap, choose the one
/// observing the most triangulated tracks, returning it with its 2D-3D
/// correspondences.
fn select_next_image(
    camera: &Camera,
    policy: NextImagePolicy,
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    trials: &[usize],
    max_trials: usize,
    track_point: &[Option<Point3<f64>>],
) -> Option<(usize, Vec<Correspondence2D3D>)> {
    // COLMAP's `IncrementalMapper::RankNextImages`: rank candidate images not by
    // the raw *count* of 2D–3D correspondences but by a multi-resolution
    // **visibility-pyramid score** that rewards correspondences *well distributed*
    // across the image (better-conditioned PnP), with the count as a tiebreak. An
    // image with many points clustered in one corner is a worse next view than one
    // with fewer points spread over the frame, and this score prefers the latter.
    let mut best: Option<(usize, (usize, usize), Vec<Correspondence2D3D>)> = None;
    for (image, observations) in obs_by_image.iter().enumerate() {
        if poses[image].is_some() || trials[image] >= max_trials {
            continue;
        }
        let mut corrs = Vec::new();
        for &(kp, track_id) in observations {
            let Some(point3d) = track_point[track_id] else {
                continue;
            };
            let Some(point2d) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            corrs.push(Correspondence2D3D {
                point2d,
                point3d,
                confidence: None,
            });
        }
        if corrs.len() < 6 {
            continue; // DLT PnP needs ≥6
        }
        let key = next_image_rank(camera, policy, &corrs);
        if best.as_ref().is_none_or(|(_, b, _)| key > *b) {
            best = Some((image, key, corrs));
        }
    }
    best.map(|(image, _, corrs)| (image, corrs))
}

fn next_image_rank(
    camera: &Camera,
    policy: NextImagePolicy,
    corrs: &[Correspondence2D3D],
) -> (usize, usize) {
    match policy {
        NextImagePolicy::VisibilityPyramid => (
            visibility_pyramid_score(
                camera.width,
                camera.height,
                corrs.iter().map(|corr| corr.point2d),
            ),
            corrs.len(),
        ),
        NextImagePolicy::CorrespondenceCount => (corrs.len(), 0),
    }
}

/// M4 diagnosis helper (`docs/colmap_port_plan.md`'s "M4 results"): classify,
/// for every still-unregistered image, *why* [`select_next_image`] will not
/// offer it — genuinely insufficient 2D-3D correspondences to a triangulated
/// track (`< 6`, the DLT/P3P minimal-sample floor), or a sufficient count but
/// an exhausted `max_registration_trials` budget. Debug-only (gated by
/// [`sfm_debug_enabled`] at the call site); this does no RANSAC of its own —
/// it only counts correspondences, so it is cheap enough to call at every
/// growth stall without affecting the release path's behaviour or perf.
fn diagnose_unregistered_images(
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    trials: &[usize],
    max_trials: usize,
    track_point: &[Option<Point3<f64>>],
) -> Vec<String> {
    let mut lines = Vec::new();
    for (image, observations) in obs_by_image.iter().enumerate() {
        if poses[image].is_some() {
            continue;
        }
        let corr_count = observations
            .iter()
            .filter(|&&(_, track_id)| track_point[track_id].is_some())
            .count();
        let reason = if corr_count < 6 {
            format!("insufficient correspondences ({corr_count} < 6)")
        } else if trials[image] >= max_trials {
            format!(
                "trials exhausted ({}/{max_trials}, {corr_count} corrs available)",
                trials[image]
            )
        } else {
            format!(
                "eligible but not selected this round ({corr_count} corrs, {}/{max_trials} trials)",
                trials[image]
            )
        };
        lines.push(format!("  image {image}: {reason}"));
    }
    lines
}

/// COLMAP visibility-pyramid score (`Image::Point3DVisibilityScore`): occupancy of
/// a stack of grids at increasing resolution (`2×2`, `4×4`, … up to `64×64`), each
/// cell counted **once** regardless of how many points land in it. Spreading
/// observations across the frame lights up more cells at every level, so the score
/// rewards spatial distribution and saturates on clusters — unlike a raw point
/// count. Returns the number of occupied cells summed over all pyramid levels.
fn visibility_pyramid_score(
    width: u32,
    height: u32,
    points: impl Iterator<Item = Point2<f64>>,
) -> usize {
    const NUM_LEVELS: u32 = 6;
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let mut occupied: Vec<HashSet<(u32, u32)>> = vec![HashSet::new(); NUM_LEVELS as usize];
    for p in points {
        // Clamp into the image so an out-of-frame keypoint cannot index past a grid.
        let fx = (p.x / w).clamp(0.0, 0.999_999);
        let fy = (p.y / h).clamp(0.0, 0.999_999);
        for level in 0..NUM_LEVELS {
            let dim = 1u32 << (level + 1); // 2, 4, 8, 16, 32, 64
            let cx = (fx * dim as f64) as u32;
            let cy = (fy * dim as f64) as u32;
            occupied[level as usize].insert((cx, cy));
        }
    }
    occupied.iter().map(|cells| cells.len()).sum()
}

/// Global BA over all registered poses + triangulated landmarks. Seed pose
/// (the lowest-index registered image) is fixed for gauge. Writes refined
/// poses and points back in place. When `refine_intrinsics` is set, the BA also
/// refines the pinhole intrinsics (alternating) and the refined camera is
/// returned as `Some` (the caller propagates it); otherwise the second tuple
/// element is `None` and the camera is untouched.
fn run_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
) -> Result<(BaResult, Option<Camera>), BaError> {
    let mut ba = BundleAdjustment::new(camera.clone());

    for (image, pose) in poses.iter().enumerate() {
        if let Some(pose) = pose {
            ba.add_pose(image as u64, pose.clone());
        }
    }

    // Gauge fixing. A monocular reconstruction (no stereo residual) has 7 gauge
    // freedoms: 6 for the rigid SE(3) frame plus **1 for global scale**. Fixing
    // a single pose pins only the 6 rigid DoF and leaves scale unconstrained, so
    // the BA's normal equations are singular along the scale direction. A single
    // solve from a perturbed state tolerates that (the damping holds the null
    // direction), but **re-optimising from an already-converged state lets the
    // scale drift and the reconstruction collapse**. Pin scale too by also
    // fixing the registered pose whose camera centre is farthest from the
    // anchor — the longest, best-conditioned baseline.
    let anchor = poses.iter().position(|p| p.is_some());
    if let Some(anchor) = anchor {
        ba.fix_pose(anchor as u64);
        let anchor_center = poses[anchor]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .translation;
        let mut farthest = None;
        let mut best_d2 = 0.0;
        for (image, pose) in poses.iter().enumerate() {
            if image == anchor {
                continue;
            }
            if let Some(pose) = pose {
                let d2 = (pose.camera_to_world().translation - anchor_center).norm_squared();
                if d2 > best_d2 {
                    best_d2 = d2;
                    farthest = Some(image);
                }
            }
        }
        if let Some(scale_anchor) = farthest {
            ba.fix_pose(scale_anchor as u64);
        }
    }

    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        let mut obs = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: px,
                });
            }
        }
        if obs.len() >= 2 {
            ba.add_landmark(track_id as u64, point);
            for o in obs {
                ba.add_observation(o);
            }
        }
    }

    let ba_config = BaConfig {
        refine_intrinsics,
        ..config.ba_config
    };
    let result = ba.optimize(&ba_config)?;

    for (image, pose) in poses.iter_mut().enumerate() {
        if pose.is_some() {
            if let Some(refined) = ba.poses.get(&(image as u64)) {
                *pose = Some(refined.clone());
            }
        }
    }
    for (track_id, point) in track_point.iter_mut().enumerate() {
        if point.is_some() {
            if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
                *point = Some(*refined);
            }
        }
    }
    let refined_camera = refine_intrinsics.then(|| ba.camera.clone());
    Ok((result, refined_camera))
}

/// COLMAP `IncrementalMapper::AdjustLocalBundle`. After registering `new_image`,
/// bundle-adjust only it and its `local_ba_num_images` most-covisible registered
/// neighbours (sharing the most triangulated tracks) plus the points they see —
/// every *other* registered image that observes one of those points is added as a
/// **fixed** pose, so it constrains the local solve without being moved. This
/// keeps the freshly grown geometry tight after every step at a fraction of a
/// global solve's cost, the schedule that lets COLMAP hold sub-centimetre
/// accuracy as the reconstruction grows. Poses/points outside the variable set
/// are untouched.
fn adjust_local_bundle(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    new_image: usize,
) -> Result<(), BaError> {
    // Covisible registered images: how many triangulated tracks each shares with
    // the newly registered one.
    let mut covis: HashMap<usize, usize> = HashMap::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        if !track
            .iter()
            .any(|&(img, _)| img == new_image && poses[img].is_some())
        {
            continue;
        }
        for &(img, _) in track {
            if img != new_image && poses[img].is_some() {
                *covis.entry(img).or_insert(0) += 1;
            }
        }
    }
    let mut neighbours: Vec<(usize, usize)> = covis.into_iter().collect();
    // Most-covisible first; break ties by index for determinism.
    neighbours.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut variable: HashSet<usize> = neighbours
        .into_iter()
        .take(config.local_ba_num_images)
        .map(|(img, _)| img)
        .collect();
    variable.insert(new_image);

    bundle_adjust_local(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        &variable,
    )
}

/// With every camera and every pre-existing landmark fixed, refine only the
/// landmarks created by a tentative structure-less insertion. This is the
/// bounded local-submap solve used after projecting the new camera into the
/// independent relative-geometry feasible region.
fn refine_structureless_new_landmarks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    new_image: usize,
    preexisting_points: &[bool],
) -> Result<(), BaError> {
    let mut landmark_ids = Vec::new();
    let mut used_images = HashSet::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if preexisting_points.get(track_id).copied().unwrap_or(false)
            || track_point[track_id].is_none()
            || !track.iter().any(|&(image, kp)| {
                image == new_image
                    && poses[image].is_some()
                    && features[image].keypoints.get(kp).is_some()
            })
        {
            continue;
        }
        let observers: Vec<usize> = track
            .iter()
            .filter_map(|&(image, kp)| {
                (poses[image].is_some() && features[image].keypoints.get(kp).is_some())
                    .then_some(image)
            })
            .collect();
        if observers.len() < 2 {
            continue;
        }
        used_images.extend(observers);
        landmark_ids.push(track_id);
    }
    if landmark_ids.is_empty() {
        return Ok(());
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    for image in used_images {
        ba.add_pose(image as u64, poses[image].clone().unwrap());
        ba.fix_pose(image as u64);
    }
    for &track_id in &landmark_ids {
        ba.add_landmark(track_id as u64, track_point[track_id].unwrap());
        for &(image, kp) in &tracks[track_id] {
            if !ba.poses.contains_key(&(image as u64)) {
                continue;
            }
            if let Some(pixel) = features[image].keypoints.get(kp).copied() {
                ba.add_observation(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: pixel,
                });
            }
        }
    }
    ba.optimize(&config.ba_config)?;
    for track_id in landmark_ids {
        if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
            track_point[track_id] = Some(*refined);
        }
    }
    Ok(())
}

/// Bundle-adjust a chosen `variable` set of poses plus every triangulated track
/// they observe. Other registered images observing those tracks join as fixed
/// poses (constraints). The gauge: with ≥2 fixed observers their baseline pins
/// the 7-DoF monocular gauge for free; otherwise (an early, loosely connected
/// neighbourhood) the variable set's own anchor + farthest pose are fixed, as in
/// the global solve. Only variable poses and the solved landmarks are written back.
fn bundle_adjust_local(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    variable: &HashSet<usize>,
) -> Result<(), BaError> {
    let mut ba = BundleAdjustment::new(camera.clone());

    // Landmarks touching ≥1 variable image, and the images that participate.
    let mut used: HashSet<usize> = HashSet::new();
    let mut lm_ids: Vec<usize> = Vec::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        let mut obs_images: Vec<usize> = Vec::new();
        let mut touches_variable = false;
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if features[image].keypoints.get(kp).is_none() {
                continue;
            }
            obs_images.push(image);
            if variable.contains(&image) {
                touches_variable = true;
            }
        }
        if obs_images.len() < 2 || !touches_variable {
            continue;
        }
        for image in obs_images {
            used.insert(image);
        }
        lm_ids.push(track_id);
    }
    if lm_ids.is_empty() {
        return Ok(());
    }

    for &image in &used {
        ba.add_pose(image as u64, poses[image].clone().unwrap());
        if !variable.contains(&image) {
            ba.fix_pose(image as u64);
        }
    }

    // Need ≥2 fixed poses to pin metric scale; otherwise fix the variable gauge.
    let n_fixed = used.iter().filter(|i| !variable.contains(i)).count();
    if n_fixed < 2 {
        let var_used: Vec<usize> = used
            .iter()
            .copied()
            .filter(|i| variable.contains(i))
            .collect();
        fix_monocular_scale_gauge(&mut ba, poses, &var_used);
    }

    for &track_id in &lm_ids {
        ba.add_landmark(track_id as u64, track_point[track_id].unwrap());
        for &(image, kp) in &tracks[track_id] {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                ba.add_observation(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: px,
                });
            }
        }
    }
    ba.optimize(&config.ba_config)?;

    for &image in &used {
        if variable.contains(&image) {
            if let Some(refined) = ba.poses.get(&(image as u64)) {
                poses[image] = Some(refined.clone());
            }
        }
    }
    for &track_id in &lm_ids {
        if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
            track_point[track_id] = Some(*refined);
        }
    }
    Ok(())
}

/// Pin the 7-DoF monocular gauge (6 rigid + scale) by fixing two of `candidates`:
/// the lowest-index pose (rigid anchor) and the one farthest from it (the scale
/// anchor — longest, best-conditioned baseline). Mirrors the global solve's gauge
/// handling; used by a local solve that lacks two fixed-observer poses of its own.
fn fix_monocular_scale_gauge(
    ba: &mut BundleAdjustment,
    poses: &[Option<Pose>],
    candidates: &[usize],
) {
    let Some(&anchor) = candidates.iter().min() else {
        return;
    };
    ba.fix_pose(anchor as u64);
    let anchor_center = poses[anchor]
        .as_ref()
        .unwrap()
        .camera_to_world()
        .translation;
    let mut farthest = None;
    let mut best_d2 = 0.0;
    for &image in candidates {
        if image == anchor {
            continue;
        }
        let d2 = (poses[image].as_ref().unwrap().camera_to_world().translation - anchor_center)
            .norm_squared();
        if d2 > best_d2 {
            best_d2 = d2;
            farthest = Some(image);
        }
    }
    if let Some(scale_anchor) = farthest {
        ba.fix_pose(scale_anchor as u64);
    }
}

/// COLMAP `IncrementalMapper::IterativeGlobalRefinement`: a global BA, then a loop
/// of {re-triangulate/complete tracks, filter outliers, global BA} until the
/// changed-observation fraction falls below `global_ba_change_rate` (or
/// `global_ba_max_refinements` rounds run). Re-triangulation is forced on here
/// regardless of `config.retriangulate` — completing tracks between global solves
/// is integral to COLMAP's schedule, not the opt-in density lever of the simple
/// path.
fn iterative_global_refinement(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &mut [Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<BaResult, BaError> {
    // Refine intrinsics on each global solve when enabled; the refined camera is
    // carried forward so the next round's filter / re-triangulation / BA all use
    // it (and the caller reads the final camera back from `*camera`).
    let refine = config.refine_intrinsics;
    let run_ba = |cam: &mut Camera,
                  tr: &[Vec<(usize, usize)>],
                  p: &mut [Option<Pose>],
                  tp: &mut [Option<Point3<f64>>]|
     -> Result<BaResult, BaError> {
        let (res, refined) = run_bundle_adjustment(cam, features, tr, config, p, tp, refine)?;
        if let Some(c) = refined {
            *cam = c;
        }
        Ok(res)
    };

    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: final global refinement begin registered={} points={} observations={} max_followup_rounds={}",
            poses.iter().filter(|pose| pose.is_some()).count(),
            track_point.iter().filter(|point| point.is_some()).count(),
            count_observations(tracks, poses, track_point),
            config.global_ba_max_refinements,
        );
    }
    let mut ba_started = std::time::Instant::now();
    let mut result = run_ba(camera, tracks, poses, track_point)?;
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: final global BA round=0 completed seconds={:.3}",
            ba_started.elapsed().as_secs_f64(),
        );
    }
    for followup in 0..config.global_ba_max_refinements {
        let round = followup + 1;
        let total_obs = count_observations(tracks, poses, track_point).max(1);
        // Filter outlier observations, then complete/re-triangulate tracks the
        // tightened frame can now place. Completing between solves is integral to
        // the schedule — it gives the next global BA more constraints and, on this
        // metric video, measurably beats filter-only (1.64 cm vs 2.21 cm); the
        // forward-motion low-parallax churn it induces against the filter is the
        // price, and the track-density ceiling it leaves is the next lever.
        let mut changed =
            filter_outlier_observations(camera, features, tracks, config, poses, track_point);
        changed += retriangulate_tracks(camera, features, tracks, config, poses, track_point);
        let change_rate = changed as f64 / total_obs as f64;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: final global refinement round={round} changed={changed}/{total_obs} rate={change_rate:.6} threshold={:.6}",
                config.global_ba_change_rate,
            );
        }
        if change_rate < config.global_ba_change_rate {
            if sfm_debug_enabled() {
                eprintln!("sfm-debug: final global refinement converged before BA round={round}");
            }
            break;
        }
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: final global BA round={round} begin");
        }
        ba_started = std::time::Instant::now();
        result = run_ba(camera, tracks, poses, track_point)?;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: final global BA round={round} completed seconds={:.3}",
                ba_started.elapsed().as_secs_f64(),
            );
        }
    }
    Ok(result)
}

/// In-growth global refinement, used during the seed search where `tracks` is
/// shared read-only across trials: global BA, then up to a couple rounds of
/// {re-triangulate/complete, global BA} while it keeps completing tracks. The
/// completion is what keeps registration moving — a freshly tightened global
/// frame lets [`retriangulate_tracks`] triangulate tracks the narrow growth-time
/// baseline had missed, and those new 3D points give the next PnP enough
/// 2D-3D matches to register (without it, registration stalls well short of full
/// coverage and the trajectory develops ATE-wrecking gaps). The track-membership
/// *filter* (which would mutate the shared tracks) is deferred to the final
/// [`iterative_global_refinement`] after a seed is committed.
fn growth_global_refinement(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<(), BaError> {
    // When intrinsics refinement is on, co-evolve them in these periodic global
    // passes (COLMAP's IterativeGlobalRefinement keeps the camera moving with the
    // structure, so a wrong focal is corrected while the model is still small
    // enough to expose it — the well-conditioned global solve, not the narrow
    // per-registration local one, is where the focal is observable). Otherwise the
    // intrinsics stay fixed and the refined slot is always None.
    let refine = config.refine_intrinsics;
    let run_global = |cam: &mut Camera,
                      p: &mut [Option<Pose>],
                      tp: &mut [Option<Point3<f64>>]|
     -> Result<(), BaError> {
        let (_, refined) = run_bundle_adjustment(cam, features, tracks, config, p, tp, refine)?;
        if let Some(c) = refined {
            *cam = c;
        }
        Ok(())
    };

    run_global(camera, poses, track_point)?;
    for _ in 0..config.global_ba_max_refinements.min(2) {
        let changed = retriangulate_tracks(camera, features, tracks, config, poses, track_point);
        if changed == 0 {
            break;
        }
        run_global(camera, poses, track_point)?;
    }
    Ok(())
}

/// Total triangulated observations: for every track with a 3D point, the number
/// of its registered observations. The denominator for the refinement-loop
/// change-rate stop test.
fn count_observations(
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> usize {
    let mut n = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        n += track
            .iter()
            .filter(|&&(img, _)| poses[img].is_some())
            .count();
    }
    n
}

/// COLMAP `Reconstruction::FilterImages`: de-register registered images whose
/// well-supported observation count has collapsed. For each registered image,
/// count its observations that are triangulated and reproject within
/// `max_reprojection_error_px`; if that count is below
/// `config.filter_min_image_observations`, set its pose to `None`. The two
/// lowest-index registered images (the seed pair) are protected as the gauge
/// anchor, and the registered count is never driven below 3. Returns how many
/// images were de-registered. The caller's grow loop resets the trial counter of
/// any now-unregistered image, so a filtered image can re-register once the
/// surrounding structure improves.
fn filter_images(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> usize {
    let threshold = config.max_reprojection_error_px;
    let min_obs = config.filter_min_image_observations;

    // Per-image count of well-supported (triangulated, in-threshold) observations.
    let mut good_obs = vec![0usize; poses.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        for &(image, kp) in track {
            let Some(pose) = &poses[image] else { continue };
            let Some(px) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            if matches!(reprojection_error_px(camera, pose, &point, &px), Some(e) if e <= threshold)
            {
                good_obs[image] += 1;
            }
        }
    }

    // Protect the seed pair (the two lowest-index registered images) — they pin the
    // 7-DoF monocular gauge — and keep at least three registered images alive.
    let registered: Vec<usize> = (0..poses.len()).filter(|&i| poses[i].is_some()).collect();
    let protected: std::collections::HashSet<usize> = registered.iter().take(2).copied().collect();
    let mut remaining = registered.len();

    let mut removed = 0usize;
    for &image in &registered {
        if remaining <= 3 || protected.contains(&image) {
            continue;
        }
        if good_obs[image] < min_obs {
            poses[image] = None;
            removed += 1;
            remaining -= 1;
        }
    }
    removed
}

/// Clean every triangulated track after the current BA, on two grounds:
///
/// 1. **Reprojection.** A contaminated union-find track — two distinct 3D points
///    merged into one — has a BA'd point that fits neither cluster, so its
///    observations reproject past `max_reprojection_error_px` and are stripped;
///    a track left below the minimum posed observations is dropped.
/// 2. **Parallax.** A point first triangulated just over the parallax gate is
///    depth-unstable: BA can slide it far along its viewing ray without changing
///    any reprojection (low parallax = depth ambiguity), so it survives the
///    reprojection test while sitting thousands of units from the scene — these
///    far-flung outliers wreck the scene scale for downstream 3DGS / MVS. So
///    re-measure parallax against the *current* point and all observing camera
///    centres (the widest angle subtended at the point), and drop the track if
///    it is below `min_triangulation_angle_deg`.
///
/// Observations in *unregistered* images are kept untouched (the BA already
/// ignores them); no pose is ever removed, so the registered-image count is
/// invariant. Returns how many tracks/observations changed (zero ⇒ converged).
fn filter_outlier_observations(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &mut [Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> usize {
    let threshold = config.max_reprojection_error_px;
    let min_obs = config.min_track_length.max(2);
    let mut changed = 0usize;

    for (track_id, track) in tracks.iter_mut().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        let before = track.len();
        track.retain(|&(image, kp)| {
            let Some(pose) = &poses[image] else {
                return true; // unregistered view: BA ignores it, cannot judge.
            };
            let Some(px) = features[image].keypoints.get(kp).copied() else {
                return false;
            };
            match reprojection_error_px(camera, pose, &point, &px) {
                Some(err) => err <= threshold,
                None => false, // behind the camera => outlier.
            }
        });
        changed += before - track.len();

        let posed_obs = track
            .iter()
            .filter(|&&(image, _)| poses[image].is_some())
            .count();
        if posed_obs < min_obs {
            if track_point[track_id].take().is_some() {
                changed += 1;
            }
            continue;
        }

        // Drop a low-parallax track unless the multi-view exemption keeps it: a
        // long forward-motion track below the strict angle but seen by many views
        // is well-constrained, while a 2-view depth-ambiguous one is not.
        if !parallax_angle_ok(track_max_parallax(poses, track, &point), posed_obs, config)
            && track_point[track_id].take().is_some()
        {
            changed += 1;
        }
    }
    changed
}

/// Re-triangulate tracks after a bundle adjustment has moved the poses — the
/// COLMAP completeness/refinement step the single-pass growth lacks. For each
/// track with ≥2 registered observations, triangulate a fresh point from the
/// current widest-parallax view pair ([`triangulate_track`], so it still passes
/// the parallax + reprojection gates) and either:
///
///  1. **Complete** an un-triangulated track. At growth time its registered
///     views were a narrow baseline and the parallax gate rejected it; the
///     BA-refined geometry (more views registered, wider baselines) can now place
///     it. The new point constrains the next BA.
///  2. **Re-seed** an existing point, but only as a **guarded swap**: keep the
///     re-triangulation only if it lowers the track's mean reprojection over its
///     registered observations. A point a multi-view BA already placed better is
///     never regressed, so the step is monotone per track.
///
/// Poses are read-only here; the caller re-runs the BA afterwards. Returns how
/// many tracks gained or improved a point (zero ⇒ nothing changed, converged).
fn retriangulate_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> usize {
    let mut changed = 0usize;

    for (track_id, track) in tracks.iter().enumerate() {
        // Registered observations of this track: (image, pixel).
        let mut obs: Vec<(usize, Point2<f64>)> = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push((image, px));
            }
        }
        if obs.len() < 2 {
            continue;
        }
        let Some(candidate) = triangulate_track(camera, poses, &obs, config) else {
            continue;
        };

        match track_point[track_id] {
            None => {
                track_point[track_id] = Some(candidate);
                changed += 1;
            }
            Some(current) => {
                // Mean reprojection of a point over this track's registered obs.
                let mean_reproj = |p: &Point3<f64>| -> f64 {
                    let mut sum = 0.0;
                    let mut n = 0usize;
                    for &(image, px) in &obs {
                        let Some(pose) = &poses[image] else { continue };
                        if let Some(err) = reprojection_error_px(camera, pose, p, &px) {
                            sum += err;
                            n += 1;
                        }
                    }
                    if n > 0 {
                        sum / n as f64
                    } else {
                        f64::INFINITY
                    }
                };
                if mean_reproj(&candidate) + 1e-9 < mean_reproj(&current) {
                    track_point[track_id] = Some(candidate);
                    changed += 1;
                }
            }
        }
    }
    changed
}

/// Widest angle (radians) subtended at `point` by any pair of registered camera
/// centres that observe it — the post-BA triangulation angle. Zero if fewer than
/// two registered views remain.
fn track_max_parallax(
    poses: &[Option<Pose>],
    track: &[(usize, usize)],
    point: &Point3<f64>,
) -> f64 {
    let dirs: Vec<Vector3<f64>> = track
        .iter()
        .filter_map(|&(image, _)| poses[image].as_ref())
        .filter_map(|pose| {
            let v = pose.camera_to_world().translation - point.coords;
            (v.norm() > f64::EPSILON).then(|| v.normalize())
        })
        .collect();
    let mut max_angle = 0.0;
    for a in 0..dirs.len() {
        for b in (a + 1)..dirs.len() {
            let angle = dirs[a].dot(&dirs[b]).clamp(-1.0, 1.0).acos();
            if angle > max_angle {
                max_angle = angle;
            }
        }
    }
    max_angle
}

/// Reprojection error (px) of `point_world` against pixel `px` in a camera.
/// `None` if the point is behind the camera or projection is degenerate.
pub(crate) fn reprojection_error_px(
    camera: &Camera,
    pose: &Pose,
    point_world: &Point3<f64>,
    px: &Point2<f64>,
) -> Option<f64> {
    let cam = pose.transform_world_point(point_world);
    if !cam.z.is_finite() || cam.z <= 0.0 {
        return None;
    }
    let projected = camera.project(&cam)?;
    Some((projected - px).norm())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    /// `LocalSubmapBuilder::build`'s scale-pathology retry
    /// (`NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(b)) relies on
    /// `seed_candidate_order` walking to the *next*-ranked seed candidate,
    /// deterministically, once the previously tried pair is excluded. This
    /// pins that mechanism directly: descending match-count order by
    /// default, and excluding a pair (regardless of which of its two image
    /// orderings is recorded) removes exactly that pair and nothing else,
    /// repeatably.
    #[test]
    fn seed_candidate_order_skips_excluded_pairs_deterministically() {
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0); 50],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0); 40],
            },
            PairwiseMatches {
                image_i: 2,
                image_j: 3,
                matches: vec![(0, 0); 30],
            },
        ];
        let config = IncrementalSfmConfig::default();
        assert_eq!(seed_candidate_order(&pairwise, &config), vec![0, 1, 2]);

        let mut excluded_first = config.clone();
        excluded_first.excluded_seed_pairs.insert((0, 1));
        assert_eq!(seed_candidate_order(&pairwise, &excluded_first), vec![1, 2]);
        // Deterministic: repeated calls on the same (excluded) config agree.
        assert_eq!(seed_candidate_order(&pairwise, &excluded_first), vec![1, 2]);

        // The pairwise-side key is normalized regardless of which image is
        // recorded as `image_i`/`image_j`: a reversed-direction entry for
        // the same underlying pair still matches a normalized `(0, 1)`
        // exclusion key.
        let mut reversed_first_pair = pairwise.clone();
        reversed_first_pair[0] = PairwiseMatches {
            image_i: 1,
            image_j: 0,
            matches: vec![(0, 0); 50],
        };
        let mut excluded_normalized = config.clone();
        excluded_normalized.excluded_seed_pairs.insert((0, 1));
        assert_eq!(
            seed_candidate_order(&reversed_first_pair, &excluded_normalized),
            vec![1, 2]
        );

        // Excluding the two strongest pairs walks to the third-ranked
        // candidate, still in descending order among what remains.
        let mut excluded_two = config;
        excluded_two.excluded_seed_pairs.insert((0, 1));
        excluded_two.excluded_seed_pairs.insert((1, 2));
        assert_eq!(seed_candidate_order(&pairwise, &excluded_two), vec![2]);
    }

    /// A synthetic 3D point cloud and a ring of cameras looking at it, used to
    /// exercise the full unordered pipeline end-to-end.
    struct Scene {
        camera: Camera,
        points: Vec<Point3<f64>>,
        poses: Vec<Pose>,
    }

    fn build_scene() -> Scene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        // A 3D grid of points around the origin.
        let mut points = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in 0..=2 {
                    points.push(Point3::new(
                        xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.3,
                    ));
                }
            }
        }
        // Cameras on an arc, all looking roughly toward the cloud centre from
        // ~3 m away (enough parallax between neighbours).
        let mut poses = Vec::new();
        for k in 0..6 {
            let angle = -0.5 + k as f64 * 0.2; // radians along the arc
            let radius = 3.0;
            let cam_center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            // Look-at the origin: build world_to_camera.
            let forward = (Point3::origin() - cam_center).normalize();
            let world_up = Vector3::new(0.0, 1.0, 0.0);
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward);
            // Rotation columns map camera axes (x=right, y=down, z=forward) to world.
            let r_cam_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let rot_c2w = nalgebra::Rotation3::from_matrix_unchecked(r_cam_to_world);
            let q_c2w = UnitQuaternion::from_rotation_matrix(&rot_c2w);
            let q_w2c = q_c2w.inverse();
            let t_w2c = -(q_w2c * cam_center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }
        Scene {
            camera,
            points,
            poses,
        }
    }

    /// Project a world point into a pose; `None` if behind camera or off-image.
    fn project(camera: &Camera, pose: &Pose, p: &Point3<f64>) -> Option<Point2<f64>> {
        let cam = pose.transform_world_point(p);
        if cam.z <= 0.05 {
            return None;
        }
        let px = camera.project(&cam)?;
        if px.x < 0.0 || px.x >= camera.width as f64 || px.y < 0.0 || px.y >= camera.height as f64 {
            return None;
        }
        Some(px)
    }

    /// Render the scene to per-image features (keypoint per visible point, the
    /// point index baked into a trivial descriptor) and ground-truth pairwise
    /// matches between every image pair that co-observes ≥8 points.
    fn render(scene: &Scene) -> (Vec<FeatureSet>, Vec<PairwiseMatches>) {
        let n = scene.poses.len();
        // visible[image] = map point_index -> keypoint_index
        let mut features = Vec::new();
        let mut visible: Vec<HashMap<usize, usize>> = Vec::new();
        for pose in &scene.poses {
            let mut kps = Vec::new();
            let mut descs = Vec::new();
            let mut vis = HashMap::new();
            for (pidx, p) in scene.points.iter().enumerate() {
                if let Some(px) = project(&scene.camera, pose, p) {
                    vis.insert(pidx, kps.len());
                    kps.push(px);
                    // Descriptor is irrelevant here (matches are ground truth),
                    // but FeatureSet wants one; use a tiny unique vector.
                    descs.push(vec![pidx as f32, 1.0, 0.0, 0.0]);
                }
            }
            features.push(FeatureSet::new(kps, descs).unwrap());
            visible.push(vis);
        }

        let mut pairwise = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut matches = Vec::new();
                for (pidx, &ki) in &visible[i] {
                    if let Some(&kj) = visible[j].get(pidx) {
                        matches.push((ki, kj));
                    }
                }
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                    });
                }
            }
        }
        (features, pairwise)
    }

    /// M2 acceptance test: on the same realistic multi-image synthetic scene
    /// every other integration test in this module uses
    /// (`build_scene`/`render` — a 45-point cloud seen by a 6-camera ring),
    /// [`build_tracks_via_graph`] must produce **byte-identical** tracks to
    /// the legacy [`build_tracks`] union-find — the refactor gate
    /// `docs/colmap_port_plan.md`'s M2 milestone specifies ("byte-identical
    /// tracks... a refactor gate, not an accuracy claim"), exercised here on
    /// real transitive (multi-hop, multi-image) structure rather than the
    /// small hand-built fixtures above.
    #[test]
    fn graph_tracks_match_union_find_tracks_on_synthetic_scene() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        assert!(
            !pairwise.is_empty(),
            "fixture sanity: the scene must produce at least one verified pair"
        );

        let union_find_tracks = build_tracks(features.len(), &pairwise, 2);
        let graph_tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert_eq!(
            union_find_tracks, graph_tracks,
            "CorrespondenceGraph-derived tracks must byte-match the legacy union-find's"
        );
        assert!(
            !union_find_tracks.is_empty(),
            "fixture sanity: some tracks must form"
        );
    }

    #[test]
    fn post_refinement_pass_registers_against_tightened_structure_once() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let mut poses = vec![None; features.len()];
        poses[0] = Some(scene.poses[0].clone());
        poses[1] = Some(scene.poses[1].clone());
        let mut track_point = vec![None; tracks.len()];
        let config = IncrementalSfmConfig {
            min_pnp_inliers: 8,
            max_reprojection_error_px: 2.0,
            ..IncrementalSfmConfig::default()
        };
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        assert!(track_point.iter().filter(|p| p.is_some()).count() >= 8);

        let added = post_refinement_registration_pass(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses,
            &mut track_point,
        )
        .unwrap();
        assert!(added > 0);
        assert_eq!(poses.iter().filter(|p| p.is_some()).count(), 2 + added);
    }

    /// M2 acceptance test, end-to-end: running the *full* `incremental_sfm`
    /// pipeline with [`TrackSource::CorrespondenceGraph`] instead of the
    /// default [`TrackSource::UnionFind`] on the same synthetic scene must
    /// register the same images and produce the same track count and mean
    /// reprojection error — i.e. the track-builder swap is invisible to
    /// every downstream stage (seeding, growth, bundle adjustment).
    #[test]
    fn incremental_sfm_matches_between_track_sources() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);

        let base_config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let union_find_config = IncrementalSfmConfig {
            track_source: TrackSource::UnionFind,
            ..base_config.clone()
        };
        let graph_config = IncrementalSfmConfig {
            track_source: TrackSource::CorrespondenceGraph,
            ..base_config
        };

        let union_find_result =
            incremental_sfm(&scene.camera, &features, &pairwise, &union_find_config)
                .expect("union-find track source must reconstruct this scene");
        let graph_result = incremental_sfm(&scene.camera, &features, &pairwise, &graph_config)
            .expect("CorrespondenceGraph track source must reconstruct this scene");

        assert_eq!(
            union_find_result.registered_images, graph_result.registered_images,
            "both track sources must register the same number of images"
        );
        assert_eq!(
            union_find_result.tracks.len(),
            graph_result.tracks.len(),
            "both track sources must produce the same number of output tracks"
        );
        assert!(
            (union_find_result.mean_reprojection_px - graph_result.mean_reprojection_px).abs()
                < 1.0e-6,
            "both track sources must reach the same mean reprojection error: {} vs {}",
            union_find_result.mean_reprojection_px,
            graph_result.mean_reprojection_px,
        );
    }

    /// M2.1 acceptance: `docs/colmap_port_plan.md`'s M2.1 milestone widens
    /// `examples/unordered_sfm_demo.rs`'s verified-pair keep-list so a
    /// `PANORAMIC` (pure-rotation, zero-baseline) pair now reaches
    /// `PairwiseMatches`/this mapper, matching COLMAP's own
    /// `database_cache.cc` `UseInlierMatchesCheck` gate. This must not make
    /// such a pair *seedable*: COLMAP's own
    /// `IncrementalMapperImpl::EstimateInitialTwoViewGeometry` re-derives its
    /// own relative pose and rejects init candidates whose triangulation
    /// angle doesn't clear `init_min_tri_angle`, independent of any stored
    /// `ConfigurationType` — this mapper's [`place_seed_pair`] already has
    /// the same independent architecture (re-estimate the relative pose,
    /// gate on how many inliers actually triangulate), so no new exclusion
    /// mechanism is needed; this test pins that the existing gate covers the
    /// newly-admitted pair type too.
    #[test]
    fn pure_rotation_pair_is_rejected_as_a_seed_even_though_it_now_reaches_pairwise() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        // A scattered point cloud with real depth variation (same shape as
        // `colmap_verification.rs`'s `general_scene_points` fixture).
        let mut points = Vec::new();
        for i in 0..6 {
            for j in 0..4 {
                points.push(Point3::new(
                    -1.5 + 0.6 * i as f64,
                    -1.0 + 0.7 * j as f64,
                    3.0 + 0.8 * ((i + j) % 5) as f64,
                ));
            }
        }

        // Camera 0 at the world origin; camera 1 at the SAME origin, only
        // rotated — a pure-rotation pair, zero baseline, exactly the
        // `PANORAMIC` configuration `TwoViewGeometryVerifier` would classify
        // this as (see `colmap_verification.rs`'s
        // `pure_rotation_classifies_panoramic`).
        let pose0 = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.12);
        let pose1 = Pose::from_world_to_camera(yaw, Vector3::zeros());

        let mut kp0 = Vec::new();
        let mut kp1 = Vec::new();
        let mut matches = Vec::new();
        for p in &points {
            if let (Some(px0), Some(px1)) =
                (project(&camera, &pose0, p), project(&camera, &pose1, p))
            {
                matches.push((kp0.len(), kp1.len()));
                kp0.push(px0);
                kp1.push(px1);
            }
        }
        assert!(
            matches.len() >= 15,
            "fixture sanity: pure rotation should still leave most points in both views"
        );

        let features = vec![
            FeatureSet::new(kp0, vec![vec![0.0f32; 4]; matches.len()]).unwrap(),
            FeatureSet::new(kp1, vec![vec![0.0f32; 4]; matches.len()]).unwrap(),
        ];
        let pair = PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches,
        };
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            ..IncrementalSfmConfig::default()
        };
        let mut poses = vec![None, None];
        assert!(
            !place_seed_pair(&camera, &features, &pair, &config, &mut poses),
            "a zero-baseline (panoramic) pair must never bootstrap a seed, \
             even though M2.1 now lets its correspondences reach PairwiseMatches"
        );
        assert!(
            poses[0].is_none() && poses[1].is_none(),
            "rejected seed must leave poses untouched"
        );
    }

    #[test]
    fn build_tracks_merges_shared_observations() {
        // Two images both see point P (kp 0 in each) and image-2 sees it too.
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
            },
        ];
        let tracks = build_tracks(3, &pairwise, 2);
        assert_eq!(tracks.len(), 1, "the chained matches form one track");
        assert_eq!(tracks[0].len(), 3, "track spans all three images");
    }

    #[test]
    fn track_build_preview_matches_union_find_topology_without_mapping() {
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
            },
        ];
        let features = vec![
            FeatureSet {
                keypoints: vec![Point2::new(0.0, 0.0)],
                descriptors: vec![vec![0.0]],
            };
            3
        ];
        let stats =
            preview_track_build_stats(&features, &pairwise, &IncrementalSfmConfig::default());
        assert_eq!(stats.input_correspondences, 2);
        assert_eq!(stats.connected_components, 1);
        assert_eq!(stats.retained_tracks, 1);
        assert_eq!(stats.retained_observations, 3);
    }

    #[test]
    fn build_tracks_drops_same_image_conflict() {
        // kp0 and kp1 of image 1 get merged into one component -> inconsistent.
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 1)],
            },
        ];
        let (tracks, stats) = build_tracks_with_stats(2, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
        assert_eq!(stats.input_correspondences, 2);
        assert_eq!(stats.connected_components, 1);
        assert_eq!(stats.conflicting_components, 1);
        assert_eq!(stats.conflicting_observations, 3);
        assert_eq!(stats.retained_tracks, 0);
        assert_eq!(stats.retained_observations, 0);
    }

    #[test]
    fn geometry_recovery_splits_conflict_from_trusted_multiview_poses() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        let kp_0_image_0 = keypoint_for_point(0, 0);
        let kp_1_image_1 = keypoint_for_point(1, 1);
        let pair_0_1 = pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap();
        // One erroneous bridge merges two otherwise-complete six-view tracks.
        pair_0_1.matches.push((kp_0_image_0, kp_1_image_1));

        let built = build_tracks_detailed(features.len(), &pairwise, 2);
        assert_eq!(built.stats.conflicting_components, 1);
        assert_eq!(built.conflicting_components.len(), 1);

        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            geometry_guided_conflict_recovery: true,
            conflict_recovery_min_views: 3,
            conflict_recovery_max_hypotheses: 32,
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            ..IncrementalSfmConfig::default()
        };
        let recovered = recover_conflict_tracks_geometry(
            &scene.camera,
            &features,
            &pairwise,
            &built.conflicting_components,
            &poses,
            &config,
        );
        assert_eq!(
            recovered.len(),
            1,
            "first slice keeps one guarded hypothesis"
        );
        let track = &recovered[0];
        assert!(track.registered_observations >= 3);
        assert!(track.mean_reprojection_px < 1e-6);
        let unique_images: HashSet<_> =
            track.observations.iter().map(|&(image, _)| image).collect();
        assert_eq!(unique_images.len(), track.observations.len());
        let nearest_truth = scene
            .points
            .iter()
            .take(2)
            .map(|point| (track.point - point).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest_truth < 1e-6,
            "recovered point error {nearest_truth}"
        );
    }

    #[test]
    fn geometry_recovery_rejects_three_view_chain_without_cycle() {
        let scene = build_scene();
        let (features, _) = render(&scene);
        let observation = |image: usize| {
            let kp = features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == 0.0)
                .unwrap();
            (image, kp)
        };
        let component = vec![observation(0), observation(1), observation(2)];
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(component[0].1, component[1].1)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(component[1].1, component[2].1)],
            },
        ];
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let recovered = recover_conflict_tracks_geometry(
            &scene.camera,
            &features,
            &pairwise,
            &[component],
            &poses,
            &IncrementalSfmConfig {
                conflict_recovery_max_reprojection_error_px: 0.1,
                conflict_recovery_max_mean_reprojection_px: 0.05,
                ..IncrementalSfmConfig::default()
            },
        );
        assert!(
            recovered.is_empty(),
            "a tree is not independent multi-view evidence"
        );
    }

    #[test]
    fn incremental_sfm_admits_geometry_recovery_only_after_clean_model() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        // Leave image 5 outside the verified component so the clean model is
        // intentionally incomplete and recovery may exercise its guarded BA.
        pairwise.retain(|pair| pair.image_i != 5 && pair.image_j != 5);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap()
            .matches
            .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));

        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            geometry_guided_conflict_recovery: true,
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            // Noise-free BA can move at floating-point epsilon around zero.
            conflict_recovery_max_clean_error_increase_ratio: 0.01,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert_eq!(result.track_build_stats.conflicting_components, 1);
        assert_eq!(result.geometry_recovered_tracks, 1);
        assert!(result.geometry_recovered_observations >= 3);
        assert!(result.geometry_recovery_pose_ba_applied);
        assert_eq!(result.registered_images, 5);
        assert!(result.mean_reprojection_px < 0.1);
    }

    #[test]
    fn complete_model_geometry_recovery_keeps_poses_byte_identical() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap()
            .matches
            .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));
        let base = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let control = incremental_sfm(&scene.camera, &features, &pairwise, &base).unwrap();
        let recovered = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                geometry_guided_conflict_recovery: true,
                conflict_recovery_max_reprojection_error_px: 0.1,
                conflict_recovery_max_mean_reprojection_px: 0.05,
                ..base
            },
        )
        .unwrap();
        assert_eq!(control.registered_images, features.len());
        assert_eq!(recovered.registered_images, features.len());
        assert_eq!(recovered.geometry_recovered_tracks, 1);
        assert!(!recovered.geometry_recovery_pose_ba_applied);
        assert_eq!(recovered.poses, control.poses);
        assert_eq!(recovered.tracks.len(), control.tracks.len() + 1);
    }

    #[test]
    fn rejected_geometry_recovery_rolls_back_byte_identical_clean_model() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap()
            .matches
            .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));

        let base = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let control = incremental_sfm(&scene.camera, &features, &pairwise, &base).unwrap();
        let rejected = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                geometry_guided_conflict_recovery: true,
                conflict_recovery_max_reprojection_error_px: 0.1,
                // Force the post-BA acceptance gate to reject every proposal.
                conflict_recovery_max_mean_reprojection_px: -1.0,
                ..base
            },
        )
        .unwrap();
        assert_eq!(rejected.geometry_recovered_tracks, 0);
        assert_eq!(rejected.geometry_recovered_observations, 0);
        assert_eq!(rejected.poses, control.poses);
        assert_eq!(rejected.tracks, control.tracks);
        assert_eq!(rejected.registered_images, control.registered_images);
        assert_eq!(rejected.mean_reprojection_px, control.mean_reprojection_px);
    }

    /// Minimal `FeatureSet`s with `kp_counts[i]` dummy keypoints per image —
    /// enough for `build_tracks_via_graph` to declare each image's point2D
    /// capacity; keypoint/descriptor content is irrelevant to track building.
    fn dummy_features(kp_counts: &[usize]) -> Vec<FeatureSet> {
        kp_counts
            .iter()
            .map(|&n| {
                let kps = vec![Point2::new(0.0, 0.0); n];
                let descs = vec![vec![0.0f32; 4]; n];
                FeatureSet::new(kps, descs).unwrap()
            })
            .collect()
    }

    /// M2: the [`TrackSource::CorrespondenceGraph`] path reproduces
    /// [`build_tracks_merges_shared_observations`] exactly.
    #[test]
    fn graph_tracks_merges_shared_observations() {
        let features = dummy_features(&[1, 1, 1]);
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
            },
        ];
        let tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert_eq!(tracks.len(), 1, "the chained matches form one track");
        assert_eq!(tracks[0].len(), 3, "track spans all three images");
    }

    /// M2: the [`TrackSource::CorrespondenceGraph`] path reproduces
    /// [`build_tracks_drops_same_image_conflict`] exactly — including the
    /// repeated-pair-entry input shape that exercises this function's
    /// pre-merge step (see `build_tracks_via_graph`'s doc).
    #[test]
    fn graph_tracks_drops_same_image_conflict() {
        let features = dummy_features(&[2, 2]);
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 1)],
            },
        ];
        let tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
    }

    /// M2 acceptance bar: on a repeated-pair input in *swapped* direction
    /// (`(1, 0)` instead of `(0, 1)`), the graph path's pre-merge
    /// canonicalization must still see both entries as the same unordered
    /// pair and produce the identical conflict-drop as
    /// [`graph_tracks_drops_same_image_conflict`] — proving the merge step
    /// doesn't silently drop the second entry via `DuplicatePair`.
    #[test]
    fn graph_tracks_drops_same_image_conflict_with_swapped_pair_direction() {
        let features = dummy_features(&[2, 2]);
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 0,
                matches: vec![(1, 0)], // (kp1 in image1, kp0 in image0)
            },
        ];
        let tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
    }

    #[test]
    fn visibility_pyramid_prefers_distribution_over_count() {
        // A small cluster of MANY points in one corner versus FEWER points spread
        // across the frame: the COLMAP visibility score must rank the spread set
        // higher (better-conditioned PnP), even though it has fewer correspondences.
        let (w, h) = (640u32, 480u32);
        let clustered: Vec<Point2<f64>> = (0..50)
            .map(|i| Point2::new(2.0 + (i % 5) as f64, 2.0 + (i / 5) as f64))
            .collect();
        let spread: Vec<Point2<f64>> = (0..5)
            .flat_map(|gy| {
                (0..4).map(move |gx| {
                    Point2::new(
                        (gx as f64 + 0.5) * w as f64 / 4.0,
                        (gy as f64 + 0.5) * h as f64 / 5.0,
                    )
                })
            })
            .collect();
        let clustered_score = visibility_pyramid_score(w, h, clustered.iter().copied());
        let spread_score = visibility_pyramid_score(w, h, spread.iter().copied());
        assert!(
            spread_score > clustered_score,
            "spread ({spread_score}, {} pts) should beat clustered ({clustered_score}, {} pts)",
            spread.len(),
            clustered.len(),
        );
        // The 50 clustered points collapse onto a handful of cells (occupancy
        // saturates), unlike a raw count which would have ranked them first.
        assert!(
            clustered_score < clustered.len(),
            "clustered occupancy {clustered_score} must saturate below the point count"
        );

        let camera = Camera::pinhole(0, w, h, 500.0, 500.0, 320.0, 240.0);
        let to_corrs = |points: &[Point2<f64>]| {
            points
                .iter()
                .copied()
                .map(|point2d| Correspondence2D3D {
                    point2d,
                    point3d: Point3::new(0.0, 0.0, 5.0),
                    confidence: None,
                })
                .collect::<Vec<_>>()
        };
        let clustered_corrs = to_corrs(&clustered);
        let spread_corrs = to_corrs(&spread);
        assert!(
            next_image_rank(&camera, NextImagePolicy::VisibilityPyramid, &spread_corrs,)
                > next_image_rank(
                    &camera,
                    NextImagePolicy::VisibilityPyramid,
                    &clustered_corrs,
                ),
            "visibility policy must prefer coverage"
        );
        assert!(
            next_image_rank(
                &camera,
                NextImagePolicy::CorrespondenceCount,
                &clustered_corrs,
            ) > next_image_rank(&camera, NextImagePolicy::CorrespondenceCount, &spread_corrs,),
            "count policy must reproduce the legacy ordering"
        );
    }

    #[test]
    fn reconstructs_synthetic_ring_scene() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        assert!(pairwise.len() >= 5, "expected an overlapping view graph");

        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();

        // Most images register and most points triangulate.
        assert!(
            result.registered_images >= 5,
            "registered only {}",
            result.registered_images
        );
        assert!(
            result.tracks.len() >= 20,
            "triangulated only {} tracks",
            result.tracks.len()
        );
        // Reprojection is tight (synthetic, noise-free).
        assert!(
            result.mean_reprojection_px < 1.0,
            "mean reprojection {} px too high",
            result.mean_reprojection_px
        );

        // The reconstruction is correct up to a similarity transform. Check the
        // recovered camera-center geometry matches GT up to scale by comparing
        // pairwise center-distance ratios between two registered images.
        let registered: Vec<usize> = (0..scene.poses.len())
            .filter(|&i| result.poses[i].is_some())
            .collect();
        assert!(registered.len() >= 3);
        let center = |i: usize| {
            result.poses[i]
                .as_ref()
                .unwrap()
                .camera_to_world()
                .translation
        };
        let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
        let (a, b, c) = (registered[0], registered[1], registered[2]);
        let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
        let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
        assert!(
            (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
            "camera-spacing ratio {est_ratio} != GT {gt_ratio} (similarity-invariant)"
        );
    }

    /// Look-at world→camera poses on an arc of `n` cameras at `radius` from
    /// `target`, spanning `span` radians (so neighbours keep a real baseline).
    fn arc_cameras(n: usize, target: Point3<f64>, radius: f64, span: f64) -> Vec<Pose> {
        let mut poses = Vec::new();
        let denom = (n.max(2) - 1) as f64;
        for k in 0..n {
            let angle = -span / 2.0 + span * (k as f64) / denom;
            let cam_center =
                target + Vector3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            let forward = (target - cam_center).normalize();
            let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
            let up = right.cross(&forward);
            let r_c2w = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let q_c2w = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r_c2w),
            );
            let q_w2c = q_c2w.inverse();
            let t_w2c = -(q_w2c * cam_center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }
        poses
    }

    /// A scene with two geometrically disjoint components: a small dense "trap"
    /// cluster (3 cameras, ~100 co-visible points → the *strongest-match* pairs in
    /// the whole graph) far to one side, and a larger "main" component (8 cameras
    /// over a grid). The trap's frustums never see the main grid and vice versa,
    /// so they form two connected components; the strongest seed reconstructs only
    /// the 3-camera trap, and recovering the main component needs the multi-seed
    /// search to look past it. Cameras: indices 0..3 trap, 3..11 main.
    fn build_two_component_scene() -> Scene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        // Trap cluster: a dense cube at the origin (every trap camera sees all of
        // it, so each trap pair carries the most matches).
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in -2..=1 {
                    points.push(Point3::new(
                        xi as f64 * 0.2,
                        yi as f64 * 0.2,
                        zi as f64 * 0.2,
                    ));
                }
            }
        }
        // Main grid: a separate, larger structure offset far along +x.
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in 0..=2 {
                    points.push(Point3::new(
                        20.0 + xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.3,
                    ));
                }
            }
        }
        let mut poses = arc_cameras(3, Point3::origin(), 3.0, 0.5);
        poses.extend(arc_cameras(8, Point3::new(20.0, 0.0, 0.0), 3.0, 1.2));
        Scene {
            camera,
            points,
            poses,
        }
    }

    #[test]
    fn multi_seed_escapes_strongest_isolated_cluster() {
        let scene = build_two_component_scene();
        let (features, pairwise) = render(&scene);

        // The strongest-match pair is inside the 3-camera trap.
        let strongest = pairwise
            .iter()
            .max_by_key(|p| p.matches.len())
            .expect("a view graph");
        assert!(
            strongest.image_i < 3 && strongest.image_j < 3,
            "expected the densest pair to be inside the trap cluster, got ({},{})",
            strongest.image_i,
            strongest.image_j
        );

        // One trial commits to that strongest seed and is trapped in the cluster.
        let trapped = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                min_seed_matches: 8,
                min_pnp_inliers: 6,
                seed_trials: 1,
                ..IncrementalSfmConfig::default()
            },
        )
        .unwrap();
        assert!(
            trapped.registered_images <= 3,
            "single-seed should be stuck in the 3-camera trap, got {}",
            trapped.registered_images
        );

        // The multi-seed search looks past the trap and recovers the 8-camera
        // main component instead.
        let escaped = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                min_seed_matches: 8,
                min_pnp_inliers: 6,
                ..IncrementalSfmConfig::default() // seed_trials = 12
            },
        )
        .unwrap();
        assert!(
            escaped.registered_images >= 7,
            "multi-seed should recover the 8-camera main component, got {}",
            escaped.registered_images
        );
    }

    type OutlierTrackFixture = (
        Camera,
        Vec<FeatureSet>,
        Vec<Option<Pose>>,
        Vec<Vec<(usize, usize)>>,
        Vec<Option<Point3<f64>>>,
    );

    /// Build three views (identity rotation, small lateral offsets) of one world
    /// point, with `outlier_views` images observing it at a planted off-by-50px
    /// outlier keypoint instead of the true projection.
    fn outlier_track_fixture(outlier_views: &[usize]) -> OutlierTrackFixture {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.1, -0.2, 5.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.3 - 0.3, 0.0, 0.0),
            );
            let mut px = camera.project(&pose.transform_world_point(&point)).unwrap();
            if outlier_views.contains(&k) {
                px += Vector3::new(50.0, 50.0, 0.0).xy();
            }
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let track_point = vec![Some(point)];
        (camera, features, poses, tracks, track_point)
    }

    #[test]
    fn filter_strips_single_outlier_observation_keeps_track() {
        let (camera, features, poses, mut tracks, mut track_point) = outlier_track_fixture(&[2]);
        let config = IncrementalSfmConfig::default();
        let removed = filter_outlier_observations(
            &camera,
            &features,
            &mut tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(removed, 1, "the planted outlier observation is removed");
        assert_eq!(
            tracks[0],
            vec![(0, 0), (1, 0)],
            "only the two inliers remain"
        );
        assert!(track_point[0].is_some(), "track survives with >= 2 inliers");
    }

    #[test]
    fn filter_drops_low_parallax_far_point() {
        // A point 500 units away, seen by three cameras 0.6 units apart, projects
        // with ZERO reprojection error (perfect) yet has ~0.07 deg parallax — the
        // depth-ambiguous far-flung outlier the reprojection test cannot catch.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.0, 0.0, 500.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.3 - 0.3, 0.0, 0.0),
            );
            let px = camera.project(&pose.transform_world_point(&point)).unwrap();
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let mut tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let mut track_point = vec![Some(point)];
        let config = IncrementalSfmConfig::default(); // min_triangulation_angle_deg = 2.0
        let changed = filter_outlier_observations(
            &camera,
            &features,
            &mut tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(changed, 1, "the low-parallax track is dropped");
        assert!(
            track_point[0].is_none(),
            "depth-ambiguous far point dropped despite zero reprojection error"
        );
    }

    #[test]
    fn filter_drops_track_below_min_observations() {
        // Two of three views are outliers -> a single inlier left -> drop track.
        let (camera, features, poses, mut tracks, mut track_point) = outlier_track_fixture(&[1, 2]);
        let config = IncrementalSfmConfig::default();
        let removed = filter_outlier_observations(
            &camera,
            &features,
            &mut tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(removed, 3, "2 observations stripped + 1 track dropped");
        assert!(
            track_point[0].is_none(),
            "track with < 2 inlier observations is dropped"
        );
    }

    #[test]
    fn filter_images_deregisters_unsupported_pose_and_protects_seed() {
        // Register all six ring cameras with their true poses, triangulate, then
        // corrupt one non-seed image's pose so none of its observations reproject.
        // FilterImages must de-register exactly that image, keep the well-supported
        // ones, and never touch the two seed (lowest-index) images.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let config = IncrementalSfmConfig {
            filter_images: true,
            filter_min_image_observations: 5,
            ..IncrementalSfmConfig::default()
        };
        let mut poses: Vec<Option<Pose>> = scene.poses.iter().map(|p| Some(p.clone())).collect();
        let mut track_point = vec![None; tracks.len()];
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );

        // Corrupt image 3 (not in the protected seed pair): aim it away from the
        // cloud so every observation reprojects far off or behind the camera.
        let bad = Pose::from_world_to_camera(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI),
            Vector3::new(0.0, 0.0, 0.0),
        );
        poses[3] = Some(bad);

        let removed = filter_images(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses,
            &track_point,
        );
        assert_eq!(removed, 1, "only the unsupported image is de-registered");
        assert!(poses[3].is_none(), "the corrupted-pose image is filtered");
        assert!(
            poses[0].is_some() && poses[1].is_some(),
            "the seed pair is protected from filtering"
        );
        assert!(
            poses[2].is_some() && poses[4].is_some() && poses[5].is_some(),
            "well-supported images stay registered"
        );
    }

    #[test]
    fn retriangulate_completes_untriangulated_track() {
        // Three identity-rotation views with a real lateral baseline see one
        // world point. The track exists in the union-find but was never given a
        // 3D point (it failed the parallax gate at growth time, say). With the
        // poses now fixed, re-triangulation must complete it.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.1, -0.2, 5.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.5 - 0.5, 0.0, 0.0),
            );
            let px = camera.project(&pose.transform_world_point(&point)).unwrap();
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let mut track_point: Vec<Option<Point3<f64>>> = vec![None]; // not yet triangulated
        let config = IncrementalSfmConfig::default();
        let changed = retriangulate_tracks(
            &camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(changed, 1, "the un-triangulated track is completed");
        let p = track_point[0].expect("track now has a 3D point");
        assert!(
            (p - point).norm() < 1e-6,
            "re-triangulated point {p:?} should recover the true point {point:?}"
        );
    }

    #[test]
    fn retriangulate_guarded_swap_replaces_noisy_point_only_when_better() {
        // Same three-view geometry, but the track already carries a *noisy* point
        // displaced far along the depth ray. Re-triangulation from the true
        // observations fits them better, so the guarded swap must replace it; a
        // second pass (now exact) must be a no-op (never regress an exact point).
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.1, -0.2, 5.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.5 - 0.5, 0.0, 0.0),
            );
            let px = camera.project(&pose.transform_world_point(&point)).unwrap();
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let noisy = Point3::new(0.3, -0.6, 8.0);
        let mut track_point = vec![Some(noisy)];
        let config = IncrementalSfmConfig::default();

        let changed = retriangulate_tracks(
            &camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(changed, 1, "the noisy point is replaced by a better fit");
        let p = track_point[0].unwrap();
        assert!(
            (p - point).norm() < 1e-6,
            "guarded swap should land on the true point, got {p:?}"
        );

        // Re-running on the now-exact point changes nothing.
        let again = retriangulate_tracks(
            &camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(again, 0, "an already-exact point must not be regressed");
    }

    #[test]
    fn colmap_style_mapper_reconstructs_ring_scene() {
        // The COLMAP schedule (per-registration local BA + growth-triggered
        // iterative global refinement + registration retries) must reconstruct the
        // synthetic ring at least as completely as the simple schedule, with tight
        // reprojection and a similarity-correct camera geometry.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            colmap_style_mapper: true,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert!(
            result.registered_images >= 5,
            "registered only {}",
            result.registered_images
        );
        assert!(
            result.tracks.len() >= 20,
            "triangulated only {} tracks",
            result.tracks.len()
        );
        assert!(
            result.mean_reprojection_px < 1.0,
            "mean reprojection {} px too high",
            result.mean_reprojection_px
        );
        let registered: Vec<usize> = (0..scene.poses.len())
            .filter(|&i| result.poses[i].is_some())
            .collect();
        let center = |i: usize| {
            result.poses[i]
                .as_ref()
                .unwrap()
                .camera_to_world()
                .translation
        };
        let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
        let (a, b, c) = (registered[0], registered[1], registered[2]);
        let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
        let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
        assert!(
            (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
            "camera-spacing ratio {est_ratio} != GT {gt_ratio} (similarity-invariant)"
        );
    }

    #[test]
    fn colmap_style_co_evolves_intrinsics_toward_truth() {
        // The synthetic ring is observable geometry, so a focal error is
        // recoverable. Render with the TRUE camera (fx=fy=500) but reconstruct from
        // a WRONG horizontal focal (fx=530). The arc moves the cameras only in the
        // x-z plane, so the *horizontal* focal fx is well constrained by the
        // azimuthal parallax (fy would need elevation change — exercised instead by
        // the anisotropic South-Building benchmark). The joint solve must pull fx
        // substantially back toward 500 — the COLMAP self-calibration formulation
        // (intrinsics co-estimated inside the Schur camera system, using the coupled
        // landmark-eliminated gradient), which a final-only alternating refinement
        // against converged structure cannot do. The orthogonal, un-perturbed
        // vertical axis (fy, cy) must stay fixed. The horizontal principal point cx
        // is allowed to co-adjust: on a pure look-at arc fx and cx are only *jointly*
        // constrained (the focal/principal-point ambiguity), so the joint solve
        // legitimately distributes the correction across both — this confound is
        // absent on the richer South-Building viewpoints, where cx stays put.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let wrong = Camera::pinhole(0, 640, 480, 530.0, 500.0, 320.0, 240.0);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            colmap_style_mapper: true,
            refine_intrinsics: true,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&wrong, &features, &pairwise, &config).unwrap();
        let cam = result
            .refined_camera
            .expect("refine_intrinsics returns the refined camera");
        let (fx, fy, cx, cy) = cam.intrinsics().unwrap();
        eprintln!("joint-refined intrinsics: fx {fx} fy {fy} cx {cx} cy {cy}");
        // fx recovers at least a third of its injected error (530 - 500 = 30).
        assert!(
            fx < 530.0 - 0.33 * 30.0,
            "fx {fx} should recover substantially toward 500 from 530"
        );
        // The orthogonal vertical axis was not perturbed and must not drift.
        assert!((fy - 500.0).abs() < 2.0, "fy {fy} drifted from 500");
        assert!((cy - 240.0).abs() < 2.0, "cy {cy} drifted from 240");
        // cx co-adjusts with fx within the look-at arc's focal/centre ambiguity, but
        // must stay sane (no blow-up).
        assert!((cx - 320.0).abs() < 20.0, "cx {cx} blew up from 320");
    }

    #[test]
    fn colmap_style_mapper_retries_a_filtered_image_up_to_its_trial_budget_then_gives_up() {
        // M4 (`docs/colmap_port_plan.md`'s "M4 results"): the growth loop's
        // stall-triggered recovery must give a `filter_images`-demoted image
        // genuine retry attempts across multiple growth stalls — not filter it
        // once and abandon it, the pre-M4 behaviour, since pre-M4
        // `growth_global_refinement` (and the `filter_images` call inside it)
        // only ever ran on the growth-*ratio* trigger, never on a stall — while
        // still terminating cleanly once `max_registration_trials` is spent,
        // rather than cycling forever. `global_ba_images_ratio` is set absurdly
        // high so the *only* thing that can ever invoke `growth_global_refinement`
        // / `filter_images` in this test is the stall path, isolating exactly
        // the mechanism this milestone added.
        //
        // Scene: 4 cameras looking at the same 40-point cloud (two z-layers so
        // the essential-matrix seed estimator sees a non-degenerate, non-planar
        // point set). The seed pair (0, 1) and camera 3 all see all 40 points;
        // camera 2 is built (by construction, not by frustum geometry) to see
        // only the first 15 — enough to clear `min_pnp_inliers` and register,
        // but below `filter_min_image_observations` (16), so every time
        // `filter_images` runs it demotes camera 2 and nothing else (the seed
        // pair is exempt from filtering by construction, and camera 3 is
        // well-supported). A 4th, well-supported camera is needed because
        // `filter_images` refuses to drop *anyone* once the registered count
        // is already at its floor of 3 (`incremental_sfm.rs`'s
        // `filter_images`: `if remaining <= 3 { continue; }`) — with only 3
        // total cameras, camera 2 could never be filtered no matter how weak
        // its support, which would make this test vacuous. Since camera 2's
        // supporting-observation count can never improve (it structurally
        // only ever sees 15 points), this is a fixed point: register, demote,
        // retry, register, demote, … — bounded only by
        // `max_registration_trials`. Never resetting `trials` on the stall
        // (see `grow_from_seed`'s module-level doc on `stalled_once`) is what
        // makes this terminate at all instead of cycling indefinitely.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for xi in -2..=2 {
            for yi in -1..=2 {
                for zi in 0..=1 {
                    points.push(Point3::new(
                        xi as f64 * 0.25,
                        yi as f64 * 0.25,
                        1.0 + zi as f64 * 0.3,
                    ));
                }
            }
        }
        assert_eq!(points.len(), 40, "test fixture must have exactly 40 points");
        let poses = arc_cameras(4, Point3::origin(), 3.0, 0.6);

        // Camera 2 ("the weak straggler") only ever observes the first 15 of
        // the 40 points; cameras 0, 1 (the seed pair) and 3 observe all 40.
        let mut features = Vec::new();
        let mut visible: Vec<HashMap<usize, usize>> = Vec::new();
        for (cam_idx, pose) in poses.iter().enumerate() {
            let n_visible = if cam_idx == 2 { 15 } else { points.len() };
            let mut kps = Vec::new();
            let mut descs = Vec::new();
            let mut vis = HashMap::new();
            for (pidx, p) in points.iter().enumerate().take(n_visible) {
                let px = project(&camera, pose, p)
                    .expect("fixture point must project in front of every camera");
                vis.insert(pidx, kps.len());
                kps.push(px);
                descs.push(vec![pidx as f32, 1.0, 0.0, 0.0]);
            }
            features.push(FeatureSet::new(kps, descs).unwrap());
            visible.push(vis);
        }

        let n_cams = poses.len();
        let mut pairwise = Vec::new();
        for i in 0..n_cams {
            for j in (i + 1)..n_cams {
                let mut matches = Vec::new();
                for (pidx, &ki) in &visible[i] {
                    if let Some(&kj) = visible[j].get(pidx) {
                        matches.push((ki, kj));
                    }
                }
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                    });
                }
            }
        }

        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 8,
            colmap_style_mapper: true,
            filter_images: true,
            filter_min_image_observations: 16,
            global_ba_images_ratio: 1000.0,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&camera, &features, &pairwise, &config).unwrap();

        assert_eq!(
            result.registered_images, 3,
            "camera 2 can never clear filter_min_image_observations=16 with its \
             fixed 15 supporting observations, so it must end up excluded, not \
             stuck mid-retry or wrongly kept, leaving the other 3 cameras registered"
        );
        assert!(
            result.poses[0].is_some() && result.poses[1].is_some(),
            "the seed pair stays registered (protected from filter_images)"
        );
        assert!(
            result.poses[2].is_none(),
            "the weakly-supported straggler ends up filtered, not registered"
        );
        assert!(
            result.poses[3].is_some(),
            "the well-supported 4th camera stays registered"
        );
    }

    #[test]
    fn colmap_style_mapper_is_deterministic_across_repeated_runs() {
        // M4 regression pin: multi-seed search (`seed_trials`) and the new
        // stall-triggered recovery must stay fully deterministic (fixed PnP
        // RANSAC seed, no reset-driven or iteration-order-driven nondeterminism)
        // — running the identical config against the identical view graph twice
        // must produce byte-identical registered counts, track counts, and mean
        // reprojection error. Uses `build_two_component_scene` (multiple seed
        // candidates, one of them a trap) with `colmap_style_mapper` on so both
        // the multi-seed sweep and the stall-recovery path are exercised.
        let scene = build_two_component_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            colmap_style_mapper: true,
            ..IncrementalSfmConfig::default()
        };
        let a = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        let b = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert_eq!(a.registered_images, b.registered_images);
        assert_eq!(a.tracks.len(), b.tracks.len());
        assert_eq!(
            a.mean_reprojection_px.to_bits(),
            b.mean_reprojection_px.to_bits(),
            "mean reprojection error must be bit-identical across repeated runs"
        );
        for i in 0..scene.poses.len() {
            assert_eq!(
                a.poses[i].is_some(),
                b.poses[i].is_some(),
                "image {i}'s registration outcome must be identical across runs"
            );
        }
    }

    #[test]
    fn repeated_bundle_adjustment_does_not_collapse_scale() {
        // A monocular reconstruction has a free scale gauge; without anchoring it
        // a second BA from the converged state collapses the reconstruction.
        // run_bundle_adjustment fixes a second (farthest) pose to pin scale, so
        // re-optimising must be stable — track refinement relies on this.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            track_filter_iterations: 4,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        // A scale collapse manifests as nearly all tracks dropping out (the EuRoC
        // symptom was 630 -> 1) and the camera geometry degenerating; with the
        // gauge anchored, structure and registration survive four BA rounds.
        assert!(
            result.registered_images >= 5,
            "registration must survive repeated BA, got {}",
            result.registered_images
        );
        assert!(
            result.tracks.len() >= 20,
            "structure must survive repeated BA, got {} tracks",
            result.tracks.len()
        );
        assert!(
            result.mean_reprojection_px < 1.0,
            "reprojection {} px too high after repeated BA",
            result.mean_reprojection_px
        );
        // Camera-spacing ratio stays similarity-correct (a collapse would warp it).
        let registered: Vec<usize> = (0..scene.poses.len())
            .filter(|&i| result.poses[i].is_some())
            .collect();
        let center = |i: usize| {
            result.poses[i]
                .as_ref()
                .unwrap()
                .camera_to_world()
                .translation
        };
        let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
        let (a, b, c) = (registered[0], registered[1], registered[2]);
        let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
        let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
        assert!(
            (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
            "camera geometry warped after repeated BA: {est_ratio} vs GT {gt_ratio}"
        );
    }

    #[test]
    fn structureless_local_bundle_keeps_registered_boundary_poses_exactly_fixed() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let config = IncrementalSfmConfig::default();
        let mut poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let mut track_point = vec![None; tracks.len()];
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );

        // Mimic a slightly inaccurate multi-neighbour structure-less proposal.
        // Only image 2 is allowed to move during the admission refinement.
        let truth = scene.poses[2].clone();
        poses[2].as_mut().unwrap().world_to_camera.translation += Vector3::new(0.03, -0.02, 0.01);
        let before = poses.clone();
        let mut variable = HashSet::new();
        variable.insert(2usize);
        bundle_adjust_local(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses,
            &mut track_point,
            &variable,
        )
        .expect("fixed-boundary structure-less BA should converge");

        for image in [0usize, 1, 3, 4, 5] {
            assert_eq!(
                poses[image], before[image],
                "registered boundary pose {image} must remain byte-for-byte unchanged"
            );
        }
        let error_before = (before[2].as_ref().unwrap().matrix() - truth.matrix()).norm();
        let error_after = (poses[2].as_ref().unwrap().matrix() - truth.matrix()).norm();
        assert!(
            error_after < error_before,
            "recovered pose should improve while its registered boundary stays fixed: \
             {error_before} -> {error_after}"
        );
    }

    #[test]
    fn structureless_fixed_pose_submap_refines_only_new_landmarks() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let config = IncrementalSfmConfig::default();
        let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let mut track_point = vec![None; tracks.len()];
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        let new_track = tracks
            .iter()
            .position(|track| track.iter().any(|&(image, _)| image == 2))
            .unwrap();
        let truth = track_point[new_track].unwrap();
        track_point[new_track] = Some(truth + Vector3::new(0.08, -0.05, 0.12));
        let points_before = track_point.clone();
        let mut preexisting = vec![true; track_point.len()];
        preexisting[new_track] = false;

        refine_structureless_new_landmarks(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
            2,
            &preexisting,
        )
        .expect("fixed-pose local submap should converge");

        for track_id in 0..track_point.len() {
            if track_id != new_track {
                assert_eq!(track_point[track_id], points_before[track_id]);
            }
        }
        assert!(
            (track_point[new_track].unwrap() - truth).norm()
                < (points_before[new_track].unwrap() - truth).norm()
        );
    }

    #[test]
    fn structureless_local_tracks_use_consensus_edges_and_unowned_observations() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let missing = 0usize;
        let missing_center = scene.poses[missing].camera_center_world();
        let missing_rotation = scene.poses[missing].world_to_camera.rotation;
        let constraints: Vec<_> = [1usize, 2, 3]
            .into_iter()
            .map(|neighbor| {
                structureless_constraint(
                    neighbor,
                    scene.poses[neighbor].camera_center_world(),
                    missing_center,
                    missing_rotation,
                )
            })
            .collect();
        let proposal = StructurelessPoseProposal {
            pose: scene.poses[missing].clone(),
            neighbor_spread: 1.0,
            line_error_ratio: 0.0,
            consensus_indices: vec![0, 1, 2],
        };
        let local = build_structureless_local_tracks(
            &scene.camera,
            &features,
            &pairwise,
            &[],
            &[],
            &poses,
            missing,
            &constraints,
            &proposal,
            &IncrementalSfmConfig::default(),
        );
        assert!(!local.is_empty());
        for (track, point) in local {
            assert!(track.len() >= 2);
            assert_eq!(
                track.iter().filter(|(image, _)| *image == missing).count(),
                1
            );
            let unique_images: HashSet<_> = track.iter().map(|(image, _)| *image).collect();
            assert_eq!(unique_images.len(), track.len());
            for &(image, keypoint) in &track {
                let error = reprojection_error_px(
                    &scene.camera,
                    poses[image].as_ref().unwrap(),
                    &point,
                    &features[image].keypoints[keypoint],
                )
                .unwrap();
                assert!(error <= 2.0);
            }
        }
    }

    fn structureless_constraint(
        neighbor: usize,
        neighbor_center: Point3<f64>,
        missing_center: Point3<f64>,
        missing_rotation: UnitQuaternion<f64>,
    ) -> StructurelessConstraint {
        StructurelessConstraint {
            neighbor,
            neighbor_center,
            missing_rotation,
            center_direction: (missing_center - neighbor_center).normalize(),
            weight: 100.0 - neighbor as f64,
        }
    }

    #[test]
    fn structureless_multineighbor_lines_recover_scaled_camera_pose() {
        let missing_center = Point3::new(1.2, -0.4, 3.5);
        let rotation = UnitQuaternion::from_euler_angles(0.05, -0.12, 0.08);
        let constraints = vec![
            structureless_constraint(0, Point3::new(-1.0, 0.0, 0.0), missing_center, rotation),
            structureless_constraint(1, Point3::new(2.0, 0.5, 0.2), missing_center, rotation),
            structureless_constraint(2, Point3::new(0.0, -2.0, 0.4), missing_center, rotation),
        ];
        let config = IncrementalSfmConfig {
            structureless_min_intersection_angle_deg: 1.0,
            structureless_max_center_line_error_ratio: 1e-8,
            ..IncrementalSfmConfig::default()
        };
        let proposal = solve_structureless_pose(&constraints, &config).unwrap();
        assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
        assert!((proposal.pose.world_to_camera.rotation.inverse() * rotation).angle() < 1e-12);
        assert!(proposal.line_error_ratio < 1e-9);
    }

    #[test]
    fn structureless_pose_interpolation_uses_camera_centers_and_slerp() {
        let from_center = Point3::new(-1.0, 0.5, 2.0);
        let to_center = Point3::new(3.0, -0.5, 4.0);
        let from_rotation = UnitQuaternion::identity();
        let to_rotation = UnitQuaternion::from_euler_angles(0.0, 0.4, 0.0);
        let from = Pose::from_world_to_camera(
            from_rotation,
            -from_rotation.transform_vector(&from_center.coords),
        );
        let to = Pose::from_world_to_camera(
            to_rotation,
            -to_rotation.transform_vector(&to_center.coords),
        );
        let midpoint = interpolate_structureless_pose(&from, &to, 0.5);
        let expected_center = Point3::from((from_center.coords + to_center.coords) * 0.5);
        assert!((midpoint.camera_center_world() - expected_center).norm() < 1e-12);
        let expected_rotation = from_rotation.slerp(&to_rotation, 0.5);
        assert!((midpoint.world_to_camera.rotation.inverse() * expected_rotation).angle() < 1e-12);
        assert_eq!(interpolate_structureless_pose(&from, &to, 0.0), from);
        assert_eq!(interpolate_structureless_pose(&from, &to, 1.0), to);
    }

    #[test]
    fn structureless_pose_rejects_single_neighbor_arbitrary_scale() {
        let missing_center = Point3::new(0.0, 0.0, 3.0);
        let constraints = vec![structureless_constraint(
            0,
            Point3::origin(),
            missing_center,
            UnitQuaternion::identity(),
        )];
        assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
    }

    #[test]
    fn structureless_pose_rejects_rotation_disagreement() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let constraints = vec![
            structureless_constraint(
                0,
                Point3::new(-1.0, 0.0, 0.0),
                missing_center,
                UnitQuaternion::identity(),
            ),
            structureless_constraint(
                1,
                Point3::new(1.0, 0.0, 0.0),
                missing_center,
                UnitQuaternion::from_euler_angles(0.0, 0.2, 0.0),
            ),
        ];
        assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
    }

    #[test]
    fn structureless_pose_uses_largest_rotation_consensus() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let good_rotation = UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03);
        let mut constraints = vec![
            structureless_constraint(
                0,
                Point3::new(-1.0, 0.0, 0.0),
                missing_center,
                good_rotation,
            ),
            structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, good_rotation),
            structureless_constraint(
                2,
                Point3::new(0.0, -1.0, 0.0),
                missing_center,
                good_rotation,
            ),
            structureless_constraint(
                3,
                Point3::new(0.0, 1.0, 0.0),
                missing_center,
                UnitQuaternion::from_euler_angles(0.0, 0.8, 0.0),
            ),
        ];
        constraints[3].weight = 1000.0;
        let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
            .expect("three coherent rotations must outvote one high-support outlier");
        assert_eq!(proposal.consensus_indices.len(), 3);
        assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
        assert!((proposal.pose.world_to_camera.rotation.inverse() * good_rotation).angle() < 1e-12);
    }

    #[test]
    fn structureless_pose_keeps_rotation_consensus_centre_not_strongest_edge() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let centre_rotation = UnitQuaternion::identity();
        let positive_edge = UnitQuaternion::from_euler_angles(0.0, 2.5f64.to_radians(), 0.0);
        let negative_edge = UnitQuaternion::from_euler_angles(0.0, -2.5f64.to_radians(), 0.0);
        let mut constraints = vec![
            structureless_constraint(
                0,
                Point3::new(-1.0, 0.0, 0.0),
                missing_center,
                centre_rotation,
            ),
            structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, positive_edge),
            structureless_constraint(
                2,
                Point3::new(0.0, -1.0, 0.0),
                missing_center,
                negative_edge,
            ),
        ];
        constraints[1].weight = 1000.0;
        let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
            .expect("the centre edge supports a valid three-rotation consensus");
        assert!(
            (proposal.pose.world_to_camera.rotation.inverse() * centre_rotation).angle() < 1e-12,
            "the high-weight +2.5deg edge would be 5deg from the negative edge"
        );
        let consistency = structureless_pose_consistency(
            &proposal.pose,
            &constraints,
            &proposal,
            &IncrementalSfmConfig::default(),
        );
        assert!(consistency.accepted);
        assert!(consistency.max_rotation_deg <= 3.0);
    }

    #[test]
    fn structureless_pose_uses_robust_translation_consensus() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let rotation = UnitQuaternion::identity();
        let mut constraints = vec![
            structureless_constraint(0, Point3::new(-1.0, 0.0, 0.0), missing_center, rotation),
            structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, rotation),
            structureless_constraint(2, Point3::new(0.0, -1.0, 0.0), missing_center, rotation),
            StructurelessConstraint {
                neighbor: 3,
                neighbor_center: Point3::new(0.0, 1.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::x(),
                weight: 1000.0,
            },
        ];
        constraints.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        let config = IncrementalSfmConfig {
            structureless_max_center_line_error_ratio: 0.01,
            ..IncrementalSfmConfig::default()
        };
        let proposal = solve_structureless_pose(&constraints, &config)
            .expect("three coherent directions must reject one high-support translation outlier");
        assert_eq!(proposal.consensus_indices.len(), 3);
        assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
    }

    #[test]
    fn structureless_pose_reclassifies_lines_after_weighted_refit() {
        let rotation = UnitQuaternion::identity();
        let mut constraints = vec![
            StructurelessConstraint {
                neighbor: 0,
                neighbor_center: Point3::new(-1.0, 0.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::x(),
                weight: 100.0,
            },
            StructurelessConstraint {
                neighbor: 1,
                neighbor_center: Point3::new(0.0, -1.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::y(),
                weight: 100.0,
            },
            StructurelessConstraint {
                neighbor: 2,
                neighbor_center: Point3::new(0.0, 1.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::new(0.1, -1.0, 0.0).normalize(),
                weight: 1000.0,
            },
            // This short directed baseline agrees with the winning pairwise
            // hypothesis at the origin, but the high-weight tilted line moves
            // the least-squares refit behind it. It must be reclassified as an
            // outlier instead of vetoing the other three consistent lines.
            StructurelessConstraint {
                neighbor: 3,
                neighbor_center: Point3::new(0.01, 0.0, 0.0),
                missing_rotation: rotation,
                center_direction: -Vector3::x(),
                weight: 100.0,
            },
        ];
        constraints.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
            .expect("a marginal directed line must not veto a stable 3-line refit");
        assert_eq!(proposal.consensus_indices.len(), 3);
        assert!(proposal
            .consensus_indices
            .iter()
            .all(|&index| constraints[index].neighbor != 3));
        assert!(
            structureless_pose_consistency(
                &proposal.pose,
                &constraints,
                &proposal,
                &IncrementalSfmConfig::default(),
            )
            .accepted
        );
    }

    #[test]
    fn structureless_pose_rejects_parallel_center_directions() {
        let constraints = vec![
            StructurelessConstraint {
                neighbor: 0,
                neighbor_center: Point3::new(0.0, 0.0, 0.0),
                missing_rotation: UnitQuaternion::identity(),
                center_direction: Vector3::z(),
                weight: 100.0,
            },
            StructurelessConstraint {
                neighbor: 1,
                neighbor_center: Point3::new(1.0, 0.0, 0.0),
                missing_rotation: UnitQuaternion::identity(),
                center_direction: Vector3::z(),
                weight: 90.0,
            },
        ];
        assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
    }

    /// Island-chain fixture. Ten arc cameras all observing one point cloud,
    /// with the verified pair graph pruned into a main component
    /// `{0, 1, 2, 3, 6, 7, 8, 9}` and a two-image island `{4, 5}` where the
    /// bridge image `5` has a *higher* index than its dependent `4`:
    ///
    /// - `4` pairs only with `{2, 3, 5}` — two registered neighbours while
    ///   `5` is unregistered, below [`IncrementalSfmConfig::
    ///   structureless_min_neighbors`];
    /// - `5` pairs with registered `{3, 6, 7}` plus the island partner `4`.
    ///
    /// Every island pair is narrow-baseline (adjacent arc steps) because the
    /// two-view essential estimate degrades on this synthetic cloud beyond
    /// ~0.4 rad of arc separation. Disjoint keypoint bands keep every
    /// island-touching union-find component at two images, below the
    /// track-length floor: the clean global model triangulates from
    /// main-component tracks only, leaving the island's observations free
    /// for local-submap synthesis — exactly the thin-per-image-structure
    /// regime the courtyard second component exposed.
    struct IslandScene {
        camera: Camera,
        poses: Vec<Pose>,
        features: Vec<FeatureSet>,
        pairwise: Vec<PairwiseMatches>,
    }

    fn build_island_scene() -> IslandScene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for xi in -3..=3 {
            for yi in -2..=2 {
                for zi in 0..=4 {
                    points.push(Point3::new(
                        xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.25,
                    ));
                }
            }
        }
        let mut poses = Vec::new();
        for k in 0..10 {
            let angle = -0.585 + k as f64 * 0.13;
            let radius = 3.0;
            let cam_center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            let forward = (Point3::origin() - cam_center).normalize();
            let world_up = Vector3::new(0.0, 1.0, 0.0);
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward);
            let r_cam_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let q_c2w = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r_cam_to_world),
            );
            let q_w2c = q_c2w.inverse();
            let t_w2c = -(q_w2c * cam_center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }

        // Every camera sees every point; keypoint index == point index.
        let features: Vec<FeatureSet> = poses
            .iter()
            .map(|pose| {
                let (kps, descs): (Vec<_>, Vec<_>) = points
                    .iter()
                    .enumerate()
                    .filter_map(|(pidx, p)| {
                        project(&camera, pose, p).map(|px| (px, vec![pidx as f32, 1.0, 0.0, 0.0]))
                    })
                    .unzip();
                FeatureSet::new(kps, descs).unwrap()
            })
            .collect();

        // Strided keypoint bands. Every band must mix points across all
        // three grid axes: a band confined to one grid slice is exactly
        // planar, and a planar correspondence set makes the two-view
        // essential estimate chirality-degenerate (the failure that
        // motivated this design).
        let all: Vec<usize> = (0..points.len()).collect();
        let main_points: Vec<usize> = all.iter().step_by(4).copied().collect();
        let remainder: Vec<usize> = {
            let main_set: HashSet<usize> = main_points.iter().copied().collect();
            all.into_iter().filter(|p| !main_set.contains(p)).collect()
        };
        let island_band =
            |k: usize| -> Vec<usize> { remainder.iter().skip(k).step_by(6).copied().collect() };
        let band_a = island_band(0);
        let band_b = island_band(1);
        let band_c = island_band(2);
        let band_d = island_band(3);
        let band_e = island_band(4);
        let band_f = island_band(5);

        let pair = |image_i: usize, image_j: usize, band: &[usize]| PairwiseMatches {
            image_i,
            image_j,
            matches: band.iter().map(|&p| (p, p)).collect(),
        };

        let mut pairwise = Vec::new();
        let main = [0usize, 1, 2, 3, 6, 7, 8, 9];
        for (a, &i) in main.iter().enumerate() {
            for &j in main.iter().skip(a + 1) {
                pairwise.push(pair(i, j, &main_points));
            }
        }
        pairwise.push(pair(4, 3, &band_a));
        pairwise.push(pair(4, 2, &band_b));
        pairwise.push(pair(4, 5, &band_c));
        pairwise.push(pair(5, 6, &band_d));
        pairwise.push(pair(5, 3, &band_e));
        pairwise.push(pair(5, 7, &band_f));

        IslandScene {
            camera,
            poses,
            features,
            pairwise,
        }
    }

    #[test]
    fn structureless_rounds_chain_an_island_through_a_higher_indexed_bridge() {
        let scene = build_island_scene();
        let min_track_length = 5;
        let mut tracks = build_tracks(scene.features.len(), &scene.pairwise, min_track_length);
        assert!(!tracks.is_empty(), "fixture sanity: main tracks must form");
        for track in &tracks {
            let images: HashSet<usize> = track.iter().map(|&(image, _)| image).collect();
            assert!(
                !images.contains(&4) && !images.contains(&5),
                "fixture sanity: island observations must not join global tracks"
            );
        }

        // Register only the main component with ground-truth poses and
        // triangulate the clean model.
        let mut poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        poses[4] = None;
        poses[5] = None;
        let mut track_point = vec![None; tracks.len()];
        let config = IncrementalSfmConfig {
            colmap_style_mapper: true,
            structureless_registration: true,
            structureless_min_pair_inliers: 5,
            structureless_min_support_tracks: 6,
            // The 20-point synthetic essentials carry ~1 deg of rotation
            // noise, which at fx=500 is ~9 px of reprojection — far beyond
            // the production-default 2 px admission gate that real
            // hundreds-of-inlier matches easily meet. This fixture exercises
            // the round-chaining mechanics, not the pixel gate (which has
            // its own dedicated tests), so the gate is widened accordingly.
            structureless_max_reprojection_error_px: 12.0,
            max_reprojection_error_px: 12.0,
            ..IncrementalSfmConfig::default()
        };
        triangulate_pending(
            &scene.camera,
            &scene.features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        let clean_points = track_point.iter().filter(|p| p.is_some()).count();
        assert!(
            clean_points >= 10,
            "fixture sanity: clean model must triangulate ({clean_points} points)"
        );

        // A single ascending scan must register the bridge `6` but leave `3`
        // behind: when the scan reaches `3`, `6` is still unregistered and `3`
        // has only two admissible neighbours.
        let single_round_config = IncrementalSfmConfig {
            structureless_max_rounds: 1,
            ..config.clone()
        };
        let single_registered = structureless_registration_rounds(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &mut tracks.clone(),
            &single_round_config,
            &mut poses.clone(),
            &mut track_point.clone(),
        );
        assert_eq!(
            single_registered, 1,
            "one ascending pass must recover exactly the bridge image"
        );

        // Multiple rounds feed `6` back in as a neighbour and chain `3`.
        let total_registered = structureless_registration_rounds(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &mut tracks,
            &config,
            &mut poses,
            &mut track_point,
        );
        assert_eq!(
            total_registered, 2,
            "rounds must chain the dependent island image through the bridge"
        );
        assert!(poses.iter().all(Option::is_some));

        // The chained pose must sit at the true (metric) geometry: rotation
        // tight, centre within a fraction of the neighbour spread.
        for image in [4usize, 5] {
            let pose = poses[image].as_ref().unwrap();
            let rotation_error = (pose.world_to_camera.rotation.inverse()
                * scene.poses[image].world_to_camera.rotation)
                .angle();
            let center_error =
                (pose.camera_center_world() - scene.poses[image].camera_center_world()).norm();
            assert!(
                rotation_error < 0.01,
                "image {image} rotation error {rotation_error} rad too large"
            );
            assert!(
                center_error < 0.05,
                "image {image} centre error {center_error} m too large"
            );
        }
    }
}
