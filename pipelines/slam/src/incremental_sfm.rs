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

use nalgebra::{Point2, Point3, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};

use crate::{BaConfig, BaError, BaObservation, BaResult, BundleAdjustment, RobustKernel};

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
            low_parallax_min_observations: None,
            low_parallax_min_angle_deg: 1.0,
            refine_intrinsics: false,
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

/// Output of [`incremental_sfm`].
#[derive(Debug, Clone)]
pub struct IncrementalSfmResult {
    /// Refined pose per input image; `None` for images that never registered.
    pub poses: Vec<Option<Pose>>,
    /// Reconstructed multi-view tracks (after the final BA, if enabled).
    pub tracks: Vec<SfmTrack>,
    /// Number of images that registered into the reconstruction.
    pub registered_images: usize,
    /// Mean reprojection error (px) over every observation of every track.
    pub mean_reprojection_px: f64,
    /// Result of the final BA solve, if one ran.
    pub ba_result: Option<BaResult>,
    /// Refined camera intrinsics, when `config.refine_intrinsics` was set. The
    /// poses, tracks, and `mean_reprojection_px` are all expressed against *this*
    /// camera, so a COLMAP / 3DGS export must use it rather than the input camera.
    /// `None` when intrinsics refinement was off.
    pub refined_camera: Option<Camera>,
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
    let n_images = features.len();

    // ---- 1. Build feature tracks via union-find over (image, keypoint) ----
    let mut tracks = build_tracks(features.len(), pairwise, config.min_track_length);

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
    let seed_order = seed_candidate_order(pairwise, config);
    let trials = config.seed_trials.max(1);
    let not_trapped = largest_connected_component(pairwise, n_images)
        .div_ceil(2)
        .max(1);
    let mut best: Option<SeedGrowth> = None;
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
        if reach == 0 {
            continue; // pair failed the seed gate — nothing placed, no grow ran
        }
        grows += 1;
        if best
            .as_ref()
            .is_none_or(|(best_reach, _, _, _)| reach > *best_reach)
        {
            best = Some((reach, trial_poses, trial_points, trial_cam));
        }
        if reach >= not_trapped || grows >= trials {
            break;
        }
    }
    let (_, mut poses, mut track_point, grown_cam) = best.ok_or(IncrementalSfmError::NoSeedPair)?;

    // ---- 4 + 5. Final refinement ----
    // When intrinsics refinement is on, growth already co-evolved them into
    // `grown_cam` (COLMAP keeps the camera moving with the structure so a wrong
    // focal cannot be silently absorbed). The final solve continues refining from
    // there; `cam` expresses the output poses/tracks/reprojection and is returned
    // to the caller for export.
    let mut cam = grown_cam;
    let ba_result = if config.colmap_style_mapper {
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

    // ---- Assemble output tracks (only triangulated, registered observations) ----
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

    Ok(IncrementalSfmResult {
        poses,
        tracks: out_tracks,
        registered_images,
        mean_reprojection_px,
        ba_result,
        refined_camera: config.refine_intrinsics.then_some(cam),
    })
}

/// Union-find over `(image, keypoint)` nodes joined by pairwise matches. Returns
/// the consistent tracks (no two keypoints from the same image) spanning at
/// least `min_track_length` distinct images.
fn build_tracks(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    let _ = n_images;
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

    let mut tracks = Vec::new();
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
    triangulate_pending(&cam, features, tracks, &poses, config, &mut track_point);

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
    loop {
        let Some((next_image, corrs)) = select_next_image(
            features,
            obs_by_image,
            &poses,
            &trials,
            max_trials,
            &track_point,
        ) else {
            break;
        };
        trials[next_image] += 1;

        // P3P (Grunert) is the default minimal solver — well-posed on coplanar
        // façades where the linear DLT degenerates. Both share the Gauss-Newton
        // refiner and the config reprojection gate.
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
        let Some(report) = report.filter(|r| r.inliers.len() >= config.min_pnp_inliers) else {
            continue; // registration failed this attempt (may be retried)
        };
        poses[next_image] = Some(report.pose);
        triangulate_pending(&cam, features, tracks, &poses, config, &mut track_point);

        if config.colmap_style_mapper {
            // COLMAP `AdjustLocalBundle`: tighten the new image + its covisible
            // neighbourhood after every registration.
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

            // Growth-ratio global refinement (COLMAP `IterativeGlobalRefinement`).
            // During the seed search `tracks` is shared read-only across trials,
            // so the in-growth refinement only re-triangulates + re-BAs (touching
            // this trial's own poses/points); the track-membership *filter* that
            // would mutate the shared tracks is deferred to the final refinement,
            // after a seed has been committed. The BA's Huber kernel keeps
            // outliers down-weighted in the meantime.
            let n_reg = poses.iter().filter(|p| p.is_some()).count();
            if n_reg as f64 >= reg_at_last_global as f64 * config.global_ba_images_ratio {
                growth_global_refinement(
                    &mut cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?;
                reg_at_last_global = n_reg;
                // Structure changed — give previously-failed images a fresh shot
                // by resetting their trial counters (COLMAP retries on change).
                for (i, t) in trials.iter_mut().enumerate() {
                    if poses[i].is_none() {
                        *t = 0;
                    }
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
    Ok((poses, track_point, registered, cam))
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
fn triangulate_track(
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

/// Among unregistered images still under the per-image trial cap, choose the one
/// observing the most triangulated tracks, returning it with its 2D-3D
/// correspondences.
fn select_next_image(
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    trials: &[usize],
    max_trials: usize,
    track_point: &[Option<Point3<f64>>],
) -> Option<(usize, Vec<Correspondence2D3D>)> {
    let mut best: Option<(usize, Vec<Correspondence2D3D>)> = None;
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
        if best.as_ref().is_none_or(|(_, b)| corrs.len() > b.len()) {
            best = Some((image, corrs));
        }
    }
    best
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

    let mut result = run_ba(camera, tracks, poses, track_point)?;
    for _ in 0..config.global_ba_max_refinements {
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
        if (changed as f64 / total_obs as f64) < config.global_ba_change_rate {
            break;
        }
        result = run_ba(camera, tracks, poses, track_point)?;
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
fn reprojection_error_px(
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
        let tracks = build_tracks(2, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
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
        // the anisotropic South-Building benchmark). With intrinsics co-evolution on,
        // the COLMAP-style growth schedule must pull fx back toward 500 (the
        // final-only alternating refinement can't — converged structure absorbs the
        // error before the intrinsics step sees it), while leaving the un-perturbed
        // fy and principal point essentially fixed (no spurious drift).
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
        eprintln!("co-evolved intrinsics: fx {fx} fy {fy} cx {cx} cy {cy}");
        // fx moves down from 530 toward 500 — directional recovery is the claim (the
        // final-only alternating refinement can't move at all once converged
        // structure absorbs the error). The magnitude scales with the number of
        // growth global passes, i.e. image count: a clear move on this 6-camera ring,
        // ~25% of the injected error on the 128-image South-Building benchmark.
        assert!(
            fx < 530.0 - 1.0,
            "fx {fx} should recover toward 500 from 530"
        );
        // The un-perturbed parameters must not drift: fy within ~2 of 500, and the
        // principal point within ~2 px of (320, 240).
        assert!((fy - 500.0).abs() < 2.0, "fy {fy} drifted from 500");
        assert!((cx - 320.0).abs() < 2.0, "cx {cx} drifted from 320");
        assert!((cy - 240.0).abs() < 2.0, "cy {cy} drifted from 240");
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
}
