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
//! 2. **Seed.** The verified pair with the most matches (and enough parallax)
//!    bootstraps the reconstruction via two-view relative pose
//!    ([`visloc_vision::two_view`]); its shared tracks are triangulated. This
//!    fixes the gauge (seed image at the origin) and the arbitrary monocular
//!    scale.
//! 3. **Grow.** Repeatedly register the unregistered image that observes the
//!    most already-triangulated tracks, by PnP RANSAC
//!    ([`visloc_vision::ransac`]); then triangulate every track that two
//!    registered views now share with sufficient parallax.
//! 4. **Bundle-adjust.** Periodically and at the end, refine all registered
//!    poses and triangulated points jointly with the Schur-complement BA
//!    ([`crate::bundle`]), seed pose fixed for gauge.
//!
//! The output ([`IncrementalSfmResult`]) carries per-image poses (`None` for
//! images that never registered) and merged multi-view tracks, ready for a
//! COLMAP `points3D.txt` export and downstream 3DGS / NeRF training.

use std::collections::HashMap;

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
    let tracks = build_tracks(features.len(), pairwise, config.min_track_length);

    // For each image, which (keypoint, track) pairs it observes — drives both
    // triangulation and next-image selection.
    let mut obs_by_image: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_images];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, kp) in track {
            obs_by_image[image].push((kp, track_id));
        }
    }

    // Reconstruction state.
    let mut poses: Vec<Option<Pose>> = vec![None; n_images];
    let mut track_point: Vec<Option<Point3<f64>>> = vec![None; tracks.len()];

    // ---- 2. Seed from the strongest verified pair ----
    seed_reconstruction(camera, features, pairwise, &tracks, config, &mut poses)?;
    triangulate_pending(camera, features, &tracks, &poses, config, &mut track_point);

    // ---- 3. Grow: register the best next image, triangulate, periodically BA ----
    let mut failed: Vec<bool> = vec![false; n_images];
    let mut registrations_since_ba = 0usize;
    loop {
        let Some((next_image, corrs)) =
            select_next_image(features, &obs_by_image, &poses, &failed, &track_point)
        else {
            break;
        };

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
            .estimate(&corrs, camera),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: config.max_reprojection_error_px,
                ..PnPRansac::default()
            }
            .estimate(&corrs, camera),
        };
        match report {
            Some(report) if report.inliers.len() >= config.min_pnp_inliers => {
                poses[next_image] = Some(report.pose);
                triangulate_pending(camera, features, &tracks, &poses, config, &mut track_point);
                registrations_since_ba += 1;
                if config.ba_every > 0 && registrations_since_ba >= config.ba_every {
                    run_bundle_adjustment(
                        camera,
                        features,
                        &tracks,
                        config,
                        &mut poses,
                        &mut track_point,
                    )
                    .map_err(IncrementalSfmError::Ba)?;
                    registrations_since_ba = 0;
                }
            }
            _ => failed[next_image] = true,
        }
    }

    // ---- 4. Final global bundle adjustment ----
    let ba_result = if config.final_global_ba {
        Some(
            run_bundle_adjustment(
                camera,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
            )
            .map_err(IncrementalSfmError::Ba)?,
        )
    } else {
        None
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
            if let Some(err) = reprojection_error_px(camera, pose, &position, &pixel) {
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

/// Pick a verified pair that bootstraps well — recover its two-view relative
/// pose and place both images (seed at the world origin). Mutates `poses`.
///
/// The strongest-match pair is *not* always a good seed: on an orbit the most
/// overlapping pair is two adjacent low-parallax frames, whose tiny baseline
/// makes triangulation depth-unstable (and fails the parallax gate, leaving the
/// reconstruction with nothing to register against). So candidates are tried in
/// descending match order but each is accepted only if enough of its
/// correspondences actually triangulate to well-conditioned points — the same
/// parallax + cheirality + reprojection gate the rest of the pipeline uses.
fn seed_reconstruction(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    _tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
) -> Result<(), IncrementalSfmError> {
    // Candidate pairs by descending match count.
    let mut order: Vec<usize> = (0..pairwise.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pairwise[i].matches.len()));

    let estimator = RelativePoseEstimator::default();
    let min_cos = config.min_triangulation_angle_deg.to_radians().cos();
    for &pi in &order {
        let pair = &pairwise[pi];
        if pair.matches.len() < config.min_seed_matches {
            break; // sorted descending — nothing weaker can qualify
        }
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
            continue;
        };
        if relative.inliers.len() < config.min_seed_matches {
            continue;
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
        // Count inlier correspondences that triangulate to well-conditioned
        // points under the shared parallax / cheirality / reprojection gate.
        let mut well_triangulated = 0usize;
        for &inl in &relative.inliers {
            let (px_i, px_j) = corr_kp[inl];
            let obs = [(pair.image_i, px_i), (pair.image_j, px_j)];
            if triangulate_track(
                camera,
                poses,
                &obs,
                min_cos,
                config.max_reprojection_error_px,
            )
            .is_some()
            {
                well_triangulated += 1;
            }
        }
        if well_triangulated >= config.min_seed_matches {
            return Ok(()); // good baseline — keep these poses
        }
        // Low parallax: undo and try the next pair.
        poses[pair.image_i] = None;
        poses[pair.image_j] = None;
    }
    Err(IncrementalSfmError::NoSeedPair)
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
    let min_cos = (config.min_triangulation_angle_deg.to_radians()).cos();
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
        if let Some(point) = triangulate_track(
            camera,
            poses,
            &obs,
            min_cos,
            config.max_reprojection_error_px,
        ) {
            track_point[track_id] = Some(point);
        }
    }
}

/// Triangulate one track from its registered observations: choose the
/// widest-parallax view pair, DLT-triangulate, and validate cheirality,
/// parallax, and reprojection in both views.
fn triangulate_track(
    camera: &Camera,
    poses: &[Option<Pose>],
    obs: &[(usize, Point2<f64>)],
    min_cos: f64,
    max_reproj: f64,
) -> Option<Point3<f64>> {
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
    if cos > min_cos {
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

/// Among unregistered, not-failed images, choose the one observing the most
/// triangulated tracks, returning it with its 2D-3D correspondences.
fn select_next_image(
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    failed: &[bool],
    track_point: &[Option<Point3<f64>>],
) -> Option<(usize, Vec<Correspondence2D3D>)> {
    let mut best: Option<(usize, Vec<Correspondence2D3D>)> = None;
    for (image, observations) in obs_by_image.iter().enumerate() {
        if poses[image].is_some() || failed[image] {
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
/// poses and points back in place.
fn run_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<BaResult, BaError> {
    let mut ba = BundleAdjustment::new(camera.clone());

    let mut fixed_done = false;
    for (image, pose) in poses.iter().enumerate() {
        if let Some(pose) = pose {
            ba.add_pose(image as u64, pose.clone());
            if !fixed_done {
                ba.fix_pose(image as u64);
                fixed_done = true;
            }
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

    let result = ba.optimize(&config.ba_config)?;

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
    Ok(result)
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
}
