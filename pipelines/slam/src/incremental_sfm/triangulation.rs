//! Track triangulation, post-BA observation filtering and re-triangulation.

use super::*;

/// Triangulate every track that has ≥2 registered observations and is not yet
/// triangulated, accepting only well-conditioned (parallax + reprojection) points.
pub(crate) fn triangulate_pending(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    triangulate_pending_track_ids(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        0..tracks.len(),
    );
}

/// Triangulate only the supplied track ids. In the ordinary, non-COLMAP-style
/// mapper a track that already has a point is never replaced during growth,
/// and an untriangulated track can only gain a newly registered observation
/// from the image just accepted. The growth loop therefore uses this helper
/// for the common path and falls back to a full scan immediately after a BA
/// (where every camera pose may have moved). COLMAP-style growth deliberately
/// retains its historical full scan because local/global BA and completion
/// can change support outside the newly registered image.
pub(super) fn triangulate_pending_track_ids<I>(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    track_ids: I,
) -> Vec<usize>
where
    I: IntoIterator<Item = usize>,
{
    let mut newly_triangulated = Vec::new();
    for track_id in track_ids {
        let Some(track) = tracks.get(track_id) else {
            continue;
        };
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
            newly_triangulated.push(track_id);
        }
    }
    newly_triangulated
}

/// Targeted image update with the IDs that became 3D points during the pass.
/// The IDs let the correspondence-count selector update its per-image score
/// cache in proportion to newly-created points rather than rescanning every
/// unregistered image after each successful PnP.
#[allow(clippy::too_many_arguments)]
pub(super) fn triangulate_pending_for_image_with_new_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    obs_by_image: &[Vec<(usize, usize)>],
    image: usize,
) -> (usize, Vec<usize>) {
    let Some(observations) = obs_by_image.get(image) else {
        return (0, Vec::new());
    };
    let count = observations.len();
    let newly_triangulated = triangulate_pending_track_ids(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        observations.iter().map(|&(_, track_id)| track_id),
    );
    (count, newly_triangulated)
}

/// Seed update with the IDs that became 3D points during the pass.
#[allow(clippy::too_many_arguments)]
pub(super) fn triangulate_pending_for_images_with_new_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    obs_by_image: &[Vec<(usize, usize)>],
    images: &[usize],
) -> (usize, Vec<usize>) {
    let mut track_ids = HashSet::new();
    for &image in images {
        if let Some(observations) = obs_by_image.get(image) {
            track_ids.extend(observations.iter().map(|&(_, track_id)| track_id));
        }
    }
    let count = track_ids.len();
    let newly_triangulated = triangulate_pending_track_ids(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        track_ids,
    );
    (count, newly_triangulated)
}

/// Build the exact count-policy ranking key once from the current point map.
/// The ordinary count selector used to recompute this join for every
/// registration attempt.  A track becomes a point at most once during plain
/// growth, so the cache can then be maintained from the small set returned by
/// each targeted triangulation pass.
pub(super) fn build_correspondence_count_cache(
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
) -> Vec<usize> {
    obs_by_image
        .iter()
        .enumerate()
        .map(|(image, observations)| {
            observations
                .iter()
                .filter(|&&(kp, track_id)| {
                    track_point.get(track_id).is_some_and(Option::is_some)
                        && features
                            .get(image)
                            .is_some_and(|set| set.keypoints.get(kp).is_some())
                })
                .count()
        })
        .collect()
}

/// Add newly triangulated observations to the count-policy ranking cache.
/// `newly_triangulated` is unique per pass, so every observation contributes
/// exactly once, matching a fresh count scan.
pub(super) fn update_correspondence_count_cache(
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    newly_triangulated: &[usize],
    counts: &mut [usize],
) {
    for &track_id in newly_triangulated {
        if track_point.get(track_id).is_none_or(Option::is_none) {
            continue;
        }
        let Some(track) = tracks.get(track_id) else {
            continue;
        };
        for &(image, kp) in track {
            if features
                .get(image)
                .is_some_and(|set| set.keypoints.get(kp).is_some())
            {
                if let Some(count) = counts.get_mut(image) {
                    *count = count.saturating_add(1);
                }
            }
        }
    }
}

/// Incremental correspondence-mode point update.
///
/// Unlike [`triangulate_pending`], this path owns an explicit
/// observation-to-point map and revisits already-created points after every
/// registration.  Newly registered observations therefore participate in the
/// widest-baseline triangulation immediately, while an existing point is
/// replaced only when the candidate lowers its mean registered-view
/// reprojection.  During ordinary growth the state is retained across calls
/// (and refreshed from `track_point` after a BA); helper callers that do not
/// own a growth state get the same deterministic map rebuilt from the immutable
/// tracks.  Malformed duplicate observations are therefore visible to the
/// same one-image-per-point invariant used by the builder.
fn triangulate_correspondence_pending_with_state(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    state: &mut CorrespondencePointState,
) {
    debug_assert_eq!(state.observations.len(), tracks.len());
    state.points.resize(tracks.len(), None);
    for (state_point, track_point) in state.points.iter_mut().zip(track_point.iter()) {
        *state_point = *track_point;
    }
    debug_assert!(state
        .observation_to_point
        .iter()
        .all(|(&(image, kp), &point)| {
            tracks
                .get(point)
                .is_some_and(|track| track.contains(&(image, kp)))
        }));
    for (track_id, track) in tracks.iter().enumerate() {
        let mut obs: Vec<(usize, Point2<f64>)> = Vec::new();
        for &(image, kp) in track {
            if poses.get(image).and_then(Option::as_ref).is_none() {
                continue;
            }
            if let Some(px) = features
                .get(image)
                .and_then(|set| set.keypoints.get(kp))
                .copied()
            {
                obs.push((image, px));
            }
        }
        if obs.len() < 2 {
            continue;
        }
        let Some(candidate) = triangulate_track(camera, poses, &obs, config) else {
            continue;
        };
        let mean_reprojection = |point: &Point3<f64>| -> f64 {
            let mut sum = 0.0;
            let mut count = 0usize;
            for &(image, pixel) in &obs {
                let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                    continue;
                };
                if let Some(error) = reprojection_error_px(camera, pose, point, &pixel) {
                    sum += error;
                    count += 1;
                }
            }
            if count == 0 {
                f64::INFINITY
            } else {
                sum / count as f64
            }
        };
        let should_replace = match state.points.get(track_id).and_then(Option::as_ref) {
            None => true,
            Some(current) => {
                let current_error = mean_reprojection(current);
                let candidate_error = mean_reprojection(&candidate);
                candidate_error.is_finite() && candidate_error + 1e-9 < current_error
            }
        };
        if should_replace {
            state.retriangulate_point(track_id, candidate);
        }
    }
    track_point.copy_from_slice(&state.points[..track_point.len()]);
}

fn triangulate_correspondence_pending(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    let mut state = CorrespondencePointState::from_tracks(tracks, track_point);
    triangulate_correspondence_pending_with_state(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        &mut state,
    );
}

pub(super) fn triangulate_pending_with_config(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    if config.incremental_correspondence_triangulation {
        triangulate_correspondence_pending(camera, features, tracks, poses, config, track_point);
    } else {
        triangulate_pending(camera, features, tracks, poses, config, track_point);
    }
}

pub(super) fn triangulate_pending_with_config_and_state(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    state: Option<&mut CorrespondencePointState>,
) {
    if config.incremental_correspondence_triangulation {
        if let Some(state) = state {
            triangulate_correspondence_pending_with_state(
                camera,
                features,
                tracks,
                poses,
                config,
                track_point,
                state,
            );
        } else {
            triangulate_correspondence_pending(
                camera,
                features,
                tracks,
                poses,
                config,
                track_point,
            );
        }
    } else {
        triangulate_pending(camera, features, tracks, poses, config, track_point);
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

#[allow(clippy::too_many_arguments)]
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
pub(super) fn filter_images(
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
pub(super) fn filter_outlier_observations(
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
pub(super) fn retriangulate_tracks(
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

/// Summary of the optional final minimum-track-length gate.  The gate is
/// intentionally a post-registration operation: it never participates in
/// seed selection or PnP growth, and a failed refinement restores the complete
/// pre-gate state.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct FinalTrackLengthGateStats {
    pub(super) requested_min_length: usize,
    pub(super) attempted: bool,
    pub(super) accepted: bool,
    pub(super) tracks_before: usize,
    pub(super) tracks_removed: usize,
    pub(super) tracks_after: usize,
    pub(super) observations_before: usize,
    pub(super) observations_removed: usize,
    pub(super) observations_after: usize,
    pub(super) retriangulated_tracks: usize,
    pub(super) registered_before: usize,
    pub(super) registered_after: usize,
    pub(super) mean_before_ba: f64,
    pub(super) mean_after_ba: f64,
    pub(super) finite_state: bool,
    pub(super) support_valid: bool,
    pub(super) objective_valid: bool,
}

/// Keep only tracks meeting a final minimum observation count while preserving
/// the parallel `track_point` indexing.  This small pure helper is also used
/// by unit tests so the length-2 removal/length-3 preservation contract does
/// not depend on a camera or solver.
pub(super) fn retain_final_track_length(
    tracks: &mut Vec<Vec<(usize, usize)>>,
    track_point: &mut Vec<Option<Point3<f64>>>,
    min_length: usize,
) -> (usize, usize) {
    debug_assert_eq!(tracks.len(), track_point.len());
    if min_length <= 2 {
        return (0, 0);
    }

    let old_tracks = std::mem::take(tracks);
    let old_points = std::mem::take(track_point);
    let mut removed_tracks = 0usize;
    let mut removed_observations = 0usize;
    let mut kept_tracks = Vec::with_capacity(old_tracks.len());
    let mut kept_points = Vec::with_capacity(old_points.len());
    for (track, point) in old_tracks.into_iter().zip(old_points) {
        if track.len() < min_length {
            removed_tracks += 1;
            removed_observations += track.len();
        } else {
            kept_tracks.push(track);
            kept_points.push(point);
        }
    }
    *tracks = kept_tracks;
    *track_point = kept_points;
    (removed_tracks, removed_observations)
}

/// A final support gate is valid only when every registered camera still has
/// at least one triangulated observation.  This is deliberately weaker than a
/// density heuristic: the gate is allowed to remove all two-view landmarks,
/// but must never turn a registered camera into an unsupported pose.
pub(super) fn final_track_length_support_is_valid(
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> bool {
    let mut image_support = vec![0usize; poses.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point.get(track_id).and_then(Option::as_ref).is_none() {
            continue;
        }
        for &(image, _) in track {
            if poses.get(image).is_some_and(Option::is_some) {
                image_support[image] += 1;
            }
        }
    }
    poses
        .iter()
        .enumerate()
        .all(|(image, pose)| pose.is_none() || image_support.get(image).copied().unwrap_or(0) > 0)
}

fn final_track_length_state_is_finite(
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
) -> bool {
    let camera_finite = camera.params.iter().all(|value| value.is_finite());
    camera_finite
        && poses.iter().all(|pose| {
            pose.as_ref().is_none_or(|pose| {
                pose.world_to_camera
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
                    && pose
                        .world_to_camera
                        .translation
                        .iter()
                        .all(|value| value.is_finite())
            })
        })
        && tracks.len() == track_point.len()
        && track_point.iter().all(|point| {
            point
                .as_ref()
                .is_none_or(|point| point.coords.iter().all(|value| value.is_finite()))
        })
}

/// Remove short landmarks after all registration/splitting work, re-triangulate
/// the remaining support, and run one guarded final BA.  Any solver error,
/// non-finite state, loss of registered-camera support, or increase of the
/// remaining-support reprojection objective rolls back the whole operation.
/// `None` and values below three are no-ops; the example CLI currently exposes
/// only the source-motivated value three.
pub(super) fn apply_final_track_length_gate(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut Vec<Option<Pose>>,
    track_point: &mut Vec<Option<Point3<f64>>>,
    ba_result: &mut Option<BaResult>,
) -> FinalTrackLengthGateStats {
    let Some(min_length) = config.final_min_track_length else {
        return FinalTrackLengthGateStats::default();
    };
    let mut stats = FinalTrackLengthGateStats {
        requested_min_length: min_length,
        tracks_before: tracks.len(),
        observations_before: tracks.iter().map(Vec::len).sum(),
        registered_before: poses.iter().filter(|pose| pose.is_some()).count(),
        ..FinalTrackLengthGateStats::default()
    };
    stats.tracks_after = stats.tracks_before;
    stats.observations_after = stats.observations_before;
    stats.registered_after = stats.registered_before;
    if min_length <= 2
        || !config.final_global_ba
        || !poses.iter().all(Option::is_some)
        || tracks.len() != track_point.len()
    {
        return stats;
    }
    stats.attempted = true;

    let tracks_before = tracks.clone();
    let points_before = track_point.clone();
    let poses_before = poses.clone();
    let camera_before = camera.clone();
    let ba_before = ba_result.clone();
    let (removed_tracks, removed_observations) =
        retain_final_track_length(tracks, track_point, min_length);
    stats.tracks_removed = removed_tracks;
    stats.observations_removed = removed_observations;
    stats.tracks_after = tracks.len();
    stats.observations_after = tracks.iter().map(Vec::len).sum();
    if removed_tracks == 0 {
        stats.accepted = true;
        return stats;
    }

    stats.retriangulated_tracks =
        retriangulate_tracks(camera, features, tracks, config, poses, track_point);
    let mean_before_ba = mean_reprojection_for_track_range(
        camera,
        features,
        tracks,
        poses,
        track_point,
        0,
        tracks.len(),
    );
    stats.mean_before_ba = mean_before_ba;
    let ba = run_bundle_adjustment(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        config.refine_intrinsics,
    );
    let (candidate_ba, refined_camera) = match ba {
        Ok(result) => result,
        Err(_) => {
            *tracks = tracks_before;
            *track_point = points_before;
            *poses = poses_before;
            *camera = camera_before;
            *ba_result = ba_before;
            return stats;
        }
    };
    if let Some(refined_camera) = refined_camera {
        *camera = refined_camera;
    }
    let mean_after = mean_reprojection_for_track_range(
        camera,
        features,
        tracks,
        poses,
        track_point,
        0,
        tracks.len(),
    );
    stats.mean_after_ba = mean_after;
    let registered_after = poses.iter().filter(|pose| pose.is_some()).count();
    stats.registered_after = registered_after;
    let finite = final_track_length_state_is_finite(camera, poses, tracks, track_point);
    let support_valid = final_track_length_support_is_valid(tracks, poses, track_point);
    // The solver's reported robust objective is the acceptance objective.  A
    // mean-pixel change can move slightly upward when short, weak tracks are
    // removed (the denominator and residual population both change), even
    // while the fixed-support BA objective decreases.  Keep that diagnostic
    // pair of means in the log, but do not reject a finite, support-preserving
    // solve solely for this population-statistic effect.
    let objective_valid = candidate_ba.initial_cost.is_finite()
        && candidate_ba.final_cost.is_finite()
        && candidate_ba.final_cost <= candidate_ba.initial_cost + 1.0e-9;
    stats.finite_state = finite;
    stats.support_valid = support_valid;
    stats.objective_valid = objective_valid;
    if finite && support_valid && registered_after == stats.registered_before && objective_valid {
        *ba_result = Some(candidate_ba);
        stats.accepted = true;
    } else {
        *tracks = tracks_before;
        *track_point = points_before;
        *poses = poses_before;
        *camera = camera_before;
        *ba_result = ba_before;
        stats.tracks_after = stats.tracks_before;
        stats.observations_after = stats.observations_before;
        stats.retriangulated_tracks = 0;
    }
    stats
}
