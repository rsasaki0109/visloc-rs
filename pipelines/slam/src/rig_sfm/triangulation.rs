//! Frontier triangulation, re-triangulation of unanchored / deferred tracks
//! and the ray / reprojection geometry helpers.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn retriangulate_unanchored_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
    track_indices: &HashSet<usize>,
) -> usize {
    let mut track_indices = track_indices.iter().copied().collect::<Vec<_>>();
    track_indices.sort_unstable();
    let mut refreshed = 0usize;
    for track_index in track_indices {
        let track = &mut tracks[track_index];
        let Some(current) = track.position else {
            continue;
        };
        if track.metric_anchored {
            continue;
        }
        let rays = track
            .observations
            .iter()
            .filter_map(|&(image, keypoint)| {
                let pose = image_poses[image].as_ref()?;
                let sensor = &rig.sensors()[image_assignment[image].1];
                let normalized = sensor
                    .camera
                    .normalize_pixel(&features[image].keypoints[keypoint])?;
                let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
                let direction_world = pose
                    .camera_to_world()
                    .rotation
                    .transform_vector(&bearing_camera)
                    .normalize();
                Some((image, keypoint, pose.camera_center_world(), direction_world))
            })
            .collect::<Vec<_>>();
        if rays.len() < 2 {
            continue;
        }

        // Bound pair generation independently of track length. Long tracks
        // are sampled deterministically, so this diagnostic recovery path
        // cannot reintroduce quadratic state or work at 10k scale.
        let rays = bounded_evenly_spaced_indices(rays.len(), 16)
            .into_iter()
            .map(|index| rays[index])
            .collect::<Vec<_>>();

        // A long conflict-preserving track can contain a bad transitive edge.
        // Evaluate a small, deterministic set of the widest-baseline ray pairs
        // instead of trusting one pair or materializing any global dense state.
        let mut pairs = Vec::new();
        for left in 0..rays.len() {
            for right in (left + 1)..rays.len() {
                let angle = rays[left].3.dot(&rays[right].3).clamp(-1.0, 1.0).acos();
                if angle.to_degrees() >= config.min_triangulation_angle_deg {
                    pairs.push((angle, left, right));
                }
            }
        }
        pairs.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        pairs.truncate(16);

        let current_score = rig_point_support(
            rig,
            features,
            image_assignment,
            image_poses,
            track,
            &current,
            config.max_reprojection_error_px,
        );
        let mut best = None;
        for (_, left, right) in pairs {
            let Some(candidate) =
                closest_ray_midpoint(&rays[left].2, &rays[left].3, &rays[right].2, &rays[right].3)
            else {
                continue;
            };
            let score = rig_point_support(
                rig,
                features,
                image_assignment,
                image_poses,
                track,
                &candidate,
                config.max_reprojection_error_px,
            );
            if score.0 < 2 {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, best_score): &(Point3<f64>, (usize, f64))| score > *best_score)
            {
                best = Some((candidate, score));
            }
        }
        let Some((candidate, candidate_score)) = best else {
            continue;
        };
        if candidate_score > current_score {
            track.position = Some(candidate);
            refreshed += 1;
        }
    }
    refreshed
}

fn rig_point_support(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    track: &WorkingTrack,
    point: &Point3<f64>,
    max_reprojection_error_px: f64,
) -> (usize, f64) {
    let mut inliers = 0usize;
    let mut negative_squared_error = 0.0;
    for &(image, keypoint) in &track.observations {
        let Some(pose) = image_poses[image].as_ref() else {
            continue;
        };
        let sensor = &rig.sensors()[image_assignment[image].1];
        let point_camera = pose.transform_world_point(point);
        if point_camera.z <= 0.0 {
            continue;
        }
        let Some(projected) = sensor.camera.project(&point_camera) else {
            continue;
        };
        let squared_error = (projected - features[image].keypoints[keypoint]).norm_squared();
        if squared_error <= max_reprojection_error_px * max_reprojection_error_px {
            inliers += 1;
            negative_squared_error -= squared_error;
        }
    }
    (inliers, negative_squared_error)
}

#[derive(Debug, Default)]
pub(super) struct TriangulationUpdate {
    pub(super) landmarks: Vec<usize>,
    pub(super) attempts: usize,
    pub(super) robust_tracks: usize,
    pub(super) pruned_observations: usize,
    pub(super) majority_rejections: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct DeferredRetriangulationUpdate {
    pub(super) tracks: usize,
    pub(super) observations: usize,
}

pub(super) fn append_retriangulated_deferred_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    deferred_pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut Vec<WorkingTrack>,
) -> DeferredRetriangulationUpdate {
    let owned = tracks
        .iter()
        .flat_map(|track| track.observations.iter().copied())
        .collect::<HashSet<_>>();
    let deferred_config = RigSfmConfig {
        track_builder: if config.deferred_retriangulation_metric_temporal_quadrilateral_tracks {
            RigTrackBuilder::MetricTemporalQuadrilateral
        } else if config.deferred_retriangulation_metric_temporal_cycle_tracks {
            RigTrackBuilder::MetricTemporalCycle
        } else {
            config.track_builder
        },
        ..*config
    };
    let candidates = build_rig_track_output(
        features,
        deferred_pairwise,
        image_assignment,
        &deferred_config,
    )
    .tracks;
    let first_new_track = tracks.len();
    tracks.extend(
        candidates
            .into_iter()
            .filter(|observations| {
                observations.len() >= config.min_track_length
                    && (config.deferred_retriangulation_min_metric_frames == 0
                        || metric_anchor_frame_count(observations, image_assignment)
                            >= config.deferred_retriangulation_min_metric_frames)
                    && observations
                        .iter()
                        .all(|observation| !owned.contains(observation))
            })
            .map(|observations| WorkingTrack {
                metric_anchored: track_is_metric_anchored(&observations, image_assignment),
                observations,
                position: None,
            }),
    );
    let frontier = (first_new_track..tracks.len()).collect::<HashSet<_>>();
    let _ = triangulate_frontier(
        rig,
        features,
        image_assignment,
        image_poses,
        config,
        tracks,
        frontier,
    );
    retain_positioned_deferred_tracks(tracks, first_new_track)
}

pub(super) fn metric_anchor_frame_count(
    observations: &[(usize, usize)],
    image_assignment: &[(usize, usize)],
) -> usize {
    let mut first_sensor = HashMap::<usize, usize>::new();
    let mut metric_frames = HashSet::new();
    for &(image, _) in observations {
        let (frame, sensor) = image_assignment[image];
        if first_sensor
            .insert(frame, sensor)
            .is_some_and(|previous| previous != sensor)
        {
            metric_frames.insert(frame);
        }
    }
    metric_frames.len()
}

pub(super) fn retain_positioned_deferred_tracks(
    tracks: &mut Vec<WorkingTrack>,
    first_new_track: usize,
) -> DeferredRetriangulationUpdate {
    let mut update = DeferredRetriangulationUpdate::default();
    let mut index = 0usize;
    tracks.retain(|track| {
        let keep = index < first_new_track || track.position.is_some();
        if index >= first_new_track && keep {
            update.tracks += 1;
            update.observations += track.observations.len();
        }
        index += 1;
        keep
    });
    update
}

pub(super) fn triangulate_frontier(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
    frontier: HashSet<usize>,
) -> TriangulationUpdate {
    let mut update = TriangulationUpdate::default();
    // Hash iteration order is process-randomized.  The order in which newly
    // triangulated landmarks enter each frame cache changes the indexed
    // RANSAC samples, so canonicalize it before any numeric work.
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    frontier.sort_unstable();
    let rig_timing = rig_timing_enabled();
    for track_index in frontier {
        if tracks[track_index].position.is_some() {
            continue;
        }
        let observation_count = tracks[track_index].observations.len();
        if config.max_track_observations_for_triangulation > 0
            && observation_count > config.max_track_observations_for_triangulation
        {
            // A real landmark cannot be seen in this many of a rig's images;
            // this is a union-find conflict blob from visual aliasing that
            // never collided two observations from the same image (so it
            // was never split into `conflicting_components`). Skip it in
            // O(1) rather than re-scanning its whole observation list -- see
            // the field doc on `max_track_observations_for_triangulation`.
            if rig_timing {
                eprintln!(
                    "rig-timing: skip-oversized-track track={track_index} \
                     observations={observation_count} cap={}",
                    config.max_track_observations_for_triangulation,
                );
            }
            continue;
        }
        if rig_timing && observation_count >= 1_000 {
            eprintln!(
                "rig-timing: large-track-triangulate-start track={track_index} \
                 observations={observation_count}",
            );
        }
        update.attempts += 1;
        let rays = tracks[track_index]
            .observations
            .iter()
            .filter_map(|(image, keypoint)| {
                let pose = image_poses[*image].as_ref()?;
                let sensor = &rig.sensors()[image_assignment[*image].1];
                let normalized = sensor
                    .camera
                    .normalize_pixel(&features[*image].keypoints[*keypoint])?;
                let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
                let direction_world = pose
                    .camera_to_world()
                    .rotation
                    .transform_vector(&bearing_camera)
                    .normalize();
                Some((
                    *image,
                    *keypoint,
                    pose.camera_center_world(),
                    direction_world,
                ))
            })
            .collect::<Vec<_>>();
        // A long track may contain thousands of observations. Searching its
        // full Cartesian ray-pair set is quadratic and can dominate a large
        // reconstruction. Keep the input scan linear, then choose a stable,
        // bounded subset for the widest-baseline search.
        let candidate_rays = bounded_evenly_spaced_indices(rays.len(), 1_024);
        let mut best = None;
        for left_offset in 0..candidate_rays.len() {
            for right_offset in (left_offset + 1)..candidate_rays.len() {
                let left = candidate_rays[left_offset];
                let right = candidate_rays[right_offset];
                let cosine = rays[left].3.dot(&rays[right].3).clamp(-1.0, 1.0);
                let angle = cosine.acos();
                if best.is_none_or(|(_, _, best_angle)| angle > best_angle) {
                    best = Some((left, right, angle));
                }
            }
        }
        let Some((left, right, angle)) = best else {
            continue;
        };
        if angle.to_degrees() < config.min_triangulation_angle_deg {
            continue;
        }
        let Some(point) =
            closest_ray_midpoint(&rays[left].2, &rays[left].3, &rays[right].2, &rays[right].3)
        else {
            continue;
        };
        let observation_is_inlier = |&(image, keypoint): &(usize, usize)| {
            let pose = image_poses[image].as_ref()?;
            let sensor = &rig.sensors()[image_assignment[image].1];
            let point_camera = pose.transform_world_point(&point);
            Some(
                point_camera.z > 0.0
                    && sensor.camera.project(&point_camera).is_some_and(|pixel| {
                        (pixel - features[image].keypoints[keypoint]).norm()
                            <= config.max_reprojection_error_px
                    }),
            )
        };
        if config.robust_triangulation_pruning {
            let registered = tracks[track_index]
                .observations
                .iter()
                .filter(|observation| observation_is_inlier(observation).is_some())
                .count();
            let inliers = tracks[track_index]
                .observations
                .iter()
                .filter(|observation| observation_is_inlier(observation) == Some(true))
                .count();
            let required = ((registered as f64 * config.triangulation_min_inlier_fraction).ceil()
                as usize)
                .max(2);
            if inliers < required {
                update.majority_rejections += 1;
                continue;
            }
            let observations_before = tracks[track_index].observations.len();
            tracks[track_index]
                .observations
                .retain(|observation| observation_is_inlier(observation).unwrap_or(true));
            update.pruned_observations +=
                observations_before - tracks[track_index].observations.len();
            tracks[track_index].metric_anchored =
                track_is_metric_anchored(&tracks[track_index].observations, image_assignment);
            tracks[track_index].position = Some(point);
            update.robust_tracks += 1;
            update.landmarks.push(track_index);
        } else if tracks[track_index]
            .observations
            .iter()
            .all(|(image, keypoint)| {
                let Some(pose) = image_poses[*image].as_ref() else {
                    return true;
                };
                let sensor = &rig.sensors()[image_assignment[*image].1];
                let point_camera = pose.transform_world_point(&point);
                point_camera.z > 0.0
                    && sensor.camera.project(&point_camera).is_some_and(|pixel| {
                        (pixel - features[*image].keypoints[*keypoint]).norm()
                            <= config.max_reprojection_error_px
                    })
            })
        {
            tracks[track_index].position = Some(point);
            update.landmarks.push(track_index);
        }
    }
    update
}

pub(super) fn bounded_evenly_spaced_indices(len: usize, max_samples: usize) -> Vec<usize> {
    if len == 0 || max_samples == 0 {
        return Vec::new();
    }
    if len <= max_samples {
        return (0..len).collect();
    }
    if max_samples == 1 {
        return vec![0];
    }
    (0..max_samples)
        .map(|sample| sample * (len - 1) / (max_samples - 1))
        .collect()
}

pub(super) fn closest_ray_midpoint(
    origin_left: &Point3<f64>,
    direction_left: &Vector3<f64>,
    origin_right: &Point3<f64>,
    direction_right: &Vector3<f64>,
) -> Option<Point3<f64>> {
    let offset = origin_left - origin_right;
    let cosine = direction_left.dot(direction_right);
    let denominator = 1.0 - cosine * cosine;
    if denominator <= 1.0e-12 {
        return None;
    }
    let left_projection = direction_left.dot(&offset);
    let right_projection = direction_right.dot(&offset);
    let left_depth = (cosine * right_projection - left_projection) / denominator;
    let right_depth = (right_projection - cosine * left_projection) / denominator;
    if left_depth <= 0.0 || right_depth <= 0.0 {
        return None;
    }
    let point_left = origin_left + left_depth * direction_left;
    let point_right = origin_right + right_depth * direction_right;
    Some(Point3::from((point_left.coords + point_right.coords) * 0.5))
}

pub(super) fn reprojection_error(
    rig: &GeneralizedCameraRig,
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for track in tracks {
        for (image, _, pixel) in &track.observations {
            let Some(pose) = image_poses[*image].as_ref() else {
                continue;
            };
            let sensor = &rig.sensors()[image_assignment[*image].1];
            if let Some(projected) = sensor
                .camera
                .project(&pose.transform_world_point(&track.position))
            {
                sum += (projected - pixel).norm();
                count += 1;
            }
        }
    }
    (sum, count)
}
