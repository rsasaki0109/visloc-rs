//! Deferred registration from a lower-priority pair suffix, pose-gap
//! interpolation and motion-bridge candidates.

use super::*;

#[derive(Debug, Clone, Copy)]
struct DeferredRigCorrespondence {
    sensor_index: usize,
    point2d: Point2<f64>,
    point3d: Point3<f64>,
    track_index: Option<usize>,
    image: usize,
    keypoint: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DeferredRigSource {
    frame: usize,
    image: usize,
    keypoint: usize,
    track_index: Option<usize>,
}

type DeferredTargetSources = HashMap<(usize, usize), Vec<DeferredRigSource>>;

#[derive(Debug, Default)]
pub(super) struct DeferredRegistrationUpdate {
    pub(super) pair_visits: usize,
    pub(super) correspondence_insertions: usize,
    pub(super) pnp_attempts: usize,
    pub(super) pnp_estimation_failures: usize,
    pub(super) pnp_inlier_rejections: usize,
    pub(super) registrations: usize,
    pub(super) interpolation_registrations: usize,
    pub(super) observations_attached: usize,
}

fn observation_track(index: &[(usize, usize)], keypoint: usize) -> Option<usize> {
    index
        .binary_search_by_key(&keypoint, |&(candidate, _)| candidate)
        .ok()
        .map(|position| index[position].1)
}

fn insert_observation_track(index: &mut Vec<(usize, usize)>, keypoint: usize, track: usize) {
    match index.binary_search_by_key(&keypoint, |&(candidate, _)| candidate) {
        Ok(position) => index[position].1 = track,
        Err(position) => index.insert(position, (keypoint, track)),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_deferred_correspondences(
    target_frame: usize,
    target_sources: &DeferredTargetSources,
    source_frame_filter: Option<usize>,
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &[WorkingTrack],
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
) -> Vec<DeferredRigCorrespondence> {
    let mut entries = target_sources
        .iter()
        .filter_map(|(&(image, keypoint), sources)| {
            sources
                .iter()
                .filter(|source| source_frame_filter.is_none_or(|frame| source.frame == frame))
                .min_by_key(|source| {
                    (
                        target_frame.abs_diff(source.frame),
                        source.track_index.is_none(),
                        source.frame,
                        source.image,
                        source.keypoint,
                    )
                })
                .copied()
                .map(|source| (image, keypoint, source))
        })
        .collect::<Vec<_>>();
    entries.sort_unstable();
    let mut used_image_tracks = HashSet::new();
    entries
        .into_iter()
        .filter(|&(image, _, source)| {
            source
                .track_index
                .is_none_or(|track| used_image_tracks.insert((image, track)))
        })
        .filter_map(|(image, keypoint, source)| {
            let point3d = source
                .track_index
                .and_then(|track| tracks[track].position)
                .or_else(|| {
                    triangulate_verified_stereo_observation(
                        rig,
                        features,
                        image_assignment,
                        image_poses,
                        config,
                        stereo_links,
                        source.image,
                        source.keypoint,
                    )
                })?;
            Some(DeferredRigCorrespondence {
                sensor_index: image_assignment[image].1,
                point2d: features[image].keypoints[keypoint],
                point3d,
                track_index: source.track_index,
                image,
                keypoint,
            })
        })
        .collect()
}

pub(super) fn interpolate_deferred_pose_gaps(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    adjacency: &[Vec<usize>],
    max_gap: usize,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
) -> usize {
    if max_gap == 0 {
        return 0;
    }
    let mut filled = 0usize;
    let mut cursor = 0usize;
    while cursor < frame_poses.len() {
        if frame_poses[cursor].is_some() {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < frame_poses.len() && frame_poses[cursor].is_none() {
            cursor += 1;
        }
        let end = cursor;
        let gap = end - start;
        if start == 0
            || end == frame_poses.len()
            || gap > max_gap
            || (start..end).any(|frame| adjacency[frame].is_empty())
        {
            continue;
        }
        let left = frame_poses[start - 1]
            .as_ref()
            .expect("gap begins after a registered frame")
            .camera_to_world();
        let right = frame_poses[end]
            .as_ref()
            .expect("bounded gap ends at a registered frame")
            .camera_to_world();
        for (offset, frame) in (start..end).enumerate() {
            let fraction = (offset + 1) as f64 / (gap + 1) as f64;
            let camera_to_world = SE3::new(
                left.rotation.slerp(&right.rotation, fraction),
                left.translation * (1.0 - fraction) + right.translation * fraction,
            );
            frame_poses[frame] = Some(Pose {
                world_to_camera: camera_to_world.inverse(),
            });
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame]
                    .as_ref()
                    .expect("interpolated pose was just installed"),
                image_poses,
            );
            filled += 1;
        }
    }
    filled
}

/// Register missing frames from a lower-priority verified-pair suffix without
/// allowing that suffix to alter the accepted base structure. Every suffix
/// pair is visited at most once, when either endpoint first becomes
/// registered, and only PnP-inlier observations are attached to base tracks.
#[allow(clippy::too_many_arguments)]
pub(super) fn register_deferred_frames(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
    tracks: &mut [WorkingTrack],
    image_tracks: &mut [Vec<(usize, usize)>],
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
) -> DeferredRegistrationUpdate {
    let adjacency = build_frame_pair_adjacency(pairwise, image_assignment, frames.len());
    let mut pair_visited = vec![false; pairwise.len()];
    let mut source_queue = frame_poses
        .iter()
        .enumerate()
        .filter_map(|(frame, pose)| pose.as_ref().map(|_| frame))
        .collect::<Vec<_>>();
    let mut target_candidates = vec![DeferredTargetSources::new(); frames.len()];
    let mut versions = vec![0usize; frames.len()];
    let mut attempted_versions = vec![None; frames.len()];
    let mut candidate_support = vec![0usize; frames.len()];
    let mut candidate_sensors = vec![0usize; frames.len()];
    let mut candidate_direct = vec![0usize; frames.len()];
    let mut heap = BinaryHeap::new();
    let mut update = DeferredRegistrationUpdate::default();
    let required_inliers = config
        .deferred_registration_min_pnp_inliers
        .unwrap_or(config.min_pnp_inliers)
        .max(6);
    let required_sensors = config
        .deferred_registration_min_pnp_sensors
        .unwrap_or(config.min_pnp_sensors);
    let pnp = GeneralizedPnPRansac {
        iterations: config
            .deferred_registration_pnp_max_iterations
            .unwrap_or(config.pnp_max_iterations),
        reprojection_threshold: config.max_reprojection_error_px,
        seed: config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    };

    loop {
        let mut changed_frames = HashSet::new();
        for source_frame in std::mem::take(&mut source_queue) {
            if frame_poses[source_frame].is_none() {
                continue;
            }
            for &pair_index in &adjacency[source_frame] {
                if pair_visited[pair_index] {
                    continue;
                }
                let pair = &pairwise[pair_index];
                let frame_i = image_assignment[pair.image_i].0;
                let frame_j = image_assignment[pair.image_j].0;
                let (target_frame, source_is_i) = if frame_i == source_frame {
                    (frame_j, true)
                } else {
                    (frame_i, false)
                };
                if target_frame == source_frame || frame_poses[target_frame].is_some() {
                    pair_visited[pair_index] = true;
                    continue;
                }
                pair_visited[pair_index] = true;
                update.pair_visits += 1;
                let mut inserted = 0usize;
                for &(keypoint_i, keypoint_j) in &pair.matches {
                    let (source_image, source_keypoint, target_image, target_keypoint) =
                        if source_is_i {
                            (pair.image_i, keypoint_i, pair.image_j, keypoint_j)
                        } else {
                            (pair.image_j, keypoint_j, pair.image_i, keypoint_i)
                        };
                    let track_index =
                        observation_track(&image_tracks[source_image], source_keypoint)
                            .filter(|&track| tracks[track].position.is_some());
                    if track_index.is_none()
                        && !stereo_links.contains_key(&(source_image, source_keypoint))
                    {
                        continue;
                    }
                    let occupied_by_positioned_track =
                        observation_track(&image_tracks[target_image], target_keypoint)
                            .is_some_and(|existing| tracks[existing].position.is_some());
                    if occupied_by_positioned_track
                        || track_index.is_some_and(|track_index| {
                            tracks[track_index]
                                .observations
                                .iter()
                                .any(|&(image, _)| image == target_image)
                        })
                    {
                        continue;
                    }
                    let source = DeferredRigSource {
                        frame: source_frame,
                        image: source_image,
                        keypoint: source_keypoint,
                        track_index,
                    };
                    target_candidates[target_frame]
                        .entry((target_image, target_keypoint))
                        .or_default();
                    let sources = target_candidates[target_frame]
                        .get_mut(&(target_image, target_keypoint))
                        .expect("candidate entry was just installed");
                    if !sources.contains(&source) {
                        sources.push(source);
                        inserted += 1;
                    }
                }
                if inserted > 0 {
                    update.correspondence_insertions += inserted;
                    changed_frames.insert(target_frame);
                }
            }
        }
        for frame in changed_frames {
            versions[frame] += 1;
            heap.push((
                target_candidates[frame].len(),
                Reverse(frame),
                versions[frame],
            ));
        }

        let mut registered_any = false;
        while let Some((_, Reverse(frame), version)) = heap.pop() {
            if frame_poses[frame].is_some()
                || versions[frame] != version
                || attempted_versions[frame] == Some(version)
            {
                continue;
            }
            let candidates = collect_deferred_correspondences(
                frame,
                &target_candidates[frame],
                None,
                rig,
                features,
                image_assignment,
                image_poses,
                config,
                tracks,
                stereo_links,
            );
            candidate_support[frame] = candidates.len();
            candidate_sensors[frame] = candidates
                .iter()
                .map(|candidate| candidate.sensor_index)
                .collect::<HashSet<_>>()
                .len();
            candidate_direct[frame] = candidates
                .iter()
                .filter(|candidate| candidate.track_index.is_none())
                .count();
            if candidates.len() < required_inliers || candidate_sensors[frame] < required_sensors {
                continue;
            }
            attempted_versions[frame] = Some(version);
            let mut source_support = HashMap::<usize, usize>::new();
            for sources in target_candidates[frame].values() {
                for source in sources {
                    *source_support.entry(source.frame).or_default() += 1;
                }
            }
            let mut source_frames = source_support.into_iter().collect::<Vec<_>>();
            source_frames.sort_unstable_by_key(|&(source, support)| {
                (Reverse(support), frame.abs_diff(source), source)
            });
            let mut hypotheses = vec![None];
            hypotheses.extend(
                source_frames
                    .into_iter()
                    .take(31)
                    .map(|(source, _)| Some(source)),
            );
            let mut accepted = None;
            for source_filter in hypotheses {
                let hypothesis = if source_filter.is_none() {
                    candidates.clone()
                } else {
                    collect_deferred_correspondences(
                        frame,
                        &target_candidates[frame],
                        source_filter,
                        rig,
                        features,
                        image_assignment,
                        image_poses,
                        config,
                        tracks,
                        stereo_links,
                    )
                };
                let sensors = hypothesis
                    .iter()
                    .map(|candidate| candidate.sensor_index)
                    .collect::<HashSet<_>>()
                    .len();
                if hypothesis.len() < required_inliers || sensors < required_sensors {
                    continue;
                }
                update.pnp_attempts += 1;
                let correspondences = hypothesis
                    .iter()
                    .map(|candidate| GeneralizedCorrespondence2D3D {
                        sensor_index: candidate.sensor_index,
                        point2d: candidate.point2d,
                        point3d: candidate.point3d,
                        confidence: None,
                    })
                    .collect::<Vec<_>>();
                let Some(report) = pnp.estimate(rig, &correspondences) else {
                    update.pnp_estimation_failures += 1;
                    continue;
                };
                if report.inliers.len() < required_inliers {
                    update.pnp_inlier_rejections += 1;
                    continue;
                }
                accepted = Some((report, hypothesis));
                break;
            }
            let Some((report, candidates)) = accepted else {
                continue;
            };
            frame_poses[frame] = Some(report.pose);
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame]
                    .as_ref()
                    .expect("pose was just installed"),
                image_poses,
            );
            for &inlier in &report.inliers {
                let candidate = candidates[inlier];
                let Some(track_index) = candidate.track_index else {
                    continue;
                };
                let existing =
                    observation_track(&image_tracks[candidate.image], candidate.keypoint);
                if existing.is_some_and(|track| tracks[track].position.is_some())
                    || tracks[track_index]
                        .observations
                        .iter()
                        .any(|&(image, _)| image == candidate.image)
                {
                    continue;
                }
                if let Some(existing) = existing {
                    tracks[existing].observations.retain(|&(image, keypoint)| {
                        image != candidate.image || keypoint != candidate.keypoint
                    });
                }
                tracks[track_index]
                    .observations
                    .push((candidate.image, candidate.keypoint));
                insert_observation_track(
                    &mut image_tracks[candidate.image],
                    candidate.keypoint,
                    track_index,
                );
                update.observations_attached += 1;
            }
            update.registrations += 1;
            source_queue.push(frame);
            registered_any = true;
        }
        if !registered_any {
            break;
        }
    }
    update.interpolation_registrations = interpolate_deferred_pose_gaps(
        rig,
        frames,
        &adjacency,
        config.deferred_registration_max_interpolation_gap,
        frame_poses,
        image_poses,
    );
    if std::env::var_os("VISLOC_DEFERRED_DEBUG").is_some() {
        for frame in 0..frames.len() {
            if frame_poses[frame].is_none() && !adjacency[frame].is_empty() {
                eprintln!(
                    "rig-deferred-debug: frame={frame} adjacent_pairs={} raw_targets={} usable={} direct={} sensors={} required_inliers={} required_sensors={}",
                    adjacency[frame].len(),
                    target_candidates[frame].len(),
                    candidate_support[frame],
                    candidate_direct[frame],
                    candidate_sensors[frame],
                    required_inliers,
                    required_sensors,
                );
            }
        }
    }
    update
}

#[derive(Debug, Clone)]
pub(super) struct MotionBridgeCandidate {
    pub(super) frame: usize,
    pub(super) pose: Pose,
    pub(super) inliers: usize,
    pub(super) frame_gap: usize,
}

#[derive(Debug, Default)]
pub(super) struct MotionBridgeUpdate {
    pub(super) candidates: Vec<MotionBridgeCandidate>,
    pub(super) pair_visits: usize,
    pub(super) estimation_failures: usize,
    pub(super) rotation_rejections: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_motion_bridge_candidates(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    pair_adjacency: &[Vec<usize>],
    source_queue: &mut Vec<usize>,
    pair_visited: &mut [bool],
) -> MotionBridgeUpdate {
    let estimator = RelativePoseEstimator::default();
    let mut update = MotionBridgeUpdate::default();
    let mut best_by_frame = HashMap::<usize, MotionBridgeCandidate>::new();
    for source_frame in std::mem::take(source_queue) {
        let Some(source_frame_pose) = frame_poses[source_frame].as_ref() else {
            continue;
        };
        for &pair_index in &pair_adjacency[source_frame] {
            if pair_visited[pair_index] {
                continue;
            }
            let pair = &pairwise[pair_index];
            let frame_i = image_assignment[pair.image_i].0;
            let frame_j = image_assignment[pair.image_j].0;
            let (target_frame, source_is_i) = if frame_i == source_frame {
                (frame_j, true)
            } else {
                (frame_i, false)
            };
            if frame_poses[target_frame].is_some() {
                pair_visited[pair_index] = true;
                continue;
            }
            let frame_gap = source_frame.abs_diff(target_frame);
            if frame_gap == 0 || frame_gap > config.motion_bridge_max_frame_gap {
                pair_visited[pair_index] = true;
                continue;
            }
            let prior_frame = (1..=config.motion_bridge_max_frame_gap).find_map(|prior_gap| {
                let candidate = if target_frame > source_frame {
                    source_frame.checked_sub(prior_gap)
                } else {
                    source_frame.checked_add(prior_gap)
                }?;
                (candidate < frames.len() && frame_poses[candidate].is_some())
                    .then_some((candidate, prior_gap))
            });
            let Some((prior_frame, prior_gap)) = prior_frame else {
                pair_visited[pair_index] = true;
                continue;
            };
            let prior_pose = frame_poses[prior_frame]
                .as_ref()
                .expect("selected prior frame is registered");
            pair_visited[pair_index] = true;
            update.pair_visits += 1;
            let (source_image, target_image) = if source_is_i {
                (pair.image_i, pair.image_j)
            } else {
                (pair.image_j, pair.image_i)
            };
            let correspondences = pair
                .matches
                .iter()
                .map(|&(keypoint_i, keypoint_j)| {
                    let (source_keypoint, target_keypoint) = if source_is_i {
                        (keypoint_i, keypoint_j)
                    } else {
                        (keypoint_j, keypoint_i)
                    };
                    TwoViewCorrespondence::new(
                        features[source_image].keypoints[source_keypoint],
                        features[target_image].keypoints[target_keypoint],
                    )
                })
                .collect::<Vec<_>>();
            let source_sensor = &rig.sensors()[image_assignment[source_image].1];
            let target_sensor = &rig.sensors()[image_assignment[target_image].1];
            let Some(relative) = estimator.estimate_with_cameras(
                &correspondences,
                &source_sensor.camera,
                &target_sensor.camera,
            ) else {
                update.estimation_failures += 1;
                continue;
            };
            if relative.inliers.len() < config.motion_bridge_min_inliers {
                update.estimation_failures += 1;
                continue;
            }
            let observed_delta = source_frame_pose
                .world_to_camera
                .compose(&prior_pose.world_to_camera.inverse());
            let motion_ratio = frame_gap as f64 / prior_gap as f64;
            let predicted_delta = SE3::new(
                nalgebra::UnitQuaternion::from_scaled_axis(
                    observed_delta.rotation.scaled_axis() * motion_ratio,
                ),
                observed_delta.translation * motion_ratio,
            );
            let to_rig_rotation = |sensor_rotation| {
                target_sensor.sensor_from_rig.rotation.inverse()
                    * sensor_rotation
                    * source_sensor.sensor_from_rig.rotation
            };
            let mut rotations = vec![to_rig_rotation(relative.previous_to_current.rotation)];
            if let Some((alternate, _)) = relative.alternate {
                rotations.push(to_rig_rotation(alternate));
            }
            rotations.sort_by(|left, right| {
                left.angle_to(&predicted_delta.rotation)
                    .total_cmp(&right.angle_to(&predicted_delta.rotation))
            });
            let rotation = rotations[0];
            let rotation_deviation_deg = rotation.angle_to(&predicted_delta.rotation).to_degrees();
            if rotation_deviation_deg > config.motion_bridge_max_rotation_deviation_deg {
                update.rotation_rejections += 1;
                continue;
            }
            let bridge_delta = SE3::new(rotation, predicted_delta.translation);
            let candidate = MotionBridgeCandidate {
                frame: target_frame,
                pose: Pose {
                    world_to_camera: bridge_delta.compose(&source_frame_pose.world_to_camera),
                },
                inliers: relative.inliers.len(),
                frame_gap,
            };
            let replace = best_by_frame
                .get(&target_frame)
                .is_none_or(|current| candidate.inliers > current.inliers);
            if replace {
                best_by_frame.insert(target_frame, candidate);
            }
        }
    }
    update.candidates = best_by_frame.into_values().collect();
    update
}
