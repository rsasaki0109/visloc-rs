//! Rig track construction, metric anchoring and conflict recovery, temporal
//! quadrilaterals, sparse-cycle tracks and positioned-track completion /
//! filtering.

use super::*;

pub(super) fn build_rig_track_output(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
) -> TrackBuildOutput {
    match config.track_builder {
        RigTrackBuilder::LegacyUnionFind => {
            build_tracks_detailed(features.len(), pairwise, config.min_track_length.max(2))
        }
        RigTrackBuilder::ConflictPreserving => build_tracks_incremental_correspondence(
            features,
            pairwise,
            config.min_track_length.max(2),
        ),
        RigTrackBuilder::StreamOrderConflictPreserving => {
            build_tracks_incremental_correspondence_in_order(
                features,
                pairwise,
                config.min_track_length.max(2),
            )
        }
        RigTrackBuilder::PairConfidence => build_tracks_confidence_ordered(
            features.len(),
            pairwise,
            config.min_track_length.max(2),
        ),
        RigTrackBuilder::TrustedPrefixPairConfidence(pair_count) => {
            build_tracks_confidence_ordered_with_trusted_prefix(
                features.len(),
                pairwise,
                config.min_track_length.max(2),
                pair_count,
            )
        }
        RigTrackBuilder::SparseCycle => build_tracks_sparse_cycle(
            features,
            pairwise,
            config.min_track_length.max(2),
            None,
            false,
        ),
        RigTrackBuilder::MetricSparseCycle => build_tracks_sparse_cycle(
            features,
            pairwise,
            config.min_track_length.max(2),
            Some(image_assignment),
            false,
        ),
        RigTrackBuilder::MetricTemporalCycle => build_tracks_sparse_cycle(
            features,
            pairwise,
            config.min_track_length.max(2),
            Some(image_assignment),
            true,
        ),
        RigTrackBuilder::MetricTemporalQuadrilateral => {
            build_metric_temporal_quadrilaterals_in_frame_gap(
                features,
                pairwise,
                image_assignment,
                config.min_track_length.max(2),
                config.deferred_retriangulation_quadrilateral_min_frame_gap,
                config.deferred_retriangulation_quadrilateral_max_frame_gap,
            )
        }
        RigTrackBuilder::MetricAnchoredCycle => {
            let output = build_tracks_sparse_cycle(
                features,
                pairwise,
                config.min_track_length.max(2),
                Some(image_assignment),
                false,
            );
            retain_metric_anchored_tracks(output, image_assignment)
        }
    }
}

fn retain_metric_anchored_tracks(
    mut output: TrackBuildOutput,
    image_assignment: &[(usize, usize)],
) -> TrackBuildOutput {
    output
        .tracks
        .retain(|track| track_is_metric_anchored(track, image_assignment));
    output.stats.retained_tracks = output.tracks.len();
    output.stats.retained_observations = output.tracks.iter().map(Vec::len).sum();
    output
}

pub(super) fn track_is_metric_anchored(
    track: &[(usize, usize)],
    image_assignment: &[(usize, usize)],
) -> bool {
    let mut frame_sensor = HashMap::<usize, usize>::new();
    track.iter().any(|&(image, _)| {
        let (frame, sensor) = image_assignment[image];
        frame_sensor
            .insert(frame, sensor)
            .is_some_and(|previous| previous != sensor)
    })
}

/// Recover a single conservative 3-D track from each legacy conflict
/// component. Verified edges propose anchors, but already-registered rig poses
/// decide admission. Requiring a graph cycle and a synchronized multi-sensor
/// observation prevents an accidental two-view corridor match from creating a
/// scale-free landmark.
pub(super) fn recover_metric_conflict_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    conflicting_components: &[Vec<(usize, usize)>],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
) -> Vec<WorkingTrack> {
    type Observation = (usize, usize);
    if config.conflict_recovery_max_hypotheses == 0 {
        return Vec::new();
    }
    let mut component_of = HashMap::new();
    for (component_id, component) in conflicting_components.iter().enumerate() {
        for &observation in component {
            component_of.insert(observation, component_id);
        }
    }
    let mut edges_by_component =
        vec![Vec::<(Observation, Observation)>::new(); conflicting_components.len()];
    for pair in pairwise {
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let left = (pair.image_i, keypoint_i);
            let right = (pair.image_j, keypoint_j);
            let Some(&component_id) = component_of.get(&left) else {
                continue;
            };
            if component_of.get(&right) != Some(&component_id) {
                continue;
            }
            edges_by_component[component_id].push(if left <= right {
                (left, right)
            } else {
                (right, left)
            });
        }
    }
    for edges in &mut edges_by_component {
        edges.sort_unstable();
        edges.dedup();
    }

    let ray = |observation: Observation| {
        let pose = image_poses.get(observation.0)?.as_ref()?;
        let sensor_index = image_assignment.get(observation.0)?.1;
        let sensor = rig.sensors().get(sensor_index)?;
        let pixel = *features.get(observation.0)?.keypoints.get(observation.1)?;
        let normalized = sensor.camera.normalize_pixel(&pixel)?;
        let direction = pose
            .camera_to_world()
            .rotation
            .transform_vector(&Vector3::new(normalized.x, normalized.y, 1.0).normalize())
            .normalize();
        Some((pose.camera_center_world(), direction))
    };
    let reprojection_error = |observation: Observation, point: &Point3<f64>| {
        let pose = image_poses.get(observation.0)?.as_ref()?;
        let sensor_index = image_assignment.get(observation.0)?.1;
        let sensor = rig.sensors().get(sensor_index)?;
        let point_camera = pose.transform_world_point(point);
        if point_camera.z <= 0.0 {
            return None;
        }
        let projected = sensor.camera.project(&point_camera)?;
        let pixel = features.get(observation.0)?.keypoints.get(observation.1)?;
        Some((projected - pixel).norm())
    };

    let mut recovered = Vec::new();
    for (component, edges) in conflicting_components.iter().zip(&edges_by_component) {
        let mut adjacency = HashMap::<Observation, Vec<Observation>>::new();
        let mut anchors = Vec::new();
        for &(left, right) in edges {
            adjacency.entry(left).or_default().push(right);
            adjacency.entry(right).or_default().push(left);
            let (Some((left_center, left_ray)), Some((right_center, right_ray))) =
                (ray(left), ray(right))
            else {
                continue;
            };
            let angle = left_ray.dot(&right_ray).clamp(-1.0, 1.0).abs().acos();
            if angle.is_finite() {
                anchors.push((
                    angle,
                    left,
                    right,
                    left_center,
                    left_ray,
                    right_center,
                    right_ray,
                ));
            }
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
        anchors.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        let mut best: Option<(WorkingTrack, f64)> = None;
        for &(_angle, anchor_left, anchor_right, left_center, left_ray, right_center, right_ray) in
            anchors.iter().take(config.conflict_recovery_max_hypotheses)
        {
            let Some(point) =
                closest_ray_midpoint(&left_center, &left_ray, &right_center, &right_ray)
            else {
                continue;
            };
            let mut valid_errors = HashMap::<Observation, f64>::new();
            for &observation in component {
                let Some(error) = reprojection_error(observation, &point) else {
                    continue;
                };
                if error <= config.conflict_recovery_max_reprojection_error_px {
                    valid_errors.insert(observation, error);
                }
            }
            if !valid_errors.contains_key(&anchor_left) || !valid_errors.contains_key(&anchor_right)
            {
                continue;
            }
            let mut reachable = HashSet::from([anchor_left]);
            let mut pending = vec![anchor_left];
            while let Some(observation) = pending.pop() {
                for &neighbour in adjacency.get(&observation).into_iter().flatten() {
                    if valid_errors.contains_key(&neighbour) && reachable.insert(neighbour) {
                        pending.push(neighbour);
                    }
                }
            }
            let mut best_by_image = HashMap::<usize, (usize, f64)>::new();
            for observation in reachable {
                let error = valid_errors[&observation];
                let entry = best_by_image
                    .entry(observation.0)
                    .or_insert((observation.1, error));
                if error < entry.1 || (error == entry.1 && observation.1 < entry.0) {
                    *entry = (observation.1, error);
                }
            }
            let mut selected = best_by_image
                .iter()
                .map(|(&image, &(keypoint, _))| (image, keypoint))
                .collect::<Vec<_>>();
            selected.sort_unstable();
            if selected.len() < 3 || !track_is_metric_anchored(&selected, image_assignment) {
                continue;
            }
            let selected_set = selected.iter().copied().collect::<HashSet<_>>();
            let cycle_edges = edges
                .iter()
                .filter(|(left, right)| selected_set.contains(left) && selected_set.contains(right))
                .count();
            if cycle_edges < selected.len() {
                continue;
            }
            let mean_error = selected
                .iter()
                .map(|observation| valid_errors[observation])
                .sum::<f64>()
                / selected.len() as f64;
            if mean_error > config.conflict_recovery_max_mean_reprojection_px {
                continue;
            }
            let candidate = WorkingTrack {
                observations: selected,
                position: Some(point),
                metric_anchored: true,
            };
            let replace = best.as_ref().is_none_or(|(current, current_error)| {
                candidate.observations.len() > current.observations.len()
                    || (candidate.observations.len() == current.observations.len()
                        && mean_error < *current_error)
            });
            if replace {
                best = Some((candidate, mean_error));
            }
        }
        if let Some((track, _)) = best {
            recovered.push(track);
        }
    }
    recovered
}

#[derive(Debug)]
struct SparsePairLookup {
    image_i: usize,
    image_j: usize,
    i_to_j: Vec<u32>,
    j_to_i: Vec<u32>,
}

const SPARSE_PAIR_AMBIGUOUS: u32 = u32::MAX - 1;

impl SparsePairLookup {
    fn counterpart(&self, image: usize, keypoint: usize) -> Option<usize> {
        let value = if image == self.image_i {
            *self.i_to_j.get(keypoint)?
        } else if image == self.image_j {
            *self.j_to_i.get(keypoint)?
        } else {
            return None;
        };
        (value < SPARSE_PAIR_AMBIGUOUS).then_some(value as usize)
    }
}

fn unique_pair_counterpart(
    lookups: &[SparsePairLookup],
    pair_indices: &BTreeMap<(usize, usize), Vec<usize>>,
    image: usize,
    keypoint: usize,
    other_image: usize,
) -> Option<usize> {
    let key = if image < other_image {
        (image, other_image)
    } else {
        (other_image, image)
    };
    let mut counterpart = None;
    for &pair_index in pair_indices.get(&key)? {
        let lookup = &lookups[pair_index];
        let Some(candidate) = lookup.counterpart(image, keypoint) else {
            continue;
        };
        // A one-way minimum is not enough: reject many-to-one descriptor
        // ambiguities unless the reverse lookup selects the same endpoint.
        if lookup.counterpart(other_image, candidate) != Some(keypoint) {
            continue;
        }
        match counterpart {
            None => counterpart = Some(candidate),
            Some(previous) if previous == candidate => {}
            Some(_) => return None,
        }
    }
    counterpart
}

pub(super) fn build_metric_temporal_quadrilaterals(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_metric_temporal_quadrilaterals_in_frame_gap(
        features,
        pairwise,
        image_assignment,
        min_track_length,
        1,
        1,
    )
}

fn build_metric_temporal_quadrilaterals_in_frame_gap(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    min_track_length: usize,
    min_frame_gap: usize,
    max_frame_gap: usize,
) -> TrackBuildOutput {
    let mut lookups = Vec::with_capacity(pairwise.len());
    let mut pair_indices = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let mut i_to_j = vec![u32::MAX; features[pair.image_i].len()];
        let mut j_to_i = vec![u32::MAX; features[pair.image_j].len()];
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if keypoint_i >= i_to_j.len()
                || keypoint_j >= j_to_i.len()
                || keypoint_i >= SPARSE_PAIR_AMBIGUOUS as usize
                || keypoint_j >= SPARSE_PAIR_AMBIGUOUS as usize
            {
                continue;
            }
            let keypoint_j = keypoint_j as u32;
            let keypoint_i = keypoint_i as u32;
            let forward = &mut i_to_j[keypoint_i as usize];
            *forward = match *forward {
                u32::MAX => keypoint_j,
                previous if previous == keypoint_j => previous,
                _ => SPARSE_PAIR_AMBIGUOUS,
            };
            let reverse = &mut j_to_i[keypoint_j as usize];
            *reverse = match *reverse {
                u32::MAX => keypoint_i,
                previous if previous == keypoint_i => previous,
                _ => SPARSE_PAIR_AMBIGUOUS,
            };
        }
        lookups.push(SparsePairLookup {
            image_i: pair.image_i,
            image_j: pair.image_j,
            i_to_j,
            j_to_i,
        });
        let key = if pair.image_i < pair.image_j {
            (pair.image_i, pair.image_j)
        } else {
            (pair.image_j, pair.image_i)
        };
        pair_indices.entry(key).or_default().push(pair_index);
    }

    let frame_count = image_assignment
        .iter()
        .map(|&(frame, _)| frame)
        .max()
        .map_or(0, |maximum| maximum + 1);
    let mut frame_images = vec![BTreeMap::<usize, Option<usize>>::new(); frame_count];
    for (image, &(frame, sensor)) in image_assignment.iter().enumerate() {
        frame_images[frame]
            .entry(sensor)
            .and_modify(|slot| *slot = None)
            .or_insert(Some(image));
    }

    let mut quadrilaterals = BTreeSet::<Vec<(usize, usize)>>::new();
    for pair in pairwise {
        let (frame_i, sensor_i) = image_assignment[pair.image_i];
        let (frame_j, sensor_j) = image_assignment[pair.image_j];
        let frame_gap = frame_i.abs_diff(frame_j);
        if !(min_frame_gap..=max_frame_gap).contains(&frame_gap) || sensor_i != sensor_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if unique_pair_counterpart(
                &lookups,
                &pair_indices,
                pair.image_i,
                keypoint_i,
                pair.image_j,
            ) != Some(keypoint_j)
            {
                continue;
            }
            for (&other_sensor, &other_image_i) in &frame_images[frame_i] {
                if other_sensor == sensor_i {
                    continue;
                }
                let (Some(other_image_i), Some(other_image_j)) = (
                    other_image_i,
                    frame_images[frame_j].get(&other_sensor).copied().flatten(),
                ) else {
                    continue;
                };
                let Some(other_keypoint_i) = unique_pair_counterpart(
                    &lookups,
                    &pair_indices,
                    pair.image_i,
                    keypoint_i,
                    other_image_i,
                ) else {
                    continue;
                };
                let Some(other_keypoint_j) = unique_pair_counterpart(
                    &lookups,
                    &pair_indices,
                    pair.image_j,
                    keypoint_j,
                    other_image_j,
                ) else {
                    continue;
                };
                if unique_pair_counterpart(
                    &lookups,
                    &pair_indices,
                    other_image_i,
                    other_keypoint_i,
                    other_image_j,
                ) != Some(other_keypoint_j)
                {
                    continue;
                }
                let mut observations = vec![
                    (pair.image_i, keypoint_i),
                    (pair.image_j, keypoint_j),
                    (other_image_i, other_keypoint_i),
                    (other_image_j, other_keypoint_j),
                ];
                observations.sort_unstable();
                quadrilaterals.insert(observations);
            }
        }
    }

    let connected_components = quadrilaterals.len();
    let mut owned_observations = HashSet::new();
    let mut rejected_conflicts = 0usize;
    let mut tracks = Vec::new();
    for observations in quadrilaterals {
        if observations.len() < min_track_length {
            continue;
        }
        if observations
            .iter()
            .any(|observation| owned_observations.contains(observation))
        {
            rejected_conflicts += 1;
            continue;
        }
        owned_observations.extend(observations.iter().copied());
        tracks.push(observations);
    }
    TrackBuildOutput {
        stats: TrackBuildStats {
            input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
            connected_components,
            conflicting_components: rejected_conflicts,
            conflicting_observations: rejected_conflicts.saturating_mul(4),
            retained_tracks: tracks.len(),
            retained_observations: tracks.iter().map(Vec::len).sum(),
        },
        tracks,
        conflicting_components: Vec::new(),
    }
}

/// Return deterministic, observation-disjoint stereo-temporal
/// quadrilaterals for diagnostics and independent correspondence screening.
/// The returned tracks contain exactly four `(image, keypoint)` observations.
pub fn metric_temporal_quadrilateral_tracks(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
) -> Vec<Vec<(usize, usize)>> {
    build_metric_temporal_quadrilaterals(features, pairwise, image_assignment, 2).tracks
}

/// Return the same deterministic, observation-disjoint exact quadrilaterals
/// for an explicit temporal-baseline range. Invalid or reversed ranges yield
/// no tracks. This is intended for bounded long-range correspondence audits.
pub fn metric_temporal_quadrilateral_tracks_in_frame_gap(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    min_frame_gap: usize,
    max_frame_gap: usize,
) -> Vec<Vec<(usize, usize)>> {
    if min_frame_gap == 0 || min_frame_gap > max_frame_gap {
        return Vec::new();
    }
    build_metric_temporal_quadrilaterals_in_frame_gap(
        features,
        pairwise,
        image_assignment,
        2,
        min_frame_gap,
        max_frame_gap,
    )
    .tracks
}

#[derive(Debug, Clone, Copy)]
struct SparseCycleCandidate {
    metric_stereo: bool,
    frame_gap: u32,
    cycles: u16,
    pair_support: u32,
    pair_index: u32,
    match_index: u32,
}

pub(super) fn build_tracks_sparse_cycle(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    image_assignment: Option<&[(usize, usize)]>,
    temporal_first: bool,
) -> TrackBuildOutput {
    let mut lookups = Vec::with_capacity(pairwise.len());
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); features.len()];
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let mut i_to_j = vec![u32::MAX; features[pair.image_i].len()];
        let mut j_to_i = vec![u32::MAX; features[pair.image_j].len()];
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if keypoint_i >= i_to_j.len()
                || keypoint_j >= j_to_i.len()
                || keypoint_i > u32::MAX as usize
                || keypoint_j > u32::MAX as usize
            {
                continue;
            }
            i_to_j[keypoint_i] = i_to_j[keypoint_i].min(keypoint_j as u32);
            j_to_i[keypoint_j] = j_to_i[keypoint_j].min(keypoint_i as u32);
        }
        lookups.push(SparsePairLookup {
            image_i: pair.image_i,
            image_j: pair.image_j,
            i_to_j,
            j_to_i,
        });
        adjacency[pair.image_i].push((pair.image_j, pair_index));
        adjacency[pair.image_j].push((pair.image_i, pair_index));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable();
    }

    let cycle_count =
        |image_i: usize, keypoint_i: usize, image_j: usize, keypoint_j: usize| -> u16 {
            let left = &adjacency[image_i];
            let right = &adjacency[image_j];
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            let mut cycles = 0u16;
            while left_index < left.len() && right_index < right.len() {
                match left[left_index].0.cmp(&right[right_index].0) {
                    std::cmp::Ordering::Less => left_index += 1,
                    std::cmp::Ordering::Greater => right_index += 1,
                    std::cmp::Ordering::Equal => {
                        let neighbour = left[left_index].0;
                        let left_end = left[left_index..]
                            .iter()
                            .position(|entry| entry.0 != neighbour)
                            .map_or(left.len(), |offset| left_index + offset);
                        let right_end = right[right_index..]
                            .iter()
                            .position(|entry| entry.0 != neighbour)
                            .map_or(right.len(), |offset| right_index + offset);
                        if neighbour != image_i
                            && neighbour != image_j
                            && left[left_index..left_end].iter().any(|&(_, left_pair)| {
                                let Some(left_keypoint) =
                                    lookups[left_pair].counterpart(image_i, keypoint_i)
                                else {
                                    return false;
                                };
                                right[right_index..right_end]
                                    .iter()
                                    .any(|&(_, right_pair)| {
                                        lookups[right_pair].counterpart(image_j, keypoint_j)
                                            == Some(left_keypoint)
                                    })
                            })
                        {
                            cycles = cycles.saturating_add(1);
                        }
                        left_index = left_end;
                        right_index = right_end;
                    }
                }
            }
            cycles
        };

    let mut candidates = Vec::with_capacity(pairwise.iter().map(|pair| pair.matches.len()).sum());
    for (pair_index, pair) in pairwise.iter().enumerate() {
        for (match_index, &(keypoint_i, keypoint_j)) in pair.matches.iter().enumerate() {
            candidates.push(SparseCycleCandidate {
                metric_stereo: image_assignment.is_some_and(|assignment| {
                    let (frame_i, sensor_i) = assignment[pair.image_i];
                    let (frame_j, sensor_j) = assignment[pair.image_j];
                    frame_i == frame_j && sensor_i != sensor_j
                }),
                frame_gap: image_assignment.map_or(u32::MAX, |assignment| {
                    assignment[pair.image_i]
                        .0
                        .abs_diff(assignment[pair.image_j].0)
                        .min(u32::MAX as usize) as u32
                }),
                cycles: cycle_count(pair.image_i, keypoint_i, pair.image_j, keypoint_j),
                pair_support: pair.matches.len().min(u32::MAX as usize) as u32,
                pair_index: pair_index as u32,
                match_index: match_index as u32,
            });
        }
    }
    candidates.sort_unstable_by(|left, right| {
        let left_pair = &pairwise[left.pair_index as usize];
        let right_pair = &pairwise[right.pair_index as usize];
        let (left_keypoint_i, left_keypoint_j) = left_pair.matches[left.match_index as usize];
        let (right_keypoint_i, right_keypoint_j) = right_pair.matches[right.match_index as usize];
        right
            .metric_stereo
            .cmp(&left.metric_stereo)
            .then_with(|| {
                if temporal_first {
                    left.frame_gap.cmp(&right.frame_gap)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| right.cycles.cmp(&left.cycles))
            .then_with(|| right.pair_support.cmp(&left.pair_support))
            .then_with(|| left_pair.image_i.cmp(&right_pair.image_i))
            .then_with(|| left_pair.image_j.cmp(&right_pair.image_j))
            .then_with(|| left_keypoint_i.cmp(&right_keypoint_i))
            .then_with(|| left_keypoint_j.cmp(&right_keypoint_j))
    });

    let mut node_id = HashMap::<(usize, usize), usize>::new();
    let mut nodes = Vec::<(usize, usize)>::new();
    let mut parent = Vec::<usize>::new();
    let mut component_size = Vec::<usize>::new();
    let mut component_images = Vec::<HashSet<usize>>::new();
    let mut rejected_conflicts = 0usize;
    for candidate in candidates {
        let pair = &pairwise[candidate.pair_index as usize];
        let (keypoint_i, keypoint_j) = pair.matches[candidate.match_index as usize];
        let mut node = |observation: (usize, usize)| {
            *node_id.entry(observation).or_insert_with(|| {
                let id = nodes.len();
                nodes.push(observation);
                parent.push(id);
                component_size.push(1);
                component_images.push(HashSet::from([observation.0]));
                id
            })
        };
        let left = node((pair.image_i, keypoint_i));
        let right = node((pair.image_j, keypoint_j));
        let left_root = sparse_find(&mut parent, left);
        let right_root = sparse_find(&mut parent, right);
        if left_root == right_root {
            continue;
        }
        if component_images[left_root]
            .iter()
            .any(|image| component_images[right_root].contains(image))
        {
            rejected_conflicts += 1;
            continue;
        }
        let (root, child) = if component_size[left_root] > component_size[right_root]
            || (component_size[left_root] == component_size[right_root] && left_root < right_root)
        {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups = HashMap::<usize, Vec<(usize, usize)>>::new();
    for (node_index, &observation) in nodes.iter().enumerate() {
        let root = sparse_find(&mut parent, node_index);
        groups.entry(root).or_default().push(observation);
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        conflicting_components: rejected_conflicts,
        conflicting_observations: rejected_conflicts.saturating_mul(2),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for mut observations in groups.into_values() {
        if observations.len() < min_track_length {
            continue;
        }
        observations.sort_unstable();
        stats.retained_observations += observations.len();
        tracks.push(observations);
    }
    tracks.sort_unstable();
    stats.retained_tracks = tracks.len();
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

fn sparse_find(parent: &mut [usize], mut node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }
    while parent[node] != node {
        let next = parent[node];
        parent[node] = root;
        node = next;
    }
    root
}

pub(super) fn filter_positioned_track_observations(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    max_reprojection_error_px: f64,
    tracks: &mut [WorkingTrack],
) -> usize {
    let mut pruned = 0usize;
    for track in tracks {
        let Some(position) = track.position else {
            continue;
        };
        let mut registered = 0usize;
        let mut inliers = 0usize;
        for &(image, keypoint) in &track.observations {
            let Some(pose) = image_poses[image].as_ref() else {
                continue;
            };
            registered += 1;
            let sensor = &rig.sensors()[image_assignment[image].1];
            let point_camera = pose.transform_world_point(&position);
            if point_camera.z > 0.0
                && sensor
                    .camera
                    .project(&point_camera)
                    .is_some_and(|projected| {
                        (projected - features[image].keypoints[keypoint]).norm()
                            <= max_reprojection_error_px
                    })
            {
                inliers += 1;
            }
        }
        if inliers < 2 {
            track.position = None;
            pruned += registered;
            continue;
        }
        track.observations.retain(|&(image, keypoint)| {
            let Some(pose) = image_poses[image].as_ref() else {
                return true;
            };
            let sensor = &rig.sensors()[image_assignment[image].1];
            let point_camera = pose.transform_world_point(&position);
            point_camera.z > 0.0
                && sensor
                    .camera
                    .project(&point_camera)
                    .is_some_and(|projected| {
                        (projected - features[image].keypoints[keypoint]).norm()
                            <= max_reprojection_error_px
                    })
        });
        pruned += registered - inliers;
        track.metric_anchored = track_is_metric_anchored(&track.observations, image_assignment);
    }
    pruned
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct TrackCompletionUpdate {
    pub(super) passes: usize,
    pub(super) pair_visits: usize,
    pub(super) observations: usize,
    pub(super) reprojection_rejections: usize,
}

/// Complete positioned tracks through the verified sparse correspondence
/// graph without merging landmarks or changing the PnP observation index.
/// Dense per-keypoint owner slots make lookup O(1), so the bound is
/// O(keypoints + observations + passes * verified matches), never O(images²).
#[allow(clippy::too_many_arguments)]
pub(super) fn complete_positioned_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    max_passes: usize,
    max_reprojection_error_px: f64,
    tracks: &mut [WorkingTrack],
) -> TrackCompletionUpdate {
    const CONFLICTED_OWNER: usize = usize::MAX;

    // This index is deliberately private to completion. In particular, it is
    // not the mapper's `image_tracks` cache used for subsequent PnP attempts.
    let mut positioned_by_image = features
        .iter()
        .map(|feature| vec![None; feature.len()])
        .collect::<Vec<Vec<Option<usize>>>>();
    let mut unpositioned_by_image = features
        .iter()
        .map(|feature| vec![None; feature.len()])
        .collect::<Vec<Vec<Option<usize>>>>();
    for (track_index, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in &track.observations {
            if track.position.is_some() {
                let owner = &mut positioned_by_image[image][keypoint];
                match owner {
                    Some(previous) if *previous != track_index => *owner = Some(CONFLICTED_OWNER),
                    None => *owner = Some(track_index),
                    _ => {}
                }
            } else {
                unpositioned_by_image[image][keypoint].get_or_insert(track_index);
            }
        }
    }
    let mut update = TrackCompletionUpdate::default();
    for _ in 0..max_passes {
        update.passes += 1;
        let before = update.observations;
        for pair in pairwise {
            update.pair_visits += 1;
            if image_poses[pair.image_i].is_none() || image_poses[pair.image_j].is_none() {
                continue;
            }
            for &(keypoint_i, keypoint_j) in &pair.matches {
                let owner_i = positioned_by_image[pair.image_i][keypoint_i];
                let owner_j = positioned_by_image[pair.image_j][keypoint_j];
                let (source_track, target_image, target_keypoint) = match (owner_i, owner_j) {
                    (Some(track), None) if track != CONFLICTED_OWNER => {
                        (track, pair.image_j, keypoint_j)
                    }
                    (None, Some(track)) if track != CONFLICTED_OWNER => {
                        (track, pair.image_i, keypoint_i)
                    }
                    // Never merge two positioned tracks, and never propagate
                    // from an observation with ambiguous positioned ownership.
                    _ => continue,
                };
                if tracks[source_track]
                    .observations
                    .iter()
                    .any(|&(image, _)| image == target_image)
                {
                    continue;
                }
                let Some(position) = tracks[source_track].position else {
                    continue;
                };
                let pose = image_poses[target_image]
                    .as_ref()
                    .expect("registered target was checked above");
                let sensor = &rig.sensors()[image_assignment[target_image].1];
                let Some(projected) = sensor
                    .camera
                    .project(&pose.transform_world_point(&position))
                else {
                    update.reprojection_rejections += 1;
                    continue;
                };
                let error = (projected - features[target_image].keypoints[target_keypoint]).norm();
                if !error.is_finite() || error > max_reprojection_error_px {
                    update.reprojection_rejections += 1;
                    continue;
                }

                // If the observation belonged to an unpositioned raw track,
                // detach it there before attaching it to the positioned track.
                // No external/PnP index is updated.
                if let Some(old_track) = unpositioned_by_image[target_image][target_keypoint].take()
                {
                    tracks[old_track]
                        .observations
                        .retain(|&observation| observation != (target_image, target_keypoint));
                }
                tracks[source_track]
                    .observations
                    .push((target_image, target_keypoint));
                positioned_by_image[target_image][target_keypoint] = Some(source_track);
                update.observations += 1;
            }
        }
        if update.observations == before {
            break;
        }
    }
    update
}
