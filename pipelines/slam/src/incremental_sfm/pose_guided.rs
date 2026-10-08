//! Pose-guided track splitting, merging and conflict-track recovery.

use super::*;

pub(super) type TrackObservation = (usize, usize);
type TrackEdge = (TrackObservation, TrackObservation);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct PoseGuidedTrackSplitStats {
    pub(super) input_components: usize,
    pub(super) bridge_cuts: usize,
    pub(super) bridge_cut_components: usize,
    pub(super) bridge_cut_sizes: Vec<(usize, usize)>,
    pub(super) preserved_components: usize,
    pub(super) split_components: usize,
    pub(super) hypotheses_tested: usize,
    pub(super) emitted_tracks: usize,
    assigned_observations: usize,
    pub(super) discarded_observations: usize,
    /// Histogram for graph-support admissions.  Buckets 0..=6 are exact and
    /// bucket 7 contains support from seven or more distinct images.
    pub(super) graph_support_histogram: [usize; 8],
    pub(super) graph_supported_tracks: usize,
    pub(super) graph_length_two_tracks: usize,
    /// Number of complementary track unions accepted by the optional
    /// post-split merge pass.
    pub(super) merged_tracks: usize,
    /// Number of active track-pair groups whose posed union was tested.
    pub(super) merge_candidates_tested: usize,
}

#[derive(Debug, Clone, Default)]
pub(super) struct PoseGuidedTrackSplitOutput {
    pub(super) tracks: Vec<Vec<TrackObservation>>,
    pub(super) points: Vec<Option<Point3<f64>>>,
    /// For every final track formed by one or more post-split unions, retain
    /// the exact pre-merge fragments.  The mapper can therefore undo only a
    /// merge whose post-BA observations fail the ordinary hard gate, without
    /// discarding healthy unions from the same candidate model.
    pub(super) merge_restorations: Vec<PoseGuidedMergeRestoration>,
    pub(super) stats: PoseGuidedTrackSplitStats,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PoseGuidedMergeRestoration {
    /// Stable indices in the pre-merge split partition.
    pub(super) source_track_ids: Vec<usize>,
    pub(super) source_tracks: Vec<Vec<TrackObservation>>,
    pub(super) source_points: Vec<Option<Point3<f64>>>,
    pub(super) merged_track: Vec<TrackObservation>,
}

#[derive(Debug, Clone)]
struct PoseGuidedTrackCandidate {
    observations: Vec<TrackObservation>,
    point: Point3<f64>,
    median_reprojection_px: f64,
    mean_reprojection_px: f64,
    anchor: TrackEdge,
    parallax_rad: f64,
    graph_support_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PoseGuidedTrackMergeCandidate {
    left: usize,
    right: usize,
    observations: Vec<TrackObservation>,
    point: Point3<f64>,
    /// Number of distinct image-pair edges crossing the two tracks.  Multiple
    /// orientation/keypoint rows from one image pair count once: they are not
    /// independent multi-view support for a merge.
    cross_image_edges: usize,
    parallax_rad: f64,
    median_reprojection_px: f64,
    mean_reprojection_px: f64,
}

impl PartialEq for PoseGuidedTrackMergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PoseGuidedTrackMergeCandidate {}

impl PartialOrd for PoseGuidedTrackMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PoseGuidedTrackMergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.observations
            .len()
            .cmp(&other.observations.len())
            .then_with(|| self.cross_image_edges.cmp(&other.cross_image_edges))
            .then_with(|| self.parallax_rad.total_cmp(&other.parallax_rad))
            // Lower robust reprojection is preferred.  Reversing the operands
            // keeps BinaryHeap's largest item as the best candidate.
            .then_with(|| {
                other
                    .median_reprojection_px
                    .total_cmp(&self.median_reprojection_px)
            })
            .then_with(|| {
                other
                    .mean_reprojection_px
                    .total_cmp(&self.mean_reprojection_px)
            })
            // Stable physical track order is the final deterministic tie
            // break.  Smaller IDs win when every geometric score ties.
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

type PoseGuidedMergeTrack = (Vec<TrackObservation>, Option<Point3<f64>>);
type PoseGuidedTrackMergeOutput = (
    Vec<Vec<TrackObservation>>,
    Vec<Option<Point3<f64>>>,
    usize,
    usize,
);

pub(super) fn pose_guided_merge_reprojection_gate(
    config: &IncrementalSfmConfig,
    split_gate: f64,
) -> Option<f64> {
    let gate = config
        .pose_guided_merge_max_reprojection_error_px
        .unwrap_or(split_gate);
    (gate.is_finite() && gate > 0.0).then_some(gate)
}

/// Fit one prospective merged track against the complete fixed-pose model.
/// `config.max_reprojection_error_px` is the split-only gate supplied by the
/// caller.  Unlike a pair-only check, every observation must be finite,
/// front-facing, and within that gate after local point refinement.
#[allow(clippy::too_many_arguments)]
fn pose_guided_merge_fit(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    config: &IncrementalSfmConfig,
) -> Option<(Point3<f64>, f64, f64, f64)> {
    if observations.len() < 2
        || !config.max_reprojection_error_px.is_finite()
        || config.max_reprojection_error_px <= 0.0
    {
        return None;
    }
    let mut images = HashSet::new();
    if !observations.iter().all(|&(image, _)| images.insert(image)) {
        return None;
    }
    let pixels = observations
        .iter()
        .map(|&(image, keypoint)| {
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint))
                .copied()
                .map(|pixel| (image, pixel))
        })
        .collect::<Option<Vec<_>>>()?;
    let initial = triangulate_track(camera, poses, &pixels, config)?;
    let point =
        refine_pose_guided_point(camera, features, poses, observations, initial).unwrap_or(initial);
    if !point.coords.iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut errors = Vec::with_capacity(observations.len());
    for &(image, keypoint) in observations {
        let pose = poses.get(image)?.as_ref()?;
        let pixel = features.get(image)?.keypoints.get(keypoint)?;
        let error = reprojection_error_px(camera, pose, &point, pixel)?;
        if !error.is_finite() || error > config.max_reprojection_error_px {
            return None;
        }
        errors.push(error);
    }
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let parallax = track_max_parallax(poses, observations, &point);
    (median.is_finite() && mean.is_finite() && parallax.is_finite())
        .then_some((point, parallax, median, mean))
}

/// Build one deterministic candidate for a pair of currently active tracks.
#[allow(clippy::too_many_arguments)]
fn pose_guided_make_merge_candidate(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    left: usize,
    right: usize,
    left_track: &[TrackObservation],
    right_track: &[TrackObservation],
    cross_edges: &BTreeSet<TrackEdge>,
    config: &IncrementalSfmConfig,
) -> Option<PoseGuidedTrackMergeCandidate> {
    if cross_edges.is_empty() {
        return None;
    }
    let mut image_set = HashSet::new();
    if !left_track
        .iter()
        .chain(right_track)
        .all(|&(image, _)| image_set.insert(image))
    {
        return None;
    }
    let mut observations = left_track
        .iter()
        .chain(right_track)
        .copied()
        .collect::<Vec<_>>();
    observations.sort_unstable();
    observations.dedup();
    if observations.len() != left_track.len() + right_track.len() {
        return None;
    }
    let (point, parallax_rad, median_reprojection_px, mean_reprojection_px) =
        pose_guided_merge_fit(camera, features, poses, &observations, config)?;
    let cross_image_edges = cross_edges
        .iter()
        .filter(|&&(first, second)| first.0 != second.0)
        .map(|&(first, second)| (first.0.min(second.0), first.0.max(second.0)))
        .collect::<BTreeSet<_>>()
        .len();
    (cross_image_edges > 0).then_some(PoseGuidedTrackMergeCandidate {
        left,
        right,
        observations,
        point,
        cross_image_edges,
        parallax_rad,
        median_reprojection_px,
        mean_reprojection_px,
    })
}

/// Collect candidates involving one active track.  Candidate groups are
/// keyed by the other active track and by exact verified observation edge, so
/// an orientation-row permutation cannot change either the geometry or the
/// support score.  Only `other > track_id` is emitted to avoid duplicates.
#[allow(clippy::too_many_arguments)]
fn pose_guided_collect_merge_candidates(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    track_id: usize,
    active: &[Option<PoseGuidedMergeTrack>],
    observation_to_track: &HashMap<TrackObservation, usize>,
    edge_adjacency: &HashMap<TrackObservation, Vec<TrackObservation>>,
    config: &IncrementalSfmConfig,
    include_lower_ids: bool,
) -> (Vec<PoseGuidedTrackMergeCandidate>, usize) {
    let Some(Some((track, _))) = active.get(track_id) else {
        return (Vec::new(), 0);
    };
    let mut grouped = BTreeMap::<usize, BTreeSet<TrackEdge>>::new();
    for &observation in track {
        for &other in edge_adjacency
            .get(&observation)
            .into_iter()
            .flat_map(|neighbours| neighbours.iter())
        {
            let Some(&other_id) = observation_to_track.get(&other) else {
                continue;
            };
            if other_id == track_id
                || (!include_lower_ids && other_id < track_id)
                || active.get(other_id).and_then(Option::as_ref).is_none()
            {
                continue;
            }
            let edge = if observation <= other {
                (observation, other)
            } else {
                (other, observation)
            };
            grouped.entry(other_id).or_default().insert(edge);
        }
    }

    let mut candidates = Vec::new();
    let mut tested = 0usize;
    for (other_id, cross_edges) in grouped {
        let Some(Some((other_track, _))) = active.get(other_id) else {
            continue;
        };
        tested += 1;
        let (left_id, left_track, right_id, right_track) = if track_id < other_id {
            (track_id, track, other_id, other_track)
        } else {
            (other_id, other_track, track_id, track)
        };
        if let Some(candidate) = pose_guided_make_merge_candidate(
            camera,
            features,
            poses,
            left_id,
            right_id,
            left_track,
            right_track,
            &cross_edges,
            config,
        ) {
            candidates.push(candidate);
        }
    }
    (candidates, tested)
}

/// Merge complementary pose-guided tracks with a verified cross-track edge.
/// The active-track table gives each union a stable identity; stale heap
/// entries are discarded, while every candidate involving a newly merged
/// track is rebuilt from the new observation set.  Thus geometry is
/// recomputed after every accepted union without rescanning unrelated pairs.
#[allow(clippy::too_many_arguments)]
pub(super) fn pose_guided_merge_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    tracks: &[Vec<TrackObservation>],
    points: &[Option<Point3<f64>>],
    config: &IncrementalSfmConfig,
) -> PoseGuidedTrackMergeOutput {
    if tracks.len() != points.len() || tracks.is_empty() {
        return (tracks.to_vec(), points.to_vec(), 0, 0);
    }

    // Canonicalise the starting partition so the stable IDs used as final
    // tie-breaks are physical observation order, not caller traversal order.
    let mut initial = tracks
        .iter()
        .zip(points.iter().copied())
        .map(|(track, point)| {
            let mut track = track.clone();
            track.sort_unstable();
            (track, point)
        })
        .collect::<Vec<_>>();
    initial.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut observation_to_track = HashMap::<TrackObservation, usize>::new();
    let mut active = Vec::<Option<PoseGuidedMergeTrack>>::with_capacity(initial.len());
    for (track_id, (track, point)) in initial.into_iter().enumerate() {
        if track
            .iter()
            .any(|observation| observation_to_track.contains_key(observation))
        {
            // The splitter normally guarantees disjoint observations.  If a
            // future caller violates that invariant, keep the canonical input
            // untouched rather than silently assigning one row twice.
            let mut unchanged = active.into_iter().flatten().collect::<Vec<_>>();
            unchanged.push((track, point));
            unchanged.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            return (
                unchanged.iter().map(|(track, _)| track.clone()).collect(),
                unchanged.into_iter().map(|(_, point)| point).collect(),
                0,
                0,
            );
        }
        for &observation in &track {
            observation_to_track.insert(observation, track_id);
        }
        active.push(Some((track, point)));
    }

    let mut edge_set = BTreeSet::<TrackEdge>::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        for &(left, right) in &pair.matches {
            let left = (pair.image_i, left);
            let right = (pair.image_j, right);
            edge_set.insert(if left <= right {
                (left, right)
            } else {
                (right, left)
            });
        }
    }
    let mut edge_adjacency = HashMap::<TrackObservation, Vec<TrackObservation>>::new();
    for (left, right) in edge_set {
        edge_adjacency.entry(left).or_default().push(right);
        edge_adjacency.entry(right).or_default().push(left);
    }
    for neighbours in edge_adjacency.values_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    let mut heap = BinaryHeap::new();
    let mut candidates_tested = 0usize;
    for track_id in 0..active.len() {
        let (candidates, tested) = pose_guided_collect_merge_candidates(
            camera,
            features,
            poses,
            track_id,
            &active,
            &observation_to_track,
            &edge_adjacency,
            config,
            false,
        );
        candidates_tested += tested;
        heap.extend(candidates);
    }

    let mut merges = 0usize;
    while let Some(candidate) = heap.pop() {
        let Some(Some((left_track, _))) = active.get(candidate.left) else {
            continue;
        };
        let Some(Some((right_track, _))) = active.get(candidate.right) else {
            continue;
        };
        // Both tracks remain unchanged while active.  Refit the popped union
        // once more before committing; this guards against accidental future
        // changes to candidate generation and makes the acceptance condition
        // explicit at the mutation point.
        let mut cross_edges = BTreeSet::new();
        for &observation in &candidate.observations {
            for &other in edge_adjacency
                .get(&observation)
                .into_iter()
                .flat_map(|neighbours| neighbours.iter())
            {
                let Some(&left_id) = observation_to_track.get(&observation) else {
                    continue;
                };
                let Some(&right_id) = observation_to_track.get(&other) else {
                    continue;
                };
                if left_id == right_id
                    || !((left_id == candidate.left && right_id == candidate.right)
                        || (left_id == candidate.right && right_id == candidate.left))
                {
                    continue;
                }
                cross_edges.insert(if observation <= other {
                    (observation, other)
                } else {
                    (other, observation)
                });
            }
        }
        let Some(refit) = pose_guided_make_merge_candidate(
            camera,
            features,
            poses,
            candidate.left,
            candidate.right,
            left_track,
            right_track,
            &cross_edges,
            config,
        ) else {
            continue;
        };

        let new_id = active.len();
        active[candidate.left] = None;
        active[candidate.right] = None;
        for &observation in &refit.observations {
            observation_to_track.insert(observation, new_id);
        }
        active.push(Some((refit.observations, Some(refit.point))));
        merges += 1;

        let (new_candidates, tested) = pose_guided_collect_merge_candidates(
            camera,
            features,
            poses,
            new_id,
            &active,
            &observation_to_track,
            &edge_adjacency,
            config,
            true,
        );
        candidates_tested += tested;
        heap.extend(new_candidates);
    }

    let mut result = active.into_iter().flatten().collect::<Vec<_>>();
    result.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    (
        result.iter().map(|(track, _)| track.clone()).collect(),
        result.into_iter().map(|(_, point)| point).collect(),
        merges,
        candidates_tested,
    )
}

/// Recover the exact source fragments for every final track that contains a
/// post-split union.  The merge routine only joins complete, disjoint tracks,
/// so an observation-to-source lookup is sufficient and keeps this provenance
/// independent of heap traversal order.
pub(super) fn pose_guided_merge_restorations(
    source_tracks: &[Vec<TrackObservation>],
    source_points: &[Option<Point3<f64>>],
    merged_tracks: &[Vec<TrackObservation>],
) -> Vec<PoseGuidedMergeRestoration> {
    if source_tracks.len() != source_points.len() {
        return Vec::new();
    }
    let mut source_by_observation = HashMap::<TrackObservation, usize>::new();
    for (source_id, track) in source_tracks.iter().enumerate() {
        for &observation in track {
            if source_by_observation
                .insert(observation, source_id)
                .is_some()
            {
                // The splitter guarantees disjoint source tracks.  Refuse to
                // manufacture restoration data if a future caller violates
                // that invariant.
                return Vec::new();
            }
        }
    }

    let mut restorations = Vec::new();
    for merged_track in merged_tracks {
        let mut source_track_ids = merged_track
            .iter()
            .filter_map(|observation| source_by_observation.get(observation).copied())
            .collect::<Vec<_>>();
        if source_track_ids.len() != merged_track.len() {
            continue;
        }
        source_track_ids.sort_unstable();
        source_track_ids.dedup();
        if source_track_ids.len() < 2 {
            continue;
        }
        let source_tracks_for_restore = source_track_ids
            .iter()
            .map(|&source_id| source_tracks[source_id].clone())
            .collect::<Vec<_>>();
        let source_points_for_restore = source_track_ids
            .iter()
            .map(|&source_id| source_points[source_id])
            .collect::<Vec<_>>();
        restorations.push(PoseGuidedMergeRestoration {
            source_track_ids,
            source_tracks: source_tracks_for_restore,
            source_points: source_points_for_restore,
            merged_track: merged_track.clone(),
        });
    }
    restorations
}

pub(super) fn pose_guided_track_reprojection_valid(
    camera: &Camera,
    features: &[FeatureSet],
    track: &[TrackObservation],
    pose_list: &[Option<Pose>],
    point: Option<&Point3<f64>>,
    max_error: f64,
) -> bool {
    let Some(point) = point else {
        return false;
    };
    point.coords.iter().all(|value| value.is_finite())
        && max_error.is_finite()
        && max_error > 0.0
        && track.iter().all(|&(image, keypoint)| {
            let (Some(pose), Some(pixel)) = (
                pose_list.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint)),
            ) else {
                return false;
            };
            reprojection_error_px(camera, pose, point, pixel)
                .is_some_and(|error| error.is_finite() && error <= max_error)
        })
}

/// Restore only merged tracks that fail the ordinary post-BA hard gate.  The
/// caller reruns BA after this mutation; if that second solve fails, its outer
/// candidate snapshot restores the complete pre-split model.
pub(super) fn pose_guided_restore_invalid_merges(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    tracks: &mut Vec<Vec<TrackObservation>>,
    points: &mut Vec<Option<Point3<f64>>>,
    restorations: &[PoseGuidedMergeRestoration],
    max_error: f64,
) -> (usize, usize) {
    if tracks.len() != points.len() || restorations.is_empty() {
        return (restorations.len(), 0);
    }
    let mut invalid_tracks = BTreeSet::<Vec<TrackObservation>>::new();
    for restoration in restorations {
        if !tracks
            .iter()
            .any(|track| track == &restoration.merged_track)
        {
            continue;
        }
        if !pose_guided_track_reprojection_valid(
            camera,
            features,
            &restoration.merged_track,
            poses,
            tracks
                .iter()
                .position(|track| track == &restoration.merged_track)
                .and_then(|index| points[index].as_ref()),
            max_error,
        ) {
            invalid_tracks.insert(restoration.merged_track.clone());
        }
    }
    if invalid_tracks.is_empty() {
        return (restorations.len(), 0);
    }

    let mut restored = Vec::with_capacity(tracks.len() + invalid_tracks.len());
    for (track, point) in tracks.iter().zip(points.iter()) {
        if invalid_tracks.contains(track) {
            let restoration = restorations
                .iter()
                .find(|restoration| restoration.merged_track == *track)
                .expect("invalid merged track has restoration provenance");
            restored.extend(
                restoration
                    .source_tracks
                    .iter()
                    .cloned()
                    .zip(restoration.source_points.iter().copied()),
            );
        } else {
            restored.push((track.clone(), *point));
        }
    }
    restored.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    *tracks = restored.iter().map(|(track, _)| track.clone()).collect();
    *points = restored.into_iter().map(|(_, point)| point).collect();
    (restorations.len(), invalid_tracks.len())
}

/// Validate only the merged tracks that survived selective restoration.  The
/// split-only tracks have their own candidate/objective guard; a bad unrelated
/// split track must not make a healthy merge appear to be the culprit.
#[allow(clippy::too_many_arguments)]
pub(super) fn pose_guided_merge_restorations_reprojection_valid(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    tracks: &[Vec<TrackObservation>],
    points: &[Option<Point3<f64>>],
    restorations: &[PoseGuidedMergeRestoration],
    restored_merges: usize,
    max_error: f64,
) -> bool {
    if tracks.len() != points.len() || !max_error.is_finite() || max_error <= 0.0 {
        return false;
    }
    let mut active_merges = 0usize;
    for restoration in restorations {
        let Some(index) = tracks
            .iter()
            .position(|track| track == &restoration.merged_track)
        else {
            continue;
        };
        active_merges += 1;
        if !pose_guided_track_reprojection_valid(
            camera,
            features,
            &tracks[index],
            poses,
            points[index].as_ref(),
            max_error,
        ) {
            return false;
        }
    }
    // If every tentative merged track was restored, there is no surviving
    // union to validate; the outer split/objective guard still applies.  A
    // mismatch indicates lost provenance and is rejected conservatively.
    active_merges + restored_merges == restorations.len()
}

#[derive(Debug, Clone, Default)]
pub(super) struct PoseGuidedBridgeCutOutput {
    pub(super) components: Vec<Vec<TrackObservation>>,
    pub(super) cut_edges: Vec<TrackEdge>,
    pub(super) cut_sizes: Vec<(usize, usize)>,
}

type PoseGuidedComponentGraph = (
    HashMap<TrackObservation, usize>,
    Vec<Vec<TrackEdge>>,
    Vec<HashMap<TrackObservation, Vec<TrackObservation>>>,
);

/// Build the deterministic verified correspondence graph used by the
/// pose-guided diagnostics.  The returned component-local edge lists are
/// deduplicated and sorted, so callers can safely run graph algorithms without
/// depending on pair or match traversal order.
fn build_pose_guided_component_graph(
    components: &[Vec<TrackObservation>],
    pairwise: &[PairwiseMatches],
) -> Option<PoseGuidedComponentGraph> {
    let mut component_of = HashMap::<TrackObservation, usize>::new();
    for (component_id, component) in components.iter().enumerate() {
        for &observation in component {
            if component_of.insert(observation, component_id).is_some() {
                return None;
            }
        }
    }

    let mut edges_by_component = vec![Vec::<TrackEdge>::new(); components.len()];
    let mut adjacency_by_component =
        vec![HashMap::<TrackObservation, Vec<TrackObservation>>::new(); components.len()];
    let mut edge_seen = HashSet::<(usize, TrackEdge)>::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let left = (pair.image_i, keypoint_i);
            let right = (pair.image_j, keypoint_j);
            let (Some(&left_component), Some(&right_component)) =
                (component_of.get(&left), component_of.get(&right))
            else {
                continue;
            };
            if left_component != right_component {
                continue;
            }
            let edge = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            if edge_seen.insert((left_component, edge)) {
                edges_by_component[left_component].push(edge);
                adjacency_by_component[left_component]
                    .entry(edge.0)
                    .or_default()
                    .push(edge.1);
                adjacency_by_component[left_component]
                    .entry(edge.1)
                    .or_default()
                    .push(edge.0);
            }
        }
    }
    for edges in &mut edges_by_component {
        edges.sort_unstable();
    }
    for adjacency in &mut adjacency_by_component {
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
    }
    Some((component_of, edges_by_component, adjacency_by_component))
}

/// Find bridges with an iterative Tarjan DFS.  The iterative form avoids
/// recursion depth depending on a large correspondence component; every
/// adjacency and tie-break is sorted before traversal for permutation
/// invariance.
fn pose_guided_find_bridges(
    observations: &[TrackObservation],
    edges: &[TrackEdge],
) -> Vec<TrackEdge> {
    let mut nodes = observations.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    let node_index = nodes
        .iter()
        .enumerate()
        .map(|(index, &observation)| (observation, index))
        .collect::<HashMap<_, _>>();

    let mut valid_edges = Vec::<TrackEdge>::new();
    let mut edge_nodes = Vec::<(usize, usize)>::new();
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); nodes.len()];
    let mut seen = HashSet::new();
    for &edge @ (left, right) in edges {
        if left == right || !seen.insert(edge) {
            continue;
        }
        let (Some(&left_index), Some(&right_index)) =
            (node_index.get(&left), node_index.get(&right))
        else {
            continue;
        };
        let edge_index = valid_edges.len();
        valid_edges.push(edge);
        edge_nodes.push((left_index, right_index));
        adjacency[left_index].push((right_index, edge_index));
        adjacency[right_index].push((left_index, edge_index));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable();
    }

    #[derive(Clone, Copy)]
    struct Frame {
        node: usize,
        parent_edge: Option<usize>,
        next_neighbour: usize,
    }

    let unvisited = usize::MAX;
    let mut discovery = vec![unvisited; nodes.len()];
    let mut low = vec![unvisited; nodes.len()];
    let mut time = 0usize;
    let mut bridges = Vec::new();
    for root in 0..nodes.len() {
        if discovery[root] != unvisited {
            continue;
        }
        discovery[root] = time;
        low[root] = time;
        time += 1;
        let mut stack = vec![Frame {
            node: root,
            parent_edge: None,
            next_neighbour: 0,
        }];
        while let Some(frame) = stack.last_mut() {
            let node = frame.node;
            if frame.next_neighbour < adjacency[node].len() {
                let (neighbour, edge_index) = adjacency[node][frame.next_neighbour];
                frame.next_neighbour += 1;
                if frame.parent_edge == Some(edge_index) {
                    continue;
                }
                if discovery[neighbour] == unvisited {
                    discovery[neighbour] = time;
                    low[neighbour] = time;
                    time += 1;
                    stack.push(Frame {
                        node: neighbour,
                        parent_edge: Some(edge_index),
                        next_neighbour: 0,
                    });
                } else {
                    low[node] = low[node].min(discovery[neighbour]);
                }
            } else {
                let finished = stack.pop().expect("non-empty Tarjan stack");
                if let Some(parent_edge) = finished.parent_edge {
                    let (left, right) = edge_nodes[parent_edge];
                    let parent = if left == finished.node { right } else { left };
                    low[parent] = low[parent].min(low[finished.node]);
                    if low[finished.node] > discovery[parent] {
                        bridges.push(valid_edges[parent_edge]);
                    }
                }
            }
        }
    }
    bridges.sort_unstable();
    bridges.dedup();
    bridges
}

/// Return the two connected sides after removing one candidate bridge.
fn pose_guided_bridge_sides(
    observations: &[TrackObservation],
    edges: &[TrackEdge],
    bridge: TrackEdge,
) -> Option<(Vec<TrackObservation>, Vec<TrackObservation>)> {
    let mut adjacency = HashMap::<TrackObservation, Vec<TrackObservation>>::new();
    for &observation in observations {
        adjacency.entry(observation).or_default();
    }
    for &edge @ (left, right) in edges {
        if edge == bridge {
            continue;
        }
        adjacency.entry(left).or_default().push(right);
        adjacency.entry(right).or_default().push(left);
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }
    let mut left_side = HashSet::new();
    let mut stack = vec![bridge.0];
    while let Some(observation) = stack.pop() {
        if !left_side.insert(observation) {
            continue;
        }
        if let Some(neighbours) = adjacency.get(&observation) {
            for &neighbour in neighbours.iter().rev() {
                if !left_side.contains(&neighbour) {
                    stack.push(neighbour);
                }
            }
        }
    }
    if left_side.len() == observations.len() {
        return None;
    }
    let mut first = left_side.into_iter().collect::<Vec<_>>();
    let mut second = observations
        .iter()
        .copied()
        .filter(|observation| !first.contains(observation))
        .collect::<Vec<_>>();
    first.sort_unstable();
    second.sort_unstable();
    (!first.is_empty() && !second.is_empty()).then_some((first, second))
}

fn pose_guided_bridge_side_is_eligible(observations: &[TrackObservation]) -> bool {
    if observations.len() < 2 {
        return false;
    }
    let mut images = HashSet::new();
    observations.iter().all(|&(image, _)| images.insert(image)) && images.len() >= 2
}

/// Fit one posed point to a component side.  The gate is deliberately the
/// same split gate used by the existing splitter; no bridge-specific pixel or
/// parallax threshold is introduced.  A `None` result means that all side
/// observations cannot be explained by one finite, cheirality-valid point.
fn pose_guided_bridge_fit(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    config: &IncrementalSfmConfig,
) -> Option<Point3<f64>> {
    if observations.len() < 2 {
        return None;
    }
    let pixels = observations
        .iter()
        .map(|&(image, keypoint)| {
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint))
                .copied()
                .map(|pixel| (image, pixel))
        })
        .collect::<Option<Vec<_>>>()?;
    let initial = triangulate_track(camera, poses, &pixels, config)?;
    let point =
        refine_pose_guided_point(camera, features, poses, observations, initial).unwrap_or(initial);
    let mut sum = 0.0;
    let mut max_error = 0.0f64;
    for &(image, keypoint) in observations {
        let pose = poses.get(image)?.as_ref()?;
        let pixel = features.get(image)?.keypoints.get(keypoint)?;
        let error = reprojection_error_px(camera, pose, &point, pixel)?;
        if !error.is_finite() {
            return None;
        }
        sum += error;
        max_error = max_error.max(error);
    }
    let mean_error = sum / observations.len() as f64;
    (mean_error.is_finite()
        && max_error.is_finite()
        && mean_error <= config.max_reprojection_error_px
        && max_error <= config.max_reprojection_error_px)
        .then_some(point)
}

/// Iteratively cut accepted graph bridges, recomputing Tarjan structure after
/// every cut.  A bridge is accepted only when both sides are multi-view posed
/// fits and the combined observations fail the same one-point fit.  This
/// conservative combined-fit test preserves genuine sparse chains while
/// separating a false bridge between two physical points.
pub(super) fn pose_guided_bridge_cut_component(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    edges: &[TrackEdge],
    config: &IncrementalSfmConfig,
) -> PoseGuidedBridgeCutOutput {
    let mut initial = observations.to_vec();
    initial.sort_unstable();
    initial.dedup();
    let mut partitions = if initial.is_empty() {
        Vec::new()
    } else {
        vec![initial]
    };
    let mut output = PoseGuidedBridgeCutOutput::default();

    loop {
        let mut accepted = None;
        for (partition_index, partition) in partitions.iter().enumerate() {
            let partition_set = partition.iter().copied().collect::<HashSet<_>>();
            let partition_edges = edges
                .iter()
                .copied()
                .filter(|&(left, right)| {
                    partition_set.contains(&left) && partition_set.contains(&right)
                })
                .collect::<Vec<_>>();
            for bridge in pose_guided_find_bridges(partition, &partition_edges) {
                let Some((first, second)) =
                    pose_guided_bridge_sides(partition, &partition_edges, bridge)
                else {
                    continue;
                };
                if !pose_guided_bridge_side_is_eligible(&first)
                    || !pose_guided_bridge_side_is_eligible(&second)
                {
                    continue;
                }
                if pose_guided_bridge_fit(camera, features, poses, &first, config).is_none()
                    || pose_guided_bridge_fit(camera, features, poses, &second, config).is_none()
                {
                    continue;
                }
                let mut combined = first.clone();
                combined.extend(second.iter().copied());
                combined.sort_unstable();
                if pose_guided_bridge_fit(camera, features, poses, &combined, config).is_some() {
                    continue;
                }
                accepted = Some((partition_index, bridge, first, second));
                break;
            }
            if accepted.is_some() {
                break;
            }
        }

        let Some((partition_index, bridge, first, second)) = accepted else {
            break;
        };
        output.cut_edges.push(bridge);
        output.cut_sizes.push((first.len(), second.len()));
        partitions[partition_index] = first;
        partitions.push(second);
        partitions.sort_unstable();
    }

    output.components = partitions;
    output
}

/// Write a pose-guided candidate partition as a compact observation table for
/// offline topology comparisons.  This is deliberately not part of the
/// reconstruction output: the caller must opt in with
/// `VISLOC_SFM_DEBUG_POSE_SPLIT_DUMP=/path/to/file.tsv`.
pub(super) fn dump_pose_guided_track_split(
    path: &std::path::Path,
    tracks: &[Vec<TrackObservation>],
) -> std::io::Result<usize> {
    let mut output = std::fs::File::create(path)?;
    writeln!(output, "track_id\timage_index\tkeypoint_index")?;
    let mut observations = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in track {
            writeln!(output, "{track_id}\t{image}\t{keypoint}")?;
            observations += 1;
        }
    }
    Ok(observations)
}

/// Split legacy union components using the current complete camera model.
///
/// The ordinary union-find path intentionally discards every component that
/// contains two observations from one image.  Once all cameras have been
/// registered, however, their fixed poses provide a safe, GT-independent way
/// to test several 3-D hypotheses inside such a component.  This pass ranks
/// verified anchor edges by posed parallax, retains at most one observation per
/// image whose reprojection and cheirality are valid, locally refines each
/// accepted point, and removes its observations before searching the residual
/// component.  With `pose_guided_graph_support`, every observation after the
/// anchor also needs two direct verified supports from distinct hypothesis
/// images, and multi-view emissions need two independent cross-image edges.
/// Clean components that already fit their existing point are copied
/// byte-for-byte at the observation level.
///
/// A partial pose model is deliberately a no-op: classifying an unregistered
/// observation by an arbitrary image-space proxy would turn this diagnostic
/// into a new growth policy.  The caller may therefore run the pass only after
/// the initial reconstruction is complete, or explicitly compare an oracle
/// complete pose model through the existing initial-pose diagnostic.
#[allow(clippy::too_many_arguments)]
pub(super) fn pose_guided_split_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<TrackObservation>],
    conflicting_components: &[Vec<TrackObservation>],
    old_points: &[Option<Point3<f64>>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
) -> Option<PoseGuidedTrackSplitOutput> {
    if !poses.iter().all(Option::is_some)
        || config.conflict_recovery_max_hypotheses == 0
        || tracks.len() != old_points.len()
    {
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: pose-guided track split skipped complete={} hypotheses={} tracks={} points={}",
                poses.iter().all(Option::is_some),
                config.conflict_recovery_max_hypotheses,
                tracks.len(),
                old_points.len(),
            );
        }
        return None;
    }
    let split_max_reprojection_error = config
        .pose_guided_split_max_reprojection_error_px
        .unwrap_or(config.max_reprojection_error_px);
    if !split_max_reprojection_error.is_finite() || split_max_reprojection_error <= 0.0 {
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: pose-guided track split skipped invalid max reprojection gate={split_max_reprojection_error}"
            );
        }
        return None;
    }
    let mut split_config = config.clone();
    split_config.max_reprojection_error_px = split_max_reprojection_error;
    let merge_max_reprojection_error =
        pose_guided_merge_reprojection_gate(config, split_max_reprojection_error);
    if config.pose_guided_track_merging && merge_max_reprojection_error.is_none() {
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: pose-guided track split skipped invalid merge reprojection gate");
        }
        return None;
    }
    let mut merge_config = split_config.clone();
    if let Some(gate) = merge_max_reprojection_error {
        merge_config.max_reprojection_error_px = gate;
    }

    #[derive(Debug)]
    struct InputComponent {
        observations: Vec<TrackObservation>,
        old_point: Option<Point3<f64>>,
        was_conflicting: bool,
    }

    let mut components = Vec::with_capacity(tracks.len() + conflicting_components.len());
    for (track, point) in tracks.iter().zip(old_points.iter()) {
        components.push(InputComponent {
            observations: track.clone(),
            old_point: *point,
            was_conflicting: false,
        });
    }
    components.extend(
        conflicting_components
            .iter()
            .map(|component| InputComponent {
                observations: component.clone(),
                old_point: None,
                was_conflicting: true,
            }),
    );
    if components.is_empty() {
        return Some(PoseGuidedTrackSplitOutput::default());
    }

    let original_component_count = components.len();
    let original_observations = components
        .iter()
        .map(|component| component.observations.clone())
        .collect::<Vec<_>>();
    let Some((_, original_edges_by_component, _)) =
        build_pose_guided_component_graph(&original_observations, pairwise)
    else {
        // The legacy builder should produce disjoint components.  If a future
        // caller violates that invariant, leaving the original model untouched
        // is safer than assigning an observation to two new landmarks.
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: pose-guided track split skipped overlapping observation");
        }
        return None;
    };

    let mut bridge_cut_sizes = Vec::new();
    let mut bridge_cut_components = 0usize;
    if config.pose_guided_bridge_cuts {
        let mut refined_components = Vec::with_capacity(components.len());
        for (component_id, component) in components.into_iter().enumerate() {
            let bridge_cut = pose_guided_bridge_cut_component(
                camera,
                features,
                poses,
                &component.observations,
                &original_edges_by_component[component_id],
                &split_config,
            );
            if bridge_cut.cut_edges.is_empty() {
                refined_components.push(component);
                continue;
            }
            bridge_cut_components += 1;
            bridge_cut_sizes.extend(bridge_cut.cut_sizes);
            for observations in bridge_cut.components {
                refined_components.push(InputComponent {
                    observations,
                    // A cut invalidates the old point as a candidate for the
                    // new sides; each side must be posed/triangulated anew.
                    old_point: None,
                    was_conflicting: component.was_conflicting,
                });
            }
        }
        components = refined_components;
    }

    let component_observations = components
        .iter()
        .map(|component| component.observations.clone())
        .collect::<Vec<_>>();
    let Some((_, edges_by_component, adjacency_by_component)) =
        build_pose_guided_component_graph(&component_observations, pairwise)
    else {
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: pose-guided track split skipped overlapping cut side");
        }
        return None;
    };

    let mut output = PoseGuidedTrackSplitOutput {
        stats: PoseGuidedTrackSplitStats {
            input_components: original_component_count,
            bridge_cuts: bridge_cut_sizes.len(),
            bridge_cut_components,
            bridge_cut_sizes,
            ..PoseGuidedTrackSplitStats::default()
        },
        ..PoseGuidedTrackSplitOutput::default()
    };
    let minimum_support = config.min_track_length.max(2);
    let max_hypotheses = config.conflict_recovery_max_hypotheses;

    for (component_id, component) in components.iter().enumerate() {
        let mut unique_images = HashSet::new();
        let conflict_free = component
            .observations
            .iter()
            .all(|&(image, _)| unique_images.insert(image));

        // Preserve a clean, already-valid component exactly.  This is the
        // important non-regression path: merely enabling the diagnostic does
        // not perturb a track whose current point explains all observations.
        if conflict_free
            && component.observations.len() >= minimum_support
            && component
                .old_point
                .is_some_and(|point| point.coords.iter().all(|value| value.is_finite()))
            && component.observations.iter().all(|&(image, keypoint)| {
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint))
                    .and_then(|pixel| {
                        poses.get(image).and_then(Option::as_ref).and_then(|pose| {
                            reprojection_error_px(
                                camera,
                                pose,
                                &component.old_point.unwrap(),
                                pixel,
                            )
                        })
                    })
                    .is_some_and(|error| error <= split_max_reprojection_error)
            })
        {
            let mut preserved = component.observations.clone();
            preserved.sort_unstable();
            output.tracks.push(preserved);
            output.points.push(component.old_point);
            output.stats.preserved_components += 1;
            output.stats.emitted_tracks += 1;
            output.stats.assigned_observations += component.observations.len();
            continue;
        }

        let mut remaining: HashSet<TrackObservation> =
            component.observations.iter().copied().collect();
        let mut emitted_from_component = 0usize;
        while remaining.len() >= minimum_support {
            let mut ranked_edges = Vec::<(f64, TrackEdge)>::new();
            for &edge in &edges_by_component[component_id] {
                let (left, right) = edge;
                if !remaining.contains(&left) || !remaining.contains(&right) {
                    continue;
                }
                let (Some(pose_a), Some(pose_b), Some(pixel_a), Some(pixel_b)) = (
                    poses.get(left.0).and_then(Option::as_ref),
                    poses.get(right.0).and_then(Option::as_ref),
                    features
                        .get(left.0)
                        .and_then(|set| set.keypoints.get(left.1))
                        .copied(),
                    features
                        .get(right.0)
                        .and_then(|set| set.keypoints.get(right.1))
                        .copied(),
                ) else {
                    continue;
                };
                let Some(normalized_a) = camera.normalize_pixel(&pixel_a) else {
                    continue;
                };
                let Some(normalized_b) = camera.normalize_pixel(&pixel_b) else {
                    continue;
                };
                let bearing_a = pose_a.camera_to_world().rotation
                    * Vector3::new(normalized_a.x, normalized_a.y, 1.0).normalize();
                let bearing_b = pose_b.camera_to_world().rotation
                    * Vector3::new(normalized_b.x, normalized_b.y, 1.0).normalize();
                let parallax = bearing_a.dot(&bearing_b).clamp(-1.0, 1.0).abs().acos();
                if parallax.is_finite() {
                    ranked_edges.push((parallax, edge));
                }
            }
            ranked_edges.sort_unstable_by(|left, right| {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
            });

            let mut best: Option<PoseGuidedTrackCandidate> = None;
            for &(parallax, edge) in ranked_edges.iter().take(max_hypotheses) {
                output.stats.hypotheses_tested += 1;
                let (left, right) = edge;
                let (Some(pixel_a), Some(pixel_b)) = (
                    features
                        .get(left.0)
                        .and_then(|set| set.keypoints.get(left.1))
                        .copied(),
                    features
                        .get(right.0)
                        .and_then(|set| set.keypoints.get(right.1))
                        .copied(),
                ) else {
                    continue;
                };
                let anchor_observations = [(left.0, pixel_a), (right.0, pixel_b)];
                let Some(point) =
                    triangulate_track(camera, poses, &anchor_observations, &split_config)
                else {
                    continue;
                };
                let (mut selected, _) = if config.pose_guided_graph_support {
                    pose_guided_select_observations_with_graph_support(
                        camera,
                        features,
                        poses,
                        &remaining,
                        &point,
                        split_max_reprojection_error,
                        edge,
                        &adjacency_by_component[component_id],
                    )
                } else {
                    (
                        pose_guided_select_observations(
                            camera,
                            features,
                            poses,
                            &remaining,
                            &point,
                            split_max_reprojection_error,
                        ),
                        Vec::new(),
                    )
                };
                if selected.len() < minimum_support
                    || !selected.contains(&left)
                    || !selected.contains(&right)
                {
                    continue;
                }
                let refined = refine_pose_guided_point(camera, features, poses, &selected, point)
                    .unwrap_or(point);
                let (refined_selected, graph_support_counts) = if config.pose_guided_graph_support {
                    pose_guided_select_observations_with_graph_support(
                        camera,
                        features,
                        poses,
                        &remaining,
                        &refined,
                        split_max_reprojection_error,
                        edge,
                        &adjacency_by_component[component_id],
                    )
                } else {
                    (
                        pose_guided_select_observations(
                            camera,
                            features,
                            poses,
                            &remaining,
                            &refined,
                            split_max_reprojection_error,
                        ),
                        Vec::new(),
                    )
                };
                selected = refined_selected;
                if selected.len() < minimum_support
                    || !selected.contains(&left)
                    || !selected.contains(&right)
                {
                    continue;
                }
                let mut errors = selected
                    .iter()
                    .filter_map(|&(image, keypoint)| {
                        let pixel = features.get(image)?.keypoints.get(keypoint)?;
                        let pose = poses.get(image)?.as_ref()?;
                        reprojection_error_px(camera, pose, &refined, pixel)
                    })
                    .collect::<Vec<_>>();
                if errors.len() != selected.len() {
                    continue;
                }
                errors.sort_by(f64::total_cmp);
                let median_reprojection_px = errors[errors.len() / 2];
                let mean_reprojection_px = errors.iter().sum::<f64>() / errors.len() as f64;
                if !mean_reprojection_px.is_finite() {
                    continue;
                }
                let independent_supports = pose_guided_cross_image_support_count(
                    &selected,
                    &adjacency_by_component[component_id],
                );
                if config.pose_guided_graph_support
                    && selected.len() > 2
                    && independent_supports < 2
                {
                    continue;
                }
                let candidate = PoseGuidedTrackCandidate {
                    observations: selected,
                    point: refined,
                    median_reprojection_px,
                    mean_reprojection_px,
                    anchor: edge,
                    parallax_rad: parallax,
                    graph_support_counts,
                };
                let replace = best.as_ref().is_none_or(|current| {
                    candidate.observations.len() > current.observations.len()
                        || (candidate.observations.len() == current.observations.len()
                            && (candidate.median_reprojection_px < current.median_reprojection_px
                                || (candidate.median_reprojection_px
                                    == current.median_reprojection_px
                                    && (candidate.mean_reprojection_px
                                        < current.mean_reprojection_px
                                        || (candidate.mean_reprojection_px
                                            == current.mean_reprojection_px
                                            && (candidate.parallax_rad > current.parallax_rad
                                                || (candidate.parallax_rad
                                                    == current.parallax_rad
                                                    && candidate.anchor < current.anchor)))))))
                });
                if replace {
                    best = Some(candidate);
                }
            }

            let Some(candidate) = best else { break };
            if candidate.observations.len() < minimum_support {
                break;
            }
            for observation in &candidate.observations {
                remaining.remove(observation);
            }
            if config.pose_guided_graph_support {
                for support in &candidate.graph_support_counts {
                    let bucket = (*support).min(output.stats.graph_support_histogram.len() - 1);
                    output.stats.graph_support_histogram[bucket] += 1;
                }
                if candidate.observations.len() > 2 {
                    output.stats.graph_supported_tracks += 1;
                } else {
                    output.stats.graph_length_two_tracks += 1;
                }
            }
            output.tracks.push(candidate.observations);
            output.points.push(Some(candidate.point));
            output.stats.emitted_tracks += 1;
            output.stats.assigned_observations += output.tracks.last().unwrap().len();
            emitted_from_component += 1;
        }

        output.stats.discarded_observations += remaining.len();
        if emitted_from_component > 0 {
            output.stats.split_components += 1;
        } else if component.was_conflicting {
            // A conflicting component without a valid posed hypothesis is
            // intentionally omitted; accepting its old transitive closure
            // would recreate the exact same same-image conflict.
            output.stats.split_components += 1;
        }
    }

    let mut paired = output
        .tracks
        .into_iter()
        .zip(output.points)
        .collect::<Vec<_>>();
    for (track, _) in &mut paired {
        track.sort_unstable();
    }
    paired.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    output.tracks = paired.iter().map(|(track, _)| track.clone()).collect();
    output.points = paired.into_iter().map(|(_, point)| point).collect();
    if config.pose_guided_track_merging {
        let source_tracks = output.tracks.clone();
        let source_points = output.points.clone();
        let (merged_tracks, merged_points, merges, candidates_tested) = pose_guided_merge_tracks(
            camera,
            features,
            pairwise,
            poses,
            &output.tracks,
            &output.points,
            &merge_config,
        );
        output.merge_restorations =
            pose_guided_merge_restorations(&source_tracks, &source_points, &merged_tracks);
        output.tracks = merged_tracks;
        output.points = merged_points;
        output.stats.merged_tracks = merges;
        output.stats.merge_candidates_tested = candidates_tested;
    }
    Some(output)
}

pub(super) fn pose_guided_split_candidate_gate(
    candidate_support: usize,
    support_floor: usize,
    candidate_mean: f64,
    split_max_reprojection_error: f64,
) -> bool {
    candidate_support >= support_floor
        && candidate_mean.is_finite()
        && split_max_reprojection_error.is_finite()
        && split_max_reprojection_error > 0.0
        && candidate_mean <= split_max_reprojection_error
}

/// Apply the GT-independent acceptance guard shared by every outer split pass.
/// The first pass preserves the historical single-pass behavior: its BA must
/// lower the candidate partition's own mean error.  A later pass must also
/// strictly lower the already accepted model's mean, which provides a
/// deterministic early-stop condition for repeated rebuilding from the same
/// source components.
#[allow(clippy::too_many_arguments)]
pub(super) fn pose_guided_split_candidate_accepts(
    iteration: usize,
    registered_before: usize,
    registered_after: usize,
    support_floor: usize,
    candidate_support: usize,
    after_support: usize,
    mean_before: f64,
    candidate_mean: f64,
    after_mean: f64,
    split_max_reprojection_error: f64,
) -> bool {
    pose_guided_split_candidate_gate(
        candidate_support,
        support_floor,
        candidate_mean,
        split_max_reprojection_error,
    ) && registered_after >= registered_before
        && after_support >= support_floor
        && after_mean.is_finite()
        && after_mean <= candidate_mean + 1.0e-9
        && (iteration == 0 || after_mean + 1.0e-9 < mean_before)
}

/// Select the best currently posed observation for every image that explains
/// a candidate point within the existing pixel gate.  The input set is
/// converted to a sorted vector before traversal so HashSet iteration cannot
/// affect either the selected topology or its tie-breaks.
fn pose_guided_select_observations(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    remaining: &HashSet<TrackObservation>,
    point: &Point3<f64>,
    max_error: f64,
) -> Vec<TrackObservation> {
    let mut observations = remaining.iter().copied().collect::<Vec<_>>();
    observations.sort_unstable();
    let mut best_by_image = HashMap::<usize, (usize, f64)>::new();
    for (image, keypoint) in observations {
        let (Some(pose), Some(pixel)) = (
            poses.get(image).and_then(Option::as_ref),
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint)),
        ) else {
            continue;
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            continue;
        };
        if !error.is_finite() || error > max_error {
            continue;
        }
        let entry = best_by_image.entry(image).or_insert((keypoint, error));
        if error < entry.1 || (error == entry.1 && keypoint < entry.0) {
            *entry = (keypoint, error);
        }
    }
    let mut selected = best_by_image
        .into_iter()
        .map(|(image, (keypoint, _))| (image, keypoint))
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

/// Select a posed hypothesis with an explicit verified-graph support rule.
/// The two anchor observations are admitted from one verified edge.  Every
/// later observation must both reproject into the current point and have
/// direct verified edges to at least two distinct observations already in the
/// hypothesis.  The strongest support is admitted first, with reprojection
/// error and physical observation key as deterministic tie-breaks; newly
/// admitted observations can unlock the next round.
#[allow(clippy::too_many_arguments)]
fn pose_guided_select_observations_with_graph_support(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    remaining: &HashSet<TrackObservation>,
    point: &Point3<f64>,
    max_error: f64,
    anchor: TrackEdge,
    adjacency: &HashMap<TrackObservation, Vec<TrackObservation>>,
) -> (Vec<TrackObservation>, Vec<usize>) {
    if anchor.0 == anchor.1
        || anchor.0 .0 == anchor.1 .0
        || !remaining.contains(&anchor.0)
        || !remaining.contains(&anchor.1)
    {
        return (Vec::new(), Vec::new());
    }
    for &(image, keypoint) in &[anchor.0, anchor.1] {
        let (Some(pose), Some(pixel)) = (
            poses.get(image).and_then(Option::as_ref),
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint)),
        ) else {
            return (Vec::new(), Vec::new());
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            return (Vec::new(), Vec::new());
        };
        if !error.is_finite() || error > max_error {
            return (Vec::new(), Vec::new());
        }
    }

    let mut selected = vec![anchor.0, anchor.1];
    selected.sort_unstable();
    let mut selected_set = selected.iter().copied().collect::<HashSet<_>>();
    let mut selected_images = selected
        .iter()
        .map(|&(image, _)| image)
        .collect::<HashSet<_>>();
    let mut support_counts = Vec::new();

    loop {
        let mut best: Option<(usize, f64, TrackObservation)> = None;
        let mut observations = remaining.iter().copied().collect::<Vec<_>>();
        observations.sort_unstable();
        for observation @ (image, keypoint) in observations {
            if selected_images.contains(&image) {
                continue;
            }
            let support_images = adjacency
                .get(&observation)
                .into_iter()
                .flat_map(|neighbours| neighbours.iter())
                .filter(|neighbour| selected_set.contains(neighbour))
                .map(|&(support_image, _)| support_image)
                .collect::<HashSet<_>>();
            let support = support_images.len();
            if support < 2 {
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
            let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
                continue;
            };
            if !error.is_finite() || error > max_error {
                continue;
            }
            let replace = best.is_none_or(|(best_support, best_error, best_observation)| {
                support > best_support
                    || (support == best_support
                        && (error < best_error
                            || (error == best_error && observation < best_observation)))
            });
            if replace {
                best = Some((support, error, observation));
            }
        }
        let Some((support, _, observation @ (image, _))) = best else {
            break;
        };
        selected.push(observation);
        selected.sort_unstable();
        selected_set.insert(observation);
        selected_images.insert(image);
        support_counts.push(support);
    }
    (selected, support_counts)
}

/// Count independent cross-image correspondence supports inside one posed
/// hypothesis.  An image pair contributes once regardless of how many
/// feature rows happen to connect it; same-image edges are never counted.
fn pose_guided_cross_image_support_count(
    observations: &[TrackObservation],
    adjacency: &HashMap<TrackObservation, Vec<TrackObservation>>,
) -> usize {
    let selected = observations.iter().copied().collect::<HashSet<_>>();
    let mut image_pairs = BTreeSet::new();
    for &(image, keypoint) in observations {
        let observation = (image, keypoint);
        for &(other_image, other_keypoint) in adjacency
            .get(&observation)
            .into_iter()
            .flat_map(|neighbours| neighbours.iter())
        {
            if selected.contains(&(other_image, other_keypoint)) && image != other_image {
                image_pairs.insert((image.min(other_image), image.max(other_image)));
            }
        }
    }
    image_pairs.len()
}

/// Locally refine one pose-guided point while keeping all camera poses fixed.
/// The same projection Jacobian used by the BA conditioning diagnostics is
/// used here, with monotone squared-reprojection acceptance and a tiny
/// diagonal damping term for nearly collinear rays.
fn refine_pose_guided_point(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    initial: Point3<f64>,
) -> Option<Point3<f64>> {
    if !initial.coords.iter().all(|value| value.is_finite()) {
        return None;
    }
    let point_cost = |point: &Point3<f64>| -> Option<f64> {
        let mut cost = 0.0;
        let mut count = 0usize;
        for &(image, keypoint) in observations {
            let pose = poses.get(image)?.as_ref()?;
            let pixel = features.get(image)?.keypoints.get(keypoint)?;
            let projected = camera.project(&pose.transform_world_point(point))?;
            let residual = projected - *pixel;
            if !residual.iter().all(|value| value.is_finite()) {
                return None;
            }
            cost += residual.norm_squared();
            count += 1;
        }
        (count >= 2 && cost.is_finite()).then_some(cost)
    };
    let mut point = initial;
    let mut cost = point_cost(&point)?;
    for _ in 0..4 {
        let mut hessian = Matrix3::<f64>::zeros();
        let mut gradient = Vector3::<f64>::zeros();
        for &(image, keypoint) in observations {
            let pose = poses.get(image)?.as_ref()?;
            let pixel = features.get(image)?.keypoints.get(keypoint)?;
            let point_camera = pose.transform_world_point(&point);
            let projected = camera.project(&point_camera)?;
            let residual = projected - *pixel;
            let projection_jacobian = ba_point_projection_jacobian(camera, &point_camera)?;
            let rotation = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let jacobian = projection_jacobian * rotation;
            hessian += jacobian.transpose() * jacobian;
            gradient += jacobian.transpose() * residual;
        }
        let damping = hessian
            .diagonal()
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0)
            * 1.0e-8;
        let system = hessian + Matrix3::identity() * damping;
        let delta = system.lu().solve(&(-gradient))?;
        if !delta.iter().all(|value| value.is_finite()) || delta.norm() < 1.0e-10 {
            break;
        }
        let candidate = Point3::from(point.coords + delta);
        let candidate_cost = point_cost(&candidate)?;
        if candidate_cost + 1.0e-12 < cost {
            point = candidate;
            cost = candidate_cost;
        } else {
            break;
        }
    }
    Some(point)
}

#[derive(Debug, Clone)]
pub(super) struct RecoveredConflictTrack {
    pub(super) observations: Vec<(usize, usize)>,
    pub(super) point: Point3<f64>,
    pub(super) registered_observations: usize,
    pub(super) mean_reprojection_px: f64,
}

/// Split dropped union-find conflict components against an already-posed model.
///
/// This deliberately does not trust descriptor distance or image-pair support
/// as a global ordering (both were catastrophic on MH_03). A verified edge is
/// only an anchor proposal. The resulting 3D hypothesis must explain a unique
/// observation in at least three registered images, and those selected
/// observations must contain a cycle in the verified correspondence graph.
pub(super) fn recover_conflict_tracks_geometry(
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

pub(super) fn mean_reprojection_for_track_range(
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
