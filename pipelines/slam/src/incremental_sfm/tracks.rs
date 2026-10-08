//! Feature-track construction from pairwise matches (union-find, correspondence, confidence and geometry ordered builders).

use super::*;

#[derive(Debug, Clone, Default)]
pub(crate) struct TrackBuildOutput {
    pub(crate) tracks: Vec<Vec<(usize, usize)>>,
    pub(crate) conflicting_components: Vec<Vec<(usize, usize)>>,
    pub(crate) stats: TrackBuildStats,
}

pub(crate) fn build_track_output(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    camera: Option<&Camera>,
) -> TrackBuildOutput {
    let mut output = if config.incremental_correspondence_triangulation {
        build_tracks_incremental_correspondence(features, pairwise, config.min_track_length)
    } else if config.cycle_supported_tracks {
        build_tracks_cycle_supported(features, camera, pairwise, config.min_track_length)
    } else if config.geometric_confidence_tracks {
        if let Some(camera) = camera {
            build_tracks_geometric_confidence(features, camera, pairwise, config.min_track_length)
        } else {
            // The preflight API predates camera-aware residuals. Keep it useful
            // (and deterministic) when called without a camera by applying the
            // same explicit pair-level fallback used for non-E/H models.
            build_tracks_confidence_ordered(features.len(), pairwise, config.min_track_length)
        }
    } else if config.confidence_ordered_tracks {
        build_tracks_confidence_ordered(features.len(), pairwise, config.min_track_length)
    } else {
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
    };
    if config.stable_track_order {
        canonicalize_track_order(features, &mut output);
    }
    output
}

/// Number of coordinate units retained by the opt-in physical ordering key.
/// A micron in the image-coordinate units used by the feature files is far
/// below the precision that can affect a track decision, while the fixed grid
/// keeps the ordering independent of the source row number for normal feature
/// data.
const STABLE_TRACK_COORD_SCALE: f64 = 1_000_000.0;

fn stable_coordinate_key(value: f64) -> (u8, i64) {
    if value.is_finite() {
        (0, (value * STABLE_TRACK_COORD_SCALE).round() as i64)
    } else if value.is_nan() {
        (2, 0)
    } else if value.is_sign_negative() {
        (1, 0)
    } else {
        (3, 0)
    }
}

fn stable_descriptor_cmp(lhs: &[f32], rhs: &[f32]) -> Ordering {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| a.total_cmp(b))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| lhs.len().cmp(&rhs.len()))
}

/// Compare two observations by a physical key rather than by their feature
/// row indices. Descriptor contents are only a deterministic tie-break for
/// co-located rows (FeatureSet currently retains no SIFT scale/orientation
/// metadata); the final index tie-break affects only byte-identical duplicate
/// rows that have no observable physical distinction.
fn stable_observation_cmp(
    features: &[FeatureSet],
    lhs: &(usize, usize),
    rhs: &(usize, usize),
) -> Ordering {
    let image_order = lhs.0.cmp(&rhs.0);
    if image_order != Ordering::Equal {
        return image_order;
    }
    let lhs_point = features.get(lhs.0).and_then(|set| set.keypoints.get(lhs.1));
    let rhs_point = features.get(rhs.0).and_then(|set| set.keypoints.get(rhs.1));
    let coordinate_order = match (lhs_point, rhs_point) {
        (Some(lhs), Some(rhs)) => stable_coordinate_key(lhs.x)
            .cmp(&stable_coordinate_key(rhs.x))
            .then_with(|| stable_coordinate_key(lhs.y).cmp(&stable_coordinate_key(rhs.y))),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    if coordinate_order != Ordering::Equal {
        return coordinate_order;
    }
    let descriptor_order = match (
        features
            .get(lhs.0)
            .and_then(|set| set.descriptors.get(lhs.1)),
        features
            .get(rhs.0)
            .and_then(|set| set.descriptors.get(rhs.1)),
    ) {
        (Some(lhs), Some(rhs)) => stable_descriptor_cmp(lhs, rhs),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    descriptor_order.then_with(|| lhs.1.cmp(&rhs.1))
}

fn stable_track_cmp(
    features: &[FeatureSet],
    lhs: &[(usize, usize)],
    rhs: &[(usize, usize)],
) -> Ordering {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| stable_observation_cmp(features, lhs, rhs))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| lhs.len().cmp(&rhs.len()))
}

/// Apply the physical ordering to every sequence consumed by the incremental
/// mapper. In particular, `tracks` drives both initial triangulation order and
/// the PnP correspondence order; sorting only the final exported points would
/// leave the order-sensitive growth path unchanged.
fn canonicalize_track_order(features: &[FeatureSet], output: &mut TrackBuildOutput) {
    for track in &mut output.tracks {
        track.sort_by(|lhs, rhs| stable_observation_cmp(features, lhs, rhs));
    }
    output
        .tracks
        .sort_by(|lhs, rhs| stable_track_cmp(features, lhs, rhs));
    for component in &mut output.conflicting_components {
        component.sort_by(|lhs, rhs| stable_observation_cmp(features, lhs, rhs));
    }
    output
        .conflicting_components
        .sort_by(|lhs, rhs| stable_track_cmp(features, lhs, rhs));
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
    build_track_output(features, pairwise, config, None).stats
}

/// Union-find over `(image, keypoint)` nodes joined by pairwise matches. Returns
/// the consistent tracks (no two keypoints from the same image) spanning at
/// least `min_track_length` distinct images.
#[cfg(test)]
pub(super) fn build_tracks(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    build_tracks_with_stats(n_images, pairwise, min_track_length).0
}

#[cfg(test)]
pub(super) fn build_tracks_with_stats(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> (Vec<Vec<(usize, usize)>>, TrackBuildStats) {
    let output = build_tracks_detailed(n_images, pairwise, min_track_length);
    (output.tracks, output.stats)
}

pub(crate) fn build_tracks_detailed(
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

/// Accept a validated external observation partition as the mapper's track
/// topology.  The partition is intentionally copied and sorted so later
/// triangulation/PnP traversal is deterministic, while all point coordinates
/// are recomputed from the current camera and poses.  Validation of indices,
/// one observation per image, and cross-track ownership is performed by the
/// public incremental entry point before this helper is called.
pub(super) fn build_track_output_from_membership(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    membership: &[Vec<(usize, usize)>],
) -> Result<TrackBuildOutput, String> {
    let mut seen = HashMap::<(usize, usize), usize>::new();
    let mut tracks = Vec::with_capacity(membership.len());
    for (track_id, source_track) in membership.iter().enumerate() {
        let mut images = HashSet::new();
        let mut track = source_track.clone();
        track.sort_unstable();
        for &(image, keypoint) in &track {
            if image >= features.len() {
                return Err(format!(
                    "track {track_id} references image {image}, but only {} images are loaded",
                    features.len()
                ));
            }
            if keypoint >= features[image].keypoints.len()
                || keypoint >= features[image].descriptors.len()
            {
                return Err(format!(
                    "track {track_id} references image {image} keypoint {keypoint}, outside the loaded feature set"
                ));
            }
            if !images.insert(image) {
                return Err(format!(
                    "track {track_id} contains more than one observation from image {image}"
                ));
            }
            if let Some(previous_track) = seen.insert((image, keypoint), track_id) {
                return Err(format!(
                    "observation ({image},{keypoint}) belongs to tracks {previous_track} and {track_id}"
                ));
            }
        }
        if images.len() >= min_track_length {
            tracks.push(track);
        }
    }
    tracks.sort_unstable();
    let stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: membership.len(),
        retained_tracks: tracks.len(),
        retained_observations: tracks.iter().map(Vec::len).sum(),
        ..TrackBuildStats::default()
    };
    Ok(TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    })
}

/// Build tracks incrementally from individual verified correspondences.
///
/// The legacy builder first takes an unrestricted transitive closure and then
/// discards the whole component when one image occurs twice.  That is a useful
/// compatibility baseline, but one bad edge can consequently hide otherwise
/// valid observations from the mapper.  This builder keeps an explicit
/// observation-to-track map while adding edges: a free observation extends a
/// track, two disjoint tracks are merged, and an edge that would introduce a
/// second observation from one image is rejected in isolation.  Edges are
/// sorted by their physical integer key, so the result does not depend on the
/// order in which a snapshot or matcher happened to emit them.
pub(crate) fn build_tracks_incremental_correspondence(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_tracks_incremental_correspondence_impl(features, pairwise, min_track_length, true)
}

/// Build conflict-preserving tracks in the verified input stream order.
///
/// This is intentionally separate from the physical-key-order policy above.
/// A caller can first supply a frozen, trusted correspondence prefix and then
/// append lower-priority bridge pairs.  Conflicting bridge edges are therefore
/// rejected without allowing their numeric image/keypoint ids to displace the
/// trusted prefix.  The caller is responsible for binding and checksumming the
/// input order when reproducibility matters.
pub(crate) fn build_tracks_incremental_correspondence_in_order(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_tracks_incremental_correspondence_impl(features, pairwise, min_track_length, false)
}

fn build_tracks_incremental_correspondence_impl(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    sort_edges: bool,
) -> TrackBuildOutput {
    #[derive(Debug, Default)]
    struct WorkingTrack {
        observations: Vec<(usize, usize)>,
        images: BTreeSet<usize>,
        active: bool,
    }

    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        ..TrackBuildStats::default()
    };
    let mut edges = Vec::new();
    for pair in pairwise {
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (keypoint_i, keypoint_j) = if swapped {
                (keypoint_j, keypoint_i)
            } else {
                (keypoint_i, keypoint_j)
            };
            // Unlike the historical union-find, the opt-in path does not
            // create unusable nodes for malformed imported rows.  The source
            // stream remains counted in `input_correspondences` above.
            if image_i == image_j
                || image_i >= features.len()
                || image_j >= features.len()
                || keypoint_i >= features[image_i].keypoints.len()
                || keypoint_j >= features[image_j].keypoints.len()
            {
                continue;
            }
            edges.push((image_i, image_j, keypoint_i, keypoint_j));
        }
    }
    if sort_edges {
        edges.sort_unstable();
    }

    let mut tracks = Vec::<WorkingTrack>::new();
    let mut observation_to_track = HashMap::<(usize, usize), usize>::new();
    let mut conflicting_components = Vec::new();

    let mut reject_conflict = |left: (usize, usize), right: (usize, usize)| {
        stats.conflicting_components += 1;
        stats.conflicting_observations += 2;
        conflicting_components.push(vec![left, right]);
    };

    for (image_i, image_j, keypoint_i, keypoint_j) in edges {
        let left = (image_i, keypoint_i);
        let right = (image_j, keypoint_j);
        let left_track = observation_to_track.get(&left).copied();
        let right_track = observation_to_track.get(&right).copied();
        match (left_track, right_track) {
            (None, None) => {
                let track_id = tracks.len();
                let mut images = BTreeSet::new();
                images.insert(image_i);
                images.insert(image_j);
                tracks.push(WorkingTrack {
                    observations: vec![left, right],
                    images,
                    active: true,
                });
                observation_to_track.insert(left, track_id);
                observation_to_track.insert(right, track_id);
            }
            (Some(track_id), None) | (None, Some(track_id)) => {
                let (observation, image) = if left_track.is_some() {
                    (right, image_j)
                } else {
                    (left, image_i)
                };
                let track = &mut tracks[track_id];
                if track.active && !track.images.contains(&image) {
                    track.observations.push(observation);
                    track.images.insert(image);
                    observation_to_track.insert(observation, track_id);
                } else {
                    reject_conflict(left, right);
                }
            }
            (Some(left_id), Some(right_id)) if left_id == right_id => {}
            (Some(left_id), Some(right_id)) => {
                let left_images = tracks[left_id].images.clone();
                let right_images = tracks[right_id].images.clone();
                if left_images.intersection(&right_images).next().is_some() {
                    reject_conflict(left, right);
                    continue;
                }
                // Union-by-size bounds map updates for highly connected view
                // graphs.  The id tie-break keeps equal-size cases stable.
                let (keep, drop) = if tracks[left_id].observations.len()
                    > tracks[right_id].observations.len()
                    || (tracks[left_id].observations.len() == tracks[right_id].observations.len()
                        && left_id < right_id)
                {
                    (left_id, right_id)
                } else {
                    (right_id, left_id)
                };
                let dropped = std::mem::take(&mut tracks[drop].observations);
                for observation in dropped {
                    observation_to_track.insert(observation, keep);
                    tracks[keep].observations.push(observation);
                }
                let dropped_images = std::mem::take(&mut tracks[drop].images);
                tracks[keep].images.extend(dropped_images);
                tracks[drop].active = false;
            }
        }
    }

    stats.connected_components = tracks.iter().filter(|track| track.active).count();
    let mut retained = Vec::new();
    for track in tracks.into_iter().filter(|track| track.active) {
        if track.observations.len() < min_track_length {
            continue;
        }
        let mut observations = track.observations;
        observations.sort_unstable();
        retained.push(observations);
    }
    retained.sort_unstable();
    conflicting_components.sort_unstable();
    stats.retained_tracks = retained.len();
    stats.retained_observations = retained.iter().map(Vec::len).sum();
    TrackBuildOutput {
        tracks: retained,
        conflicting_components,
        stats,
    }
}

/// Small, explicit observation-to-point state used by the incremental
/// correspondence triangulator.  Keeping this state separate from the
/// mapper's exported `SfmTrack` makes create/continue/merge conflict rules
/// unit-testable without a camera or a solver.
#[derive(Debug, Clone, Default)]
pub(super) struct CorrespondencePointState {
    pub(super) observation_to_point: HashMap<(usize, usize), usize>,
    pub(super) observations: Vec<Vec<(usize, usize)>>,
    pub(super) points: Vec<Option<Point3<f64>>>,
}

impl CorrespondencePointState {
    pub(super) fn from_tracks(
        tracks: &[Vec<(usize, usize)>],
        points: &[Option<Point3<f64>>],
    ) -> Self {
        let mut state = Self {
            observations: tracks.to_vec(),
            points: points.to_vec(),
            ..Self::default()
        };
        state.points.resize(tracks.len(), None);
        for (point_id, track) in tracks.iter().enumerate() {
            for &observation in track {
                state
                    .observation_to_point
                    .entry(observation)
                    .or_insert(point_id);
            }
        }
        state
    }

    #[cfg(test)]
    fn has_image(&self, point_id: usize, image: usize) -> bool {
        self.observations
            .get(point_id)
            .is_some_and(|track| track.iter().any(|&(track_image, _)| track_image == image))
    }

    #[cfg(test)]
    pub(super) fn create_point(
        &mut self,
        observations: &[(usize, usize)],
        point: Point3<f64>,
    ) -> Option<usize> {
        if !point.coords.iter().all(|value| value.is_finite())
            || observations.is_empty()
            || observations.iter().enumerate().any(|(index, &(image, _))| {
                observations[..index]
                    .iter()
                    .any(|&(other_image, other_kp)| {
                        other_image == image
                            || self
                                .observation_to_point
                                .contains_key(&(other_image, other_kp))
                    })
            })
        {
            return None;
        }
        let point_id = self.observations.len();
        self.observations.push(observations.to_vec());
        self.points.push(Some(point));
        for &observation in observations {
            self.observation_to_point.insert(observation, point_id);
        }
        Some(point_id)
    }

    #[cfg(test)]
    pub(super) fn continue_point(&mut self, point_id: usize, observation: (usize, usize)) -> bool {
        if point_id >= self.observations.len()
            || self.observation_to_point.contains_key(&observation)
            || self.has_image(point_id, observation.0)
        {
            return false;
        }
        self.observations[point_id].push(observation);
        self.observation_to_point.insert(observation, point_id);
        true
    }

    #[cfg(test)]
    pub(super) fn merge_points(&mut self, left: usize, right: usize, point: Point3<f64>) -> bool {
        if left >= self.observations.len()
            || right >= self.observations.len()
            || left == right
            || !point.coords.iter().all(|value| value.is_finite())
        {
            return false;
        }
        if self.observations[left]
            .iter()
            .any(|&(image, _)| self.has_image(right, image))
        {
            return false;
        }
        let right_observations = std::mem::take(&mut self.observations[right]);
        for observation in right_observations {
            self.observation_to_point.insert(observation, left);
            self.observations[left].push(observation);
        }
        self.points[left] = Some(point);
        self.points[right] = None;
        true
    }

    pub(super) fn retriangulate_point(&mut self, point_id: usize, point: Point3<f64>) -> bool {
        if point_id >= self.points.len() || !point.coords.iter().all(|value| value.is_finite()) {
            return false;
        }
        let changed = self.points[point_id] != Some(point);
        self.points[point_id] = Some(point);
        changed
    }
}

#[derive(Clone, Copy)]
struct ConfidenceCandidate {
    image_i: usize,
    image_j: usize,
    keypoint_i: usize,
    keypoint_j: usize,
    verified_inliers: usize,
    essential_inliers: usize,
    trusted: bool,
}

fn confidence_ordered_candidates(
    pairwise: &[PairwiseMatches],
    trusted_prefix: usize,
) -> Vec<ConfidenceCandidate> {
    let trusted_prefix = trusted_prefix.min(pairwise.len());
    let mut candidates = Vec::new();
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let essential_inliers = pair.essential_matches.as_ref().map_or(0, Vec::len);
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (keypoint_i, keypoint_j) = if swapped {
                (keypoint_j, keypoint_i)
            } else {
                (keypoint_i, keypoint_j)
            };
            candidates.push(ConfidenceCandidate {
                image_i,
                image_j,
                keypoint_i,
                keypoint_j,
                verified_inliers: pair.matches.len(),
                essential_inliers,
                trusted: pair_index < trusted_prefix,
            });
        }
    }
    // Stronger verified pairs first. Every remaining field makes ties
    // independent of the input pair/vector order; duplicate candidates are
    // harmless because the second one finds the same component.
    candidates.sort_unstable_by(|a, b| {
        b.trusted
            .cmp(&a.trusted)
            .then_with(|| b.verified_inliers.cmp(&a.verified_inliers))
            .then_with(|| b.essential_inliers.cmp(&a.essential_inliers))
            .then_with(|| a.image_i.cmp(&b.image_i))
            .then_with(|| a.image_j.cmp(&b.image_j))
            .then_with(|| a.keypoint_i.cmp(&b.keypoint_i))
            .then_with(|| a.keypoint_j.cmp(&b.keypoint_j))
    });
    candidates
}

/// Build tracks in a deterministic confidence order while refusing a merge
/// that would put two observations from the same image in one component.
///
/// `PairwiseMatches` retains pair-level verified inlier counts (and, when the
/// full verifier ran, an essential-inlier count), but not per-match residuals
/// or descriptor distances. Those retained geometric counts are therefore the
/// complete confidence signal used here; no synthetic per-match score is
/// invented. The legacy builder remains untouched and is still the default.
pub(crate) fn build_tracks_confidence_ordered(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_tracks_confidence_ordered_impl(n_images, pairwise, min_track_length, 0)
}

/// Build confidence-ordered tracks while processing a trusted pair prefix
/// before every remaining pair.
///
/// Confidence ordering is retained independently inside both tiers. A frozen
/// base snapshot can therefore establish its proven tracks before newly
/// verified component bridges compete for observations.
pub(crate) fn build_tracks_confidence_ordered_with_trusted_prefix(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    trusted_pair_prefix: usize,
) -> TrackBuildOutput {
    build_tracks_confidence_ordered_impl(
        n_images,
        pairwise,
        min_track_length,
        trusted_pair_prefix.min(pairwise.len()),
    )
}

/// Preview the conflict topology produced by confidence-ordered track
/// construction without materializing tracks or changing any mapper state.
///
/// The candidate order and image-compatible union-find decisions mirror
/// [`build_tracks_confidence_ordered_impl`]. Rejected edge endpoints are then
/// projected onto the final successful-component roots and connected by a
/// second compact union-find. The resulting regions are therefore sparse
/// summaries of the rejected topology, not a component-pair matrix. Apart
/// from the deterministic candidate sort and PairConfidence image-set
/// conflict checks, the second compact DSU pass over rejected endpoints and
/// final roots is linear in the touched observations and rejected edges; no
/// N² state is allocated.
pub fn preview_pair_confidence_conflicts(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    trusted_prefix: usize,
) -> PairConfidenceConflictStats {
    let _ = n_images;
    let candidates = confidence_ordered_candidates(pairwise, trusted_prefix);

    let mut node_id = HashMap::<(usize, usize), usize>::new();
    let mut nodes = Vec::<(usize, usize)>::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images = Vec::<HashSet<usize>>::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };

    let mut accepted_edges = 0usize;
    let mut rejected_endpoints = Vec::<(usize, usize)>::new();
    let mut max_overlapping_images_per_rejected_edge = 0usize;
    for candidate in candidates {
        let left = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let right = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let left_root = find(&mut parent, left);
        let right_root = find(&mut parent, right);
        if left_root == right_root {
            continue;
        }
        let overlap = if component_images[left_root].len() <= component_images[right_root].len() {
            component_images[left_root]
                .iter()
                .filter(|image| component_images[right_root].contains(image))
                .count()
        } else {
            component_images[right_root]
                .iter()
                .filter(|image| component_images[left_root].contains(image))
                .count()
        };
        if overlap != 0 {
            rejected_endpoints.push((left, right));
            max_overlapping_images_per_rejected_edge =
                max_overlapping_images_per_rejected_edge.max(overlap);
            continue;
        }
        accepted_edges += 1;
        // Union-by-size is the same movement bound and root tie-break as the
        // confidence builder; component image sets are moved, never copied.
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

    let mut node_roots = Vec::with_capacity(nodes.len());
    let mut final_roots = Vec::new();
    for node in 0..nodes.len() {
        let root = find(&mut parent, node);
        node_roots.push(root);
        final_roots.push(root);
    }
    final_roots.sort_unstable();
    final_roots.dedup();
    let root_to_component = final_roots
        .iter()
        .enumerate()
        .map(|(component, &root)| (root, component))
        .collect::<HashMap<_, _>>();

    // The rejected-edge graph is built only over final roots that actually
    // participate in a rejection. Thus a long chain collapses to one region,
    // while unrelated successful components remain entirely untouched.
    let mut region_parent = (0..final_roots.len()).collect::<Vec<_>>();
    let mut involved = HashSet::new();
    for &(left, right) in &rejected_endpoints {
        let left_component = root_to_component[&node_roots[left]];
        let right_component = root_to_component[&node_roots[right]];
        involved.insert(left_component);
        involved.insert(right_component);
        union(&mut region_parent, left_component, right_component);
    }
    let mut region_members = HashMap::<usize, Vec<usize>>::new();
    for component in involved {
        let region = find(&mut region_parent, component);
        region_members.entry(region).or_default().push(component);
    }

    let mut histogram = BTreeMap::<usize, usize>::new();
    let mut involved_components = 0usize;
    let mut involved_observations = 0usize;
    let mut max_region_components = 0usize;
    let mut max_region_observations = 0usize;
    for members in region_members.values_mut() {
        members.sort_unstable();
        members.dedup();
        let region_components = members.len();
        let region_observations = members
            .iter()
            .map(|&component| component_size[final_roots[component]])
            .sum::<usize>();
        *histogram.entry(region_components).or_default() += 1;
        involved_components += region_components;
        involved_observations += region_observations;
        max_region_components = max_region_components.max(region_components);
        max_region_observations = max_region_observations.max(region_observations);
    }

    PairConfidenceConflictStats {
        correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        nodes: nodes.len(),
        accepted_edges,
        rejected_edges: rejected_endpoints.len(),
        final_components: final_roots.len(),
        conflict_regions: region_members.len(),
        involved_components,
        involved_observations,
        max_region_components,
        max_region_observations,
        max_overlapping_images_per_rejected_edge,
        region_component_count_histogram: histogram.into_iter().collect(),
    }
}

fn build_tracks_confidence_ordered_impl(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    trusted_pair_prefix: usize,
) -> TrackBuildOutput {
    let _ = n_images;
    let candidates = confidence_ordered_candidates(pairwise, trusted_pair_prefix);

    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images: Vec<HashSet<usize>> = Vec::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };
    let mut rejected_conflicts = 0usize;
    for candidate in candidates {
        let a = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let b = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            continue;
        }
        if component_images[ra]
            .iter()
            .any(|image| component_images[rb].contains(image))
        {
            rejected_conflicts += 1;
            continue;
        }
        // Union-by-size bounds the set movement; the root-id tie-break keeps
        // the topology independent of HashMap/set iteration details.
        let (root, child) = if component_size[ra] > component_size[rb]
            || (component_size[ra] == component_size[rb] && ra < rb)
        {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, keypoint)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, keypoint));
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for (_root, mut observations) in groups {
        observations.sort_unstable();
        let distinct_images = observations
            .iter()
            .map(|&(image, _)| image)
            .collect::<HashSet<_>>()
            .len();
        if distinct_images >= min_track_length {
            stats.retained_observations += observations.len();
            tracks.push(observations);
        }
    }
    tracks.sort_unstable();
    stats.retained_tracks = tracks.len();
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: confidence-ordered tracks rejected_conflicts={} retained_tracks={} retained_obs={}",
            rejected_conflicts, stats.retained_tracks, stats.retained_observations
        );
    }
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

/// Build tracks by preferring correspondences that are independently
/// supported by a third view.  An edge `(i,a)-(j,b)` has one exact cycle for
/// every feature `c` in a distinct image `k` for which both `(i,a)-(k,c)` and
/// `(j,b)-(k,c)` are accepted edges.  The number of distinct supporting
/// images is the primary score; the exact number of matching third-view
/// features is the secondary score.  This prevents duplicate matches in one
/// view from masquerading as broad multi-view support.
///
/// The edge list is deduplicated and sorted before the conflict-aware
/// union-find pass.  Consequently pair/vector input order cannot affect the
/// result except for physically indistinguishable, byte-identical feature
/// rows (where the existing stable feature-index fallback is unavoidable).
/// Pair-level verified/essential support and a calibrated-E Sampson residual
/// are used only after cycle support.  Descriptor distances are not retained
/// by `PairwiseMatches`, so no synthetic descriptor score is introduced.
#[allow(clippy::type_complexity)]
pub(crate) fn build_tracks_cycle_supported(
    features: &[FeatureSet],
    camera: Option<&Camera>,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    #[derive(Clone, Copy)]
    struct Candidate {
        image_i: usize,
        image_j: usize,
        keypoint_i: usize,
        keypoint_j: usize,
        distinct_third_images: usize,
        exact_cycles: usize,
        verified_inliers: usize,
        essential_inliers: usize,
        residual: Option<f64>,
    }

    // Keep both directions so a cycle lookup never depends on the orientation
    // in which a PairwiseMatches record happened to be supplied.
    type DirectedLookup = HashMap<(usize, usize), HashMap<usize, HashSet<usize>>>;
    let mut adjacency: DirectedLookup = HashMap::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        let forward = adjacency.entry((pair.image_i, pair.image_j)).or_default();
        for &(keypoint_i, keypoint_j) in &pair.matches {
            forward.entry(keypoint_i).or_default().insert(keypoint_j);
        }
        let reverse = adjacency.entry((pair.image_j, pair.image_i)).or_default();
        for &(keypoint_i, keypoint_j) in &pair.matches {
            reverse.entry(keypoint_j).or_default().insert(keypoint_i);
        }
    }

    // Keep the strongest metadata for duplicate endpoint rows.  The cycle
    // score is global and therefore computed once below from the deduplicated
    // physical edge rather than once per duplicate PairwiseMatches record.
    let mut endpoint_metadata: HashMap<(usize, usize, usize, usize), (usize, usize, Option<f64>)> =
        HashMap::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        let essential_set = if pair.two_view_config == Some(ConfigurationType::Calibrated) {
            pair.essential_matches
                .as_ref()
                .map(|matches| matches.iter().copied().collect::<HashSet<_>>())
        } else {
            None
        };
        for &(raw_keypoint_i, raw_keypoint_j) in &pair.matches {
            let residual = if essential_set
                .as_ref()
                .is_some_and(|set| set.contains(&(raw_keypoint_i, raw_keypoint_j)))
            {
                match (camera, pair.essential_matrix.as_ref()) {
                    (Some(camera), Some(essential)) => {
                        let point_i = features
                            .get(pair.image_i)
                            .and_then(|set| set.keypoints.get(raw_keypoint_i));
                        let point_j = features
                            .get(pair.image_j)
                            .and_then(|set| set.keypoints.get(raw_keypoint_j));
                        point_i.and_then(|point_i| {
                            point_j.and_then(|point_j| {
                                normalized_sampson_residual(camera, essential, point_i, point_j)
                            })
                        })
                    }
                    _ => None,
                }
            } else {
                None
            };
            let (image_i, image_j, keypoint_i, keypoint_j) = if pair.image_i < pair.image_j {
                (pair.image_i, pair.image_j, raw_keypoint_i, raw_keypoint_j)
            } else {
                (pair.image_j, pair.image_i, raw_keypoint_j, raw_keypoint_i)
            };
            let key = (image_i, image_j, keypoint_i, keypoint_j);
            let metadata = (
                pair.matches.len(),
                pair.essential_matches.as_ref().map_or(0, Vec::len),
                residual,
            );
            endpoint_metadata
                .entry(key)
                .and_modify(|existing| {
                    let residual_order = match (existing.2, metadata.2) {
                        (Some(lhs), Some(rhs)) => rhs.total_cmp(&lhs),
                        (None, Some(_)) => Ordering::Less,
                        (Some(_), None) => Ordering::Greater,
                        (None, None) => Ordering::Equal,
                    };
                    if metadata.0 > existing.0
                        || (metadata.0 == existing.0
                            && (metadata.1 > existing.1
                                || (metadata.1 == existing.1 && residual_order == Ordering::Less)))
                    {
                        *existing = metadata;
                    }
                })
                .or_insert(metadata);
        }
    }

    let mut candidates = Vec::with_capacity(endpoint_metadata.len());
    for (
        (image_i, image_j, keypoint_i, keypoint_j),
        (verified_inliers, essential_inliers, residual),
    ) in endpoint_metadata
    {
        let (distinct_third_images, exact_cycles) = cycle_support_for_edge(
            features.len(),
            image_i,
            keypoint_i,
            image_j,
            keypoint_j,
            &adjacency,
        );
        candidates.push(Candidate {
            image_i,
            image_j,
            keypoint_i,
            keypoint_j,
            distinct_third_images,
            exact_cycles,
            verified_inliers,
            essential_inliers,
            residual,
        });
    }

    candidates.sort_unstable_by(|a, b| {
        b.distinct_third_images
            .cmp(&a.distinct_third_images)
            .then_with(|| b.exact_cycles.cmp(&a.exact_cycles))
            .then_with(|| match (a.residual, b.residual) {
                (Some(lhs), Some(rhs)) => lhs.total_cmp(&rhs),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .then_with(|| b.verified_inliers.cmp(&a.verified_inliers))
            .then_with(|| b.essential_inliers.cmp(&a.essential_inliers))
            .then_with(|| {
                stable_observation_cmp(
                    features,
                    &(a.image_i, a.keypoint_i),
                    &(b.image_i, b.keypoint_i),
                )
            })
            .then_with(|| {
                stable_observation_cmp(
                    features,
                    &(a.image_j, a.keypoint_j),
                    &(b.image_j, b.keypoint_j),
                )
            })
    });

    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images: Vec<HashSet<usize>> = Vec::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };

    for candidate in candidates {
        let a = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let b = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb
            || component_images[ra]
                .iter()
                .any(|image| component_images[rb].contains(image))
        {
            continue;
        }
        let (root, child) = if component_size[ra] > component_size[rb]
            || (component_size[ra] == component_size[rb] && ra < rb)
        {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, keypoint)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, keypoint));
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for (_root, mut observations) in groups {
        let distinct_images = observations
            .iter()
            .map(|&(image, _)| image)
            .collect::<HashSet<_>>()
            .len();
        if distinct_images >= min_track_length {
            observations.sort_by(|lhs, rhs| stable_observation_cmp(features, lhs, rhs));
            stats.retained_observations += observations.len();
            tracks.push(observations);
        }
    }
    tracks.sort_by(|lhs, rhs| stable_track_cmp(features, lhs, rhs));
    stats.retained_tracks = tracks.len();
    if sfm_debug_enabled() {
        let cycle_edges = adjacency
            .values()
            .map(|directed| directed.values().map(HashSet::len).sum::<usize>())
            .sum::<usize>();
        eprintln!(
            "sfm-debug: cycle-supported tracks edges={} directed_edges={} retained_tracks={} retained_obs={}",
            stats.input_correspondences,
            cycle_edges,
            stats.retained_tracks,
            stats.retained_observations,
        );
    }
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

pub(super) fn cycle_support_for_edge(
    n_images: usize,
    image_i: usize,
    keypoint_i: usize,
    image_j: usize,
    keypoint_j: usize,
    adjacency: &HashMap<(usize, usize), HashMap<usize, HashSet<usize>>>,
) -> (usize, usize) {
    let mut distinct_third_images = 0;
    let mut exact_cycles = 0;
    for image_k in 0..n_images {
        if image_k == image_i || image_k == image_j {
            continue;
        }
        let left = adjacency
            .get(&(image_i, image_k))
            .and_then(|map| map.get(&keypoint_i));
        let right = adjacency
            .get(&(image_j, image_k))
            .and_then(|map| map.get(&keypoint_j));
        let Some((left, right)) = left.zip(right) else {
            continue;
        };
        let exact_here = left.intersection(right).count();
        if exact_here != 0 {
            distinct_third_images += 1;
            exact_cycles += exact_here;
        }
    }
    (distinct_third_images, exact_cycles)
}

/// Compute the dimensionless normalized Sampson residual for one calibrated
/// correspondence. The essential matrix is scale-invariant, and the pixels
/// are undistorted/normalized through the camera before evaluating
/// `x_jᵀ E x_i`. Returning `None` for any non-finite or degenerate quantity is
/// deliberate: callers must not order an invalid residual ahead of a valid
/// one or compare it with a pixel-space F/H error.
pub(super) fn normalized_sampson_residual(
    camera: &Camera,
    essential: &Matrix3<f64>,
    point_i: &Point2<f64>,
    point_j: &Point2<f64>,
) -> Option<f64> {
    if !essential.iter().all(|value| value.is_finite())
        || !point_i.x.is_finite()
        || !point_i.y.is_finite()
        || !point_j.x.is_finite()
        || !point_j.y.is_finite()
    {
        return None;
    }
    let normalized_i = camera.normalize_pixel(point_i)?;
    let normalized_j = camera.normalize_pixel(point_j)?;
    if !normalized_i.x.is_finite()
        || !normalized_i.y.is_finite()
        || !normalized_j.x.is_finite()
        || !normalized_j.y.is_finite()
    {
        return None;
    }
    let bearing_i = Vector3::new(normalized_i.x, normalized_i.y, 1.0);
    let bearing_j = Vector3::new(normalized_j.x, normalized_j.y, 1.0);
    let epipolar_i = essential * bearing_i;
    let epipolar_j = essential.transpose() * bearing_j;
    let numerator = bearing_j.dot(&epipolar_i);
    let denominator = epipolar_i.x * epipolar_i.x
        + epipolar_i.y * epipolar_i.y
        + epipolar_j.x * epipolar_j.x
        + epipolar_j.y * epipolar_j.y;
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 1.0e-24 {
        return None;
    }
    let residual = numerator.abs() / denominator.sqrt();
    residual.is_finite().then_some(residual)
}

/// Build tracks with per-correspondence normalized Sampson confidence where it
/// is safe to do so. Only E-supported matches from a `Calibrated` verifier
/// result receive a residual: F-won, planar/panoramic, watermark, multiple,
/// degenerate, missing-model, and invalid entries all use the pair-level
/// fallback. This keeps model families incomparable by design while retaining
/// the historical pair-level confidence policy for all unsupported edges.
pub(crate) fn build_tracks_geometric_confidence(
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    #[derive(Clone, Copy)]
    struct Candidate {
        image_i: usize,
        image_j: usize,
        keypoint_i: usize,
        keypoint_j: usize,
        verified_inliers: usize,
        essential_inliers: usize,
        residual: Option<f64>,
    }

    let mut candidates = Vec::new();
    for pair in pairwise {
        let essential_inliers = pair.essential_matches.as_ref().map_or(0, Vec::len);
        // E/F agreement is the explicit safety gate. An essential matrix on a
        // F-won, planar, or otherwise ambiguous pair is not a comparable
        // confidence score for the winning match set.
        let essential_set = if pair.two_view_config == Some(ConfigurationType::Calibrated) {
            pair.essential_matches
                .as_ref()
                .map(|matches| matches.iter().copied().collect::<HashSet<_>>())
        } else {
            None
        };
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(raw_keypoint_i, raw_keypoint_j) in &pair.matches {
            let residual = if essential_set
                .as_ref()
                .is_some_and(|set| set.contains(&(raw_keypoint_i, raw_keypoint_j)))
            {
                pair.essential_matrix.as_ref().and_then(|essential| {
                    let point_i = features
                        .get(pair.image_i)
                        .and_then(|set| set.keypoints.get(raw_keypoint_i))?;
                    let point_j = features
                        .get(pair.image_j)
                        .and_then(|set| set.keypoints.get(raw_keypoint_j))?;
                    normalized_sampson_residual(camera, essential, point_i, point_j)
                })
            } else {
                None
            };
            let (keypoint_i, keypoint_j) = if swapped {
                (raw_keypoint_j, raw_keypoint_i)
            } else {
                (raw_keypoint_i, raw_keypoint_j)
            };
            candidates.push(Candidate {
                image_i,
                image_j,
                keypoint_i,
                keypoint_j,
                verified_inliers: pair.matches.len(),
                essential_inliers,
                residual,
            });
        }
    }
    // Finite, normalized residuals first (ascending), then the old pair-level
    // support tie-breakers and deterministic endpoint indices. Invalid/model
    // incomparable entries cannot displace a geometrically stronger edge.
    candidates.sort_unstable_by(|a, b| {
        let residual_order = match (a.residual, b.residual) {
            (Some(lhs), Some(rhs)) => lhs.total_cmp(&rhs),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        };
        residual_order
            .then_with(|| b.verified_inliers.cmp(&a.verified_inliers))
            .then_with(|| b.essential_inliers.cmp(&a.essential_inliers))
            .then_with(|| a.image_i.cmp(&b.image_i))
            .then_with(|| a.image_j.cmp(&b.image_j))
            .then_with(|| a.keypoint_i.cmp(&b.keypoint_i))
            .then_with(|| a.keypoint_j.cmp(&b.keypoint_j))
    });

    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images: Vec<HashSet<usize>> = Vec::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };
    let mut rejected_conflicts = 0usize;
    for candidate in candidates {
        let a = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let b = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            continue;
        }
        if component_images[ra]
            .iter()
            .any(|image| component_images[rb].contains(image))
        {
            rejected_conflicts += 1;
            continue;
        }
        let (root, child) = if component_size[ra] > component_size[rb]
            || (component_size[ra] == component_size[rb] && ra < rb)
        {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, keypoint)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, keypoint));
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for (_root, mut observations) in groups {
        observations.sort_unstable();
        let distinct_images = observations
            .iter()
            .map(|&(image, _)| image)
            .collect::<HashSet<_>>()
            .len();
        if distinct_images >= min_track_length {
            stats.retained_observations += observations.len();
            tracks.push(observations);
        }
    }
    tracks.sort_unstable();
    stats.retained_tracks = tracks.len();
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: geometric-confidence tracks rejected_conflicts={} retained_tracks={} retained_obs={}",
            rejected_conflicts, stats.retained_tracks, stats.retained_observations
        );
    }
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
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
pub(super) fn build_tracks_via_graph(
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

pub(super) fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

pub(super) fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}
