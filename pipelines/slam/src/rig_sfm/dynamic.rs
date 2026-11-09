//! Dynamic correspondence tracking: owner/CSR-based seed search, frame
//! activation and growth (`incremental_rig_sfm_dynamic`).

use super::*;

/// A correspondence inserted into the dynamic registration cache.  Unlike a
/// track observation, this is only a *proposal* for an unregistered image:
/// ownership is installed after the target frame's PnP report accepts the
/// entry.  Keeping the flattened target id here makes cache deduplication and
/// deterministic inlier write-back independent of pixel-coordinate equality.
#[derive(Debug, Clone, Copy)]
struct DynamicCachedRigCorrespondence {
    observation: RigObservationId,
    sensor_index: usize,
    point2d: Point2<f64>,
    source_track: usize,
}

pub(super) const DYNAMIC_EMPTY_SLOT: u32 = u32::MAX;

/// Compact flattened-observation inverse lookup.  A pair of `usize`s costs
/// 16 bytes on the supported 64-bit targets; the CSR already bounds both
/// fields by u32, so this keeps the mapper's O(observations) lookup at eight
/// bytes per entry.
#[derive(Debug, Clone, Copy)]
pub(super) struct DynamicObservation {
    image: u32,
    keypoint: u32,
}

impl DynamicObservation {
    pub(super) fn image(self) -> usize {
        usize::try_from(self.image).expect("u32 image index must fit usize")
    }

    fn keypoint(self) -> usize {
        usize::try_from(self.keypoint).expect("u32 keypoint index must fit usize")
    }
}

pub(super) fn dynamic_track_id(track_index: usize) -> Result<u32, RigSfmError> {
    let id = u32::try_from(track_index).map_err(|_| RigSfmError::DynamicIndexOverflow {
        kind: "track",
        index: track_index,
    })?;
    (id != DYNAMIC_EMPTY_SLOT)
        .then_some(id)
        .ok_or(RigSfmError::DynamicIndexOverflow {
            kind: "track",
            index: track_index,
        })
}

fn dynamic_cache_slot(slot: usize) -> Result<u32, RigSfmError> {
    let id = u32::try_from(slot).map_err(|_| RigSfmError::DynamicIndexOverflow {
        kind: "PnP cache",
        index: slot,
    })?;
    (id != DYNAMIC_EMPTY_SLOT)
        .then_some(id)
        .ok_or(RigSfmError::DynamicIndexOverflow {
            kind: "PnP cache",
            index: slot,
        })
}

fn dynamic_owner_index(owner: u32) -> Option<usize> {
    (owner != DYNAMIC_EMPTY_SLOT)
        .then(|| usize::try_from(owner).expect("u32 owner index must fit usize"))
}

#[derive(Debug, Default)]
pub(super) struct DynamicActivationUpdate {
    pub(super) activated_rows: usize,
    pub(super) activated_edges: usize,
    pub(super) creates: usize,
    pub(super) continues: usize,
    pub(super) owner_conflicts: usize,
    pub(super) same_image_conflicts: usize,
    pub(super) geometry_rejections: usize,
    frontier: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct DynamicBootstrapStats {
    legacy_tracks: usize,
    candidates: usize,
    seed_support: usize,
    seed_pairs: usize,
    seed_landmarks: usize,
    direct_stereo_fallbacks: usize,
}

#[derive(Debug)]
struct DynamicGrowth {
    frame_poses: Vec<Option<Pose>>,
    image_poses: Vec<Option<Pose>>,
    tracks: Vec<WorkingTrack>,
    track_build_stats: TrackBuildStats,
    work: RigSfmWorkStats,
    seed_frame_index: usize,
}

/// Validate the combinations for which the first dynamic correspondence-map
/// implementation has no transactional owner-state update.  Returning an
/// explicit error is important: silently running the old deferred/completion
/// behavior would make the opt-in flag appear to work while bypassing the
/// requested registration-time semantics.
fn validate_dynamic_configuration(config: &RigSfmConfig) -> Result<(), RigSfmError> {
    if config.deferred_registration_pair_prefix.is_some()
        || config.deferred_retriangulation_pair_prefix.is_some()
        || config.retriangulate_deferred_tracks_after_registration
        || config.deferred_retriangulation_metric_temporal_cycle_tracks
        || config.deferred_retriangulation_metric_temporal_quadrilateral_tracks
    {
        return Err(RigSfmError::DynamicIncompatible(
            "deferred registration/retriangulation",
        ));
    }
    if config.complete_tracks_after_registration {
        return Err(RigSfmError::DynamicIncompatible(
            "post-registration track completion",
        ));
    }
    if config.recover_metric_conflict_tracks {
        return Err(RigSfmError::DynamicIncompatible("metric conflict recovery"));
    }
    if config.track_builder != RigTrackBuilder::LegacyUnionFind {
        return Err(RigSfmError::DynamicIncompatible(
            "an alternate prebuilt track builder",
        ));
    }
    if config.robust_triangulation_pruning {
        return Err(RigSfmError::DynamicIncompatible(
            "robust triangulation pruning",
        ));
    }
    if config.direct_stereo_pnp_max_frame_gap > 0 || config.motion_bridge_max_frame_gap > 0 {
        return Err(RigSfmError::DynamicIncompatible(
            "direct or motion bridge registration",
        ));
    }
    Ok(())
}

/// Build one dense flattened-id lookup.  The CSR's observation ids are
/// contiguous by image, so this is a single O(observations) pass and avoids a
/// logarithmic image-boundary lookup in every activated edge.
pub(super) fn dynamic_observation_lookup(
    csr: &RigCorrespondenceCsr,
    features: &[FeatureSet],
) -> Result<Vec<DynamicObservation>, RigSfmError> {
    let mut observations = Vec::with_capacity(csr.total_observations());
    for (image, feature_set) in features.iter().enumerate() {
        let image = u32::try_from(image).map_err(|_| RigSfmError::DynamicIndexOverflow {
            kind: "image",
            index: image,
        })?;
        for keypoint in 0..feature_set.len() {
            let keypoint =
                u32::try_from(keypoint).map_err(|_| RigSfmError::DynamicIndexOverflow {
                    kind: "keypoint",
                    index: keypoint,
                })?;
            observations.push(DynamicObservation { image, keypoint });
        }
    }
    Ok(observations)
}

/// One bootstrap candidate.  The pair list is deliberately local to one
/// frame: the global legacy track that proposed it is not imported into the
/// dynamic owner map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DynamicSeedCandidate {
    pub(super) frame: usize,
    pub(super) support: usize,
    pub(super) pairs: Vec<(RigObservationId, RigObservationId)>,
    direct_stereo_fallback: bool,
}

/// Extract one deterministic same-frame, different-sensor pair from every
/// retained legacy track/frame combination.  A legacy track can contain more
/// than two sensors at one timestamp; enumerating every combination would
/// turn one high-degree observation into quadratic bootstrap work.  The first
/// pair in the track's stable observation order is sufficient for the metric
/// seed support count and keeps this pass O(retained observations).
pub(super) fn dynamic_membership_seed_edges(
    tracks: &[Vec<(usize, usize)>],
    csr: &RigCorrespondenceCsr,
    image_assignment: &[(usize, usize)],
    frame_count: usize,
) -> Vec<Vec<(RigObservationId, RigObservationId)>> {
    let mut by_frame = vec![Vec::new(); frame_count];
    let mut frame_marks = vec![usize::MAX; frame_count];
    let mut pair_marks = vec![usize::MAX; frame_count];
    let mut first_observation = vec![None::<(RigObservationId, usize)>; frame_count];

    for (track_index, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in track {
            let (frame, sensor) = image_assignment[image];
            if frame_marks[frame] != track_index {
                frame_marks[frame] = track_index;
                pair_marks[frame] = usize::MAX;
                first_observation[frame] = Some((
                    csr.observation_id(image, keypoint)
                        .expect("legacy track observations originate in the CSR feature universe"),
                    sensor,
                ));
                continue;
            }
            if pair_marks[frame] == track_index {
                continue;
            }
            let Some((first_id, first_sensor)) = first_observation[frame] else {
                continue;
            };
            if first_sensor == sensor {
                continue;
            }
            let observation = csr
                .observation_id(image, keypoint)
                .expect("legacy track observations originate in the CSR feature universe");
            by_frame[frame].push(if first_id <= observation {
                (first_id, observation)
            } else {
                (observation, first_id)
            });
            pair_marks[frame] = track_index;
        }
    }
    by_frame
}

/// Return deterministic, unique stereo edges for every possible seed frame.
/// Only same-frame, different-sensor CSR relations are eligible.  Rows are
/// visited once and only the canonical `left < right` direction is retained.
/// This is retained as the direct-stereo bootstrap fallback when legacy
/// membership does not provide a sufficient seed.
pub(super) fn dynamic_direct_seed_edges(
    csr: &RigCorrespondenceCsr,
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    frame_count: usize,
) -> Vec<Vec<(RigObservationId, RigObservationId)>> {
    let mut by_frame = vec![Vec::new(); frame_count];
    for left in 0..csr.total_observations() {
        let left_id =
            RigObservationId::try_from(left).expect("CSR observation id must fit RigObservationId");
        let left_image = observations[left].image();
        let (left_frame, left_sensor) = image_assignment[left_image];
        let Some(neighbors) = csr.neighbors_for(left_id) else {
            continue;
        };
        for &right_id in neighbors {
            if right_id <= left_id {
                continue;
            }
            let right = usize::try_from(right_id).expect("u32 observation id must fit usize");
            let right_image = observations[right].image();
            let (right_frame, right_sensor) = image_assignment[right_image];
            if left_frame == right_frame && left_sensor != right_sensor {
                by_frame[left_frame].push((left_id, right_id));
            }
        }
    }
    // `left` is visited in flattened-id order and every CSR row is already
    // sorted and duplicate-free.  Keeping the stream order therefore gives
    // lexicographically sorted unique pairs without an O(E log E) resort.
    by_frame
}

pub(super) fn dynamic_seed_candidates(
    seed_edges: Vec<Vec<(RigObservationId, RigObservationId)>>,
    direct_stereo_fallback: bool,
) -> Vec<DynamicSeedCandidate> {
    let mut candidates = seed_edges
        .into_iter()
        .enumerate()
        // Keep the same support floor as `metric_seed_candidates` for legacy
        // membership.  The direct fallback retains every non-empty frame so
        // a caller still gets the historical `InsufficientSeedStructure`
        // error (rather than a new `NoMetricSeed`) when only a small stereo
        // support exists.
        .filter(|(_, pairs)| !pairs.is_empty() && (direct_stereo_fallback || pairs.len() >= 6))
        .map(|(frame, pairs)| DynamicSeedCandidate {
            frame,
            support: pairs.len(),
            pairs,
            direct_stereo_fallback,
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|left, right| {
        right
            .support
            .cmp(&left.support)
            .then_with(|| left.frame.cmp(&right.frame))
    });
    candidates
}

#[allow(clippy::too_many_arguments)]
fn dynamic_select_seed_candidate(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    candidates: &[DynamicSeedCandidate],
    image_poses: &mut [Option<Pose>],
) -> Option<(usize, usize)> {
    let required_landmarks = config.min_pnp_inliers.max(6);
    let mut selected = None::<(usize, usize)>;
    for (rank, candidate) in candidates.iter().enumerate() {
        let landmarks = dynamic_seed_landmark_count(
            rig,
            frames,
            features,
            observations,
            image_assignment,
            config,
            candidate.frame,
            &candidate.pairs,
            image_poses,
        );
        if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
            let source = if candidate.direct_stereo_fallback {
                "direct-stereo"
            } else {
                "legacy-membership"
            };
            eprintln!(
                "rig-sfm-debug: dynamic-seed-candidate rank={rank} source={source} frame={} support={} seed_pairs={} triangulated={landmarks}",
                candidate.frame,
                candidate.support,
                candidate.pairs.len(),
            );
        }
        // Preserve the historical mapper's seed policy: candidates are
        // already ordered by metric support descending and frame index
        // ascending, so the first geometrically viable candidate wins.
        // Besides matching the frozen control, this avoids triangulating
        // every later frame once a valid seed has been found.
        if landmarks >= required_landmarks {
            return Some((rank, landmarks));
        }
        let better = selected.is_none_or(|(selected_index, selected_landmarks)| {
            landmarks > selected_landmarks
                || (landmarks == selected_landmarks
                    && (candidate.support > candidates[selected_index].support
                        || (candidate.support == candidates[selected_index].support
                            && candidate.frame < candidates[selected_index].frame)))
        });
        if better {
            selected = Some((rank, landmarks));
        }
    }
    selected
}

fn dynamic_track_build_stats(
    pairwise: &[PairwiseMatches],
    tracks: &[WorkingTrack],
) -> TrackBuildStats {
    let observations = tracks.iter().map(|track| track.observations.len()).sum();
    TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        // Dynamic mode has no transitive connected-component phase.  These
        // fields describe the seed owner set that actually entered the mapper;
        // they must not pretend that all CSR edges were pre-unioned.
        connected_components: tracks.len(),
        retained_tracks: tracks.len(),
        retained_observations: observations,
        ..TrackBuildStats::default()
    }
}

/// Create the metric seed owner state from same-frame stereo CSR edges.  Seed
/// tracks are intentionally exactly two observations; later registration is
/// what extends them.  Edges are already canonicalized by the CSR and are
/// processed in flattened-id order, giving stable track indices.
#[allow(clippy::too_many_arguments)]
pub(super) fn dynamic_create_seed_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    seed_frame: usize,
    seed_edges: &[(RigObservationId, RigObservationId)],
    owner: &mut [u32],
    image_tracks: &mut [Vec<(usize, usize)>],
    tracks: &mut Vec<WorkingTrack>,
) -> Result<TriangulationUpdate, RigSfmError> {
    let mut frontier = HashSet::new();
    for &(left_id, right_id) in seed_edges {
        let left = usize::try_from(left_id).expect("u32 observation id must fit usize");
        let right = usize::try_from(right_id).expect("u32 observation id must fit usize");
        if owner[left] != DYNAMIC_EMPTY_SLOT || owner[right] != DYNAMIC_EMPTY_SLOT {
            continue;
        }
        let left_observation = observations[left];
        let right_observation = observations[right];
        let left_image = left_observation.image();
        let left_keypoint = left_observation.keypoint();
        let right_image = right_observation.image();
        let right_keypoint = right_observation.keypoint();
        let left_assignment = image_assignment[left_image];
        let right_assignment = image_assignment[right_image];
        if left_assignment.0 != seed_frame
            || right_assignment.0 != seed_frame
            || left_assignment.1 == right_assignment.1
        {
            continue;
        }
        let track_index = tracks.len();
        let owner_id = dynamic_track_id(track_index)?;
        tracks.push(WorkingTrack {
            observations: vec![(left_image, left_keypoint), (right_image, right_keypoint)],
            position: None,
            metric_anchored: true,
        });
        owner[left] = owner_id;
        owner[right] = owner_id;
        insert_dynamic_image_track(&mut image_tracks[left_image], left_keypoint, track_index);
        insert_dynamic_image_track(&mut image_tracks[right_image], right_keypoint, track_index);
        frontier.insert(track_index);
    }
    Ok(triangulate_frontier(
        rig,
        features,
        image_assignment,
        image_poses,
        config,
        tracks,
        frontier,
    ))
}

#[allow(clippy::too_many_arguments)]
fn dynamic_seed_landmark_count(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    seed_frame: usize,
    seed_edges: &[(RigObservationId, RigObservationId)],
    image_poses: &mut [Option<Pose>],
) -> usize {
    install_image_poses(rig, &frames[seed_frame], &Pose::identity(), image_poses);
    let mut tracks = Vec::new();
    let mut used = HashSet::new();
    let mut frontier = HashSet::new();
    for &(left_id, right_id) in seed_edges {
        if used.contains(&left_id) || used.contains(&right_id) {
            continue;
        }
        used.insert(left_id);
        used.insert(right_id);
        let left_observation =
            observations[usize::try_from(left_id).expect("u32 observation id must fit usize")];
        let right_observation =
            observations[usize::try_from(right_id).expect("u32 observation id must fit usize")];
        let left_image = left_observation.image();
        let left_keypoint = left_observation.keypoint();
        let right_image = right_observation.image();
        let right_keypoint = right_observation.keypoint();
        let track_index = tracks.len();
        tracks.push(WorkingTrack {
            observations: vec![(left_image, left_keypoint), (right_image, right_keypoint)],
            position: None,
            metric_anchored: true,
        });
        frontier.insert(track_index);
    }
    let triangulation = triangulate_frontier(
        rig,
        features,
        image_assignment,
        image_poses,
        config,
        &mut tracks,
        frontier,
    );
    for image in &frames[seed_frame].images {
        image_poses[image.image_index] = None;
    }
    triangulation.landmarks.len()
}

fn insert_dynamic_image_track(index: &mut Vec<(usize, usize)>, keypoint: usize, track: usize) {
    match index.binary_search_by_key(&keypoint, |&(candidate, _)| candidate) {
        Ok(position) if index[position].1 == track => {}
        Ok(position) => {
            // One feature observation has one dynamic owner.  A caller that
            // reaches this branch has already checked the dense owner slot;
            // retaining the lower index makes this helper robust to a stale
            // image index without introducing duplicate PnP rows.
            index[position].1 = index[position].1.min(track);
        }
        Err(position) => index.insert(position, (keypoint, track)),
    }
}

fn dynamic_image_registered(
    observation: RigObservationId,
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
) -> bool {
    let image = observations
        [usize::try_from(observation).expect("u32 observation id must fit usize")]
    .image();
    frame_poses[image_assignment[image].0].is_some()
}

/// Attach an owned observation after a successful PnP.  This is deliberately
/// separate from graph-cache insertion: cache entries for unregistered
/// targets are proposals and must not claim owner slots before PnP accepts
/// them.
#[allow(clippy::too_many_arguments)]
fn dynamic_attach_observation(
    observation: RigObservationId,
    track_index: usize,
    observations: &[DynamicObservation],
    owner: &mut [u32],
    image_tracks: &mut [Vec<(usize, usize)>],
    tracks: &mut [WorkingTrack],
    update: &mut DynamicActivationUpdate,
) -> Result<bool, RigSfmError> {
    let observation_index =
        usize::try_from(observation).expect("u32 observation id must fit usize");
    let image_observation = observations[observation_index];
    let image = image_observation.image();
    let keypoint = image_observation.keypoint();
    let track_id = dynamic_track_id(track_index)?;
    match dynamic_owner_index(owner[observation_index]) {
        Some(previous) if previous == track_index => return Ok(false),
        Some(_) => {
            update.owner_conflicts += 1;
            return Ok(false);
        }
        None => {}
    }
    if tracks[track_index]
        .observations
        .iter()
        .any(|&(track_image, _)| track_image == image)
    {
        update.same_image_conflicts += 1;
        return Ok(false);
    }
    tracks[track_index].observations.push((image, keypoint));
    tracks[track_index].observations.sort_unstable();
    owner[observation_index] = track_id;
    insert_dynamic_image_track(&mut image_tracks[image], keypoint, track_index);
    update.continues += 1;
    update.frontier.push(track_index);
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn dynamic_continuation_geometry_ok(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    track_index: usize,
    target: RigObservationId,
    observations: &[DynamicObservation],
    tracks: &[WorkingTrack],
) -> bool {
    let Some(position) = tracks[track_index].position else {
        // An unpositioned owner can still accumulate registered observations;
        // the frontier triangulation gate will decide when it becomes a
        // landmark.
        return true;
    };
    let observation =
        observations[usize::try_from(target).expect("u32 observation id must fit usize")];
    let image = observation.image();
    let keypoint = observation.keypoint();
    let Some(pose) = image_poses[image].as_ref() else {
        return false;
    };
    let sensor = &rig.sensors()[image_assignment[image].1];
    let point_camera = pose.transform_world_point(&position);
    if point_camera.z <= 0.0 {
        return false;
    }
    sensor
        .camera
        .project(&point_camera)
        .is_some_and(|projected| {
            let error = (projected - features[image].keypoints[keypoint]).norm();
            error.is_finite() && error <= config.max_reprojection_error_px
        })
}

/// Apply all CSR edges incident to a newly registered frame.  Each flattened
/// row is activated once for the lifetime of a growth run; rows whose other
/// image is still unregistered are intentionally left ownership-neutral and
/// are revisited from that image's frame later.  This is the central
/// registration-time create/continue state transition.
#[allow(clippy::too_many_arguments)]
pub(super) fn dynamic_activate_frame(
    frame: usize,
    frames: &[RigFrame],
    csr: &RigCorrespondenceCsr,
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    config: &RigSfmConfig,
    owner: &mut [u32],
    image_tracks: &mut [Vec<(usize, usize)>],
    tracks: &mut Vec<WorkingTrack>,
    activated_rows: &mut [bool],
    update: &mut DynamicActivationUpdate,
) -> Result<(), RigSfmError> {
    // Images are sorted once (the rig has a fixed, usually tiny sensor count),
    // then their contiguous CSR ranges are streamed directly.  This avoids
    // materializing and sorting every row of a large frame.
    let mut images = frames[frame].images.iter().collect::<Vec<_>>();
    images.sort_unstable_by_key(|image| image.image_index);
    for image in images {
        let start = usize::try_from(csr.image_offsets[image.image_index])
            .expect("CSR image offset must fit usize");
        let end = usize::try_from(csr.image_offsets[image.image_index + 1])
            .expect("CSR image offset must fit usize");
        for row in start..end {
            if activated_rows[row] {
                continue;
            }
            activated_rows[row] = true;
            update.activated_rows += 1;
            let left_id = RigObservationId::try_from(row)
                .expect("CSR observation row must fit RigObservationId");
            let left_observation = observations[row];
            let left_image = left_observation.image();
            let left_keypoint = left_observation.keypoint();
            let Some(neighbors) = csr.neighbors_for(left_id) else {
                continue;
            };
            // CSR rows are sorted; do not reorder this loop.  It is also the
            // deterministic tie-break when several edges touch free endpoints.
            for &right_id in neighbors {
                update.activated_edges += 1;
                let right = usize::try_from(right_id).expect("u32 observation id must fit usize");
                let right_observation = observations[right];
                let right_image = right_observation.image();
                let right_keypoint = right_observation.keypoint();
                if !dynamic_image_registered(left_id, observations, image_assignment, frame_poses)
                    || !dynamic_image_registered(
                        right_id,
                        observations,
                        image_assignment,
                        frame_poses,
                    )
                {
                    continue;
                }
                let left_owner = dynamic_owner_index(owner[row]);
                let right_owner = dynamic_owner_index(owner[right]);
                match (left_owner, right_owner) {
                    (Some(left_track), Some(right_track)) if left_track == right_track => {}
                    (Some(_), Some(_)) => update.owner_conflicts += 1,
                    (Some(track_index), None) => {
                        if !dynamic_continuation_geometry_ok(
                            rig,
                            features,
                            image_assignment,
                            image_poses,
                            config,
                            track_index,
                            right_id,
                            observations,
                            tracks,
                        ) {
                            update.geometry_rejections += 1;
                            continue;
                        }
                        let _ = dynamic_attach_observation(
                            right_id,
                            track_index,
                            observations,
                            owner,
                            image_tracks,
                            tracks,
                            update,
                        )?;
                    }
                    (None, Some(track_index)) => {
                        if !dynamic_continuation_geometry_ok(
                            rig,
                            features,
                            image_assignment,
                            image_poses,
                            config,
                            track_index,
                            left_id,
                            observations,
                            tracks,
                        ) {
                            update.geometry_rejections += 1;
                            continue;
                        }
                        let _ = dynamic_attach_observation(
                            left_id,
                            track_index,
                            observations,
                            owner,
                            image_tracks,
                            tracks,
                            update,
                        )?;
                    }
                    (None, None) => {
                        if left_image == right_image {
                            update.same_image_conflicts += 1;
                            continue;
                        }
                        let track_index = tracks.len();
                        let owner_id = dynamic_track_id(track_index)?;
                        let mut track_observations =
                            vec![(left_image, left_keypoint), (right_image, right_keypoint)];
                        track_observations.sort_unstable();
                        tracks.push(WorkingTrack {
                            observations: track_observations,
                            position: None,
                            metric_anchored: {
                                let (left_frame, left_sensor) = image_assignment[left_image];
                                let (right_frame, right_sensor) = image_assignment[right_image];
                                left_frame == right_frame && left_sensor != right_sensor
                            },
                        });
                        owner[row] = owner_id;
                        owner[right] = owner_id;
                        insert_dynamic_image_track(
                            &mut image_tracks[left_image],
                            left_keypoint,
                            track_index,
                        );
                        insert_dynamic_image_track(
                            &mut image_tracks[right_image],
                            right_keypoint,
                            track_index,
                        );
                        update.creates += 1;
                        update.frontier.push(track_index);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Insert graph neighbours of positioned tracks into the existing heap-driven
/// frame cache.  `cache_slots` is dense in flattened observation id, so a
/// target feature receives at most one PnP row even when several positioned
/// owners point at it.  The lower track index wins ties, which is stable under
/// CSR row and track ordering.
#[allow(clippy::too_many_arguments)]
fn dynamic_expose_positioned_tracks(
    csr: &RigCorrespondenceCsr,
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    tracks: &[WorkingTrack],
    track_indices: &[usize],
    owner: &[u32],
    features: &[FeatureSet],
    exposed_rows: &mut [bool],
    cache_slots: &mut [u32],
    frame_correspondences: &mut [Vec<DynamicCachedRigCorrespondence>],
    cache_frames_by_track: &mut Vec<HashMap<usize, usize>>,
    frame_versions: &mut [usize],
    candidate_heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> Result<usize, RigSfmError> {
    if cache_frames_by_track.len() < tracks.len() {
        cache_frames_by_track.resize_with(tracks.len(), HashMap::new);
    }
    let mut changed_frames = BTreeSet::new();
    let mut insertions = 0usize;
    let mut track_indices = track_indices.to_vec();
    track_indices.sort_unstable();
    track_indices.dedup();
    for track_index in track_indices {
        if tracks[track_index].position.is_none() {
            continue;
        }
        let track = &tracks[track_index];
        for &(source_image, source_keypoint) in &track.observations {
            let source_id = csr
                .observation_id(source_image, source_keypoint)
                .expect("dynamic track observations originate in CSR features");
            let source_row = usize::try_from(source_id).expect("u32 observation id must fit usize");
            if exposed_rows[source_row] {
                continue;
            }
            exposed_rows[source_row] = true;
            let Some(neighbors) = csr.neighbors_for(source_id) else {
                continue;
            };
            for &target_id in neighbors {
                let target_index =
                    usize::try_from(target_id).expect("u32 observation id must fit usize");
                let target_observation = observations[target_index];
                let target_image = target_observation.image();
                let target_keypoint = target_observation.keypoint();
                let target_frame = image_assignment[target_image].0;
                if frame_poses[target_frame].is_some() || owner[target_index] != DYNAMIC_EMPTY_SLOT
                {
                    continue;
                }
                let entry = DynamicCachedRigCorrespondence {
                    observation: target_id,
                    sensor_index: image_assignment[target_image].1,
                    point2d: features[target_image].keypoints[target_keypoint],
                    source_track: track_index,
                };
                match cache_slots[target_index] {
                    DYNAMIC_EMPTY_SLOT => {
                        let slot = dynamic_cache_slot(frame_correspondences[target_frame].len())?;
                        cache_slots[target_index] = slot;
                        frame_correspondences[target_frame].push(entry);
                        *cache_frames_by_track[track_index]
                            .entry(target_frame)
                            .or_default() += 1;
                        insertions += 1;
                        changed_frames.insert(target_frame);
                    }
                    slot => {
                        let slot = usize::try_from(slot).expect("u32 cache slot must fit usize");
                        let cached = &mut frame_correspondences[target_frame][slot];
                        if track_index < cached.source_track {
                            let old_source = cached.source_track;
                            cached.source_track = track_index;
                            cached.sensor_index = entry.sensor_index;
                            cached.point2d = entry.point2d;
                            if let Some(count) =
                                cache_frames_by_track[old_source].get_mut(&target_frame)
                            {
                                if *count <= 1 {
                                    cache_frames_by_track[old_source].remove(&target_frame);
                                } else {
                                    *count -= 1;
                                }
                            }
                            *cache_frames_by_track[track_index]
                                .entry(target_frame)
                                .or_default() += 1;
                            changed_frames.insert(target_frame);
                        }
                    }
                }
            }
        }
    }
    for frame in changed_frames {
        frame_versions[frame] += 1;
        candidate_heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    Ok(insertions)
}

fn dynamic_requeue_cached_frames(
    track_indices: &[usize],
    frame_poses: &[Option<Pose>],
    frame_correspondences: &[Vec<DynamicCachedRigCorrespondence>],
    cache_frames_by_track: &[HashMap<usize, usize>],
    frame_versions: &mut [usize],
    candidate_heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> usize {
    let mut frames = BTreeSet::new();
    for &track_index in track_indices {
        if let Some(track_frames) = cache_frames_by_track.get(track_index) {
            for &frame in track_frames.keys() {
                if frame_poses[frame].is_none() {
                    frames.insert(frame);
                }
            }
        }
    }
    for &frame in &frames {
        frame_versions[frame] += 1;
        candidate_heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    frames.len()
}

/// Run one dynamic growth schedule from a selected metric seed.  The owner
/// and cache arrays are rebuilt per seed trial; only the selected trial is
/// returned to the caller, and all CSR rows are activated at most once in that
/// returned growth.
#[allow(clippy::too_many_arguments)]
fn dynamic_grow_from_seed(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &RigSfmConfig,
    csr: &RigCorrespondenceCsr,
    observations: &[DynamicObservation],
    image_assignment: &[(usize, usize)],
    seed_frame: usize,
    seed_edges: &[(RigObservationId, RigObservationId)],
    bootstrap: DynamicBootstrapStats,
) -> Result<Option<DynamicGrowth>, RigSfmError> {
    let trace_registration = std::env::var_os("VISLOC_SFM_TRACE_REGISTRATION").is_some();
    let mut frame_poses = vec![None; frames.len()];
    frame_poses[seed_frame] = Some(Pose::identity());
    let mut image_poses = vec![None; features.len()];
    install_image_poses(
        rig,
        &frames[seed_frame],
        frame_poses[seed_frame]
            .as_ref()
            .expect("seed pose installed"),
        &mut image_poses,
    );
    let mut owner = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    let mut image_tracks = vec![Vec::new(); features.len()];
    let mut tracks = Vec::new();
    let seed_triangulation = dynamic_create_seed_tracks(
        rig,
        features,
        observations,
        image_assignment,
        &image_poses,
        config,
        seed_frame,
        seed_edges,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
    )?;
    let required_seed_landmarks = config.min_pnp_inliers.max(6);
    if seed_triangulation.landmarks.len() < required_seed_landmarks {
        return Ok(None);
    }
    emit_registration_trace(
        trace_registration,
        RegistrationTrace {
            mode: "dynamic",
            order: 0,
            frame: seed_frame,
            support: seed_triangulation.landmarks.len(),
            inliers: seed_triangulation.landmarks.len(),
            sensors: rig.sensors().len(),
        },
        frame_poses[seed_frame]
            .as_ref()
            .expect("seed pose installed"),
    );

    let mut work = RigSfmWorkStats {
        triangulation_attempts: seed_triangulation.attempts,
        robust_triangulation_tracks: seed_triangulation.robust_tracks,
        robust_triangulation_pruned_observations: seed_triangulation.pruned_observations,
        robust_triangulation_majority_rejections: seed_triangulation.majority_rejections,
        dynamic_observation_lookup_entries: observations.len(),
        dynamic_track_creates: tracks.len(),
        dynamic_bootstrap_legacy_tracks: bootstrap.legacy_tracks,
        dynamic_bootstrap_candidates: bootstrap.candidates,
        dynamic_bootstrap_seed_support: bootstrap.seed_support,
        dynamic_bootstrap_seed_pairs: bootstrap.seed_pairs,
        dynamic_bootstrap_seed_landmarks: bootstrap.seed_landmarks,
        dynamic_bootstrap_direct_fallbacks: bootstrap.direct_stereo_fallbacks,
        ..RigSfmWorkStats::default()
    };
    let mut activated_rows = vec![false; csr.total_observations()];
    let mut exposed_rows = vec![false; csr.total_observations()];
    let mut cache_slots = vec![DYNAMIC_EMPTY_SLOT; csr.total_observations()];
    let mut frame_correspondences: Vec<Vec<DynamicCachedRigCorrespondence>> =
        vec![Vec::new(); frames.len()];
    let mut cache_frames_by_track = vec![HashMap::new(); tracks.len()];
    let mut frame_versions = vec![0usize; frames.len()];
    let mut attempted_versions = vec![None; frames.len()];
    let mut candidate_heap = BinaryHeap::new();
    let mut seed_activation = DynamicActivationUpdate::default();
    dynamic_activate_frame(
        seed_frame,
        frames,
        csr,
        observations,
        image_assignment,
        &frame_poses,
        &image_poses,
        rig,
        features,
        config,
        &mut owner,
        &mut image_tracks,
        &mut tracks,
        &mut activated_rows,
        &mut seed_activation,
    )?;
    work.dynamic_activated_rows += seed_activation.activated_rows;
    work.dynamic_activated_edges += seed_activation.activated_edges;
    work.dynamic_track_creates += seed_activation.creates;
    work.dynamic_track_continues += seed_activation.continues;
    work.dynamic_owner_conflicts += seed_activation.owner_conflicts;
    work.dynamic_same_image_conflicts += seed_activation.same_image_conflicts;
    work.dynamic_geometry_rejections += seed_activation.geometry_rejections;
    let seed_insertions = dynamic_expose_positioned_tracks(
        csr,
        observations,
        image_assignment,
        &frame_poses,
        &tracks,
        &(0..tracks.len()).collect::<Vec<_>>(),
        &owner,
        features,
        &mut exposed_rows,
        &mut cache_slots,
        &mut frame_correspondences,
        &mut cache_frames_by_track,
        &mut frame_versions,
        &mut candidate_heap,
    )?;
    work.dynamic_pnp_graph_insertions += seed_insertions;
    work.correspondence_cache_insertions += seed_insertions;

    let pnp = GeneralizedPnPRansac {
        iterations: config.pnp_max_iterations,
        reprojection_threshold: config.max_reprojection_error_px,
        seed: config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    };
    let mut registration_order = vec![seed_frame];
    while let Some((support, Reverse(frame), version)) = candidate_heap.pop() {
        if frame_poses[frame].is_some()
            || frame_versions[frame] != version
            || frame_correspondences[frame].len() != support
            || attempted_versions[frame] == Some(version)
            || support < config.min_pnp_inliers.max(6)
        {
            continue;
        }
        attempted_versions[frame] = Some(version);
        work.pnp_attempts += 1;
        let entries = frame_correspondences[frame].clone();
        let mut correspondences = Vec::with_capacity(entries.len());
        let mut entry_indices = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let Some(point3d) = tracks[entry.source_track].position else {
                continue;
            };
            correspondences.push(GeneralizedCorrespondence2D3D {
                sensor_index: entry.sensor_index,
                point2d: entry.point2d,
                point3d,
                confidence: None,
            });
            entry_indices.push(index);
        }
        let distinct_sensors = correspondences
            .iter()
            .map(|entry| entry.sensor_index)
            .collect::<HashSet<_>>()
            .len();
        if distinct_sensors < config.min_pnp_sensors {
            work.pnp_insufficient_sensor_attempts += 1;
            continue;
        }
        let Some(report) = pnp.estimate(rig, &correspondences) else {
            work.pnp_estimation_failures += 1;
            continue;
        };
        if report.inliers.len() < config.min_pnp_inliers.max(6) {
            work.pnp_inlier_rejections += 1;
            continue;
        }
        emit_registration_trace(
            trace_registration,
            RegistrationTrace {
                mode: "dynamic",
                order: registration_order.len(),
                frame,
                support: correspondences.len(),
                inliers: report.inliers.len(),
                sensors: distinct_sensors,
            },
            &report.pose,
        );
        frame_poses[frame] = Some(report.pose);
        install_image_poses(
            rig,
            &frames[frame],
            frame_poses[frame]
                .as_ref()
                .expect("registered pose installed"),
            &mut image_poses,
        );
        registration_order.push(frame);
        work.pnp_registrations += 1;

        // Use the report's original correspondence indices before activating
        // all incident rows.  The cache is deduplicated by target observation,
        // so each accepted 2D location can claim at most one owner here.
        let mut inliers = report.inliers;
        inliers.sort_unstable_by_key(|&index| entries[entry_indices[index]].observation);
        let mut frontier = Vec::new();
        let mut attachment = DynamicActivationUpdate::default();
        for correspondence_index in inliers {
            let Some(&entry_index) = entry_indices.get(correspondence_index) else {
                continue;
            };
            let entry = entries[entry_index];
            let attached = dynamic_attach_observation(
                entry.observation,
                entry.source_track,
                observations,
                &mut owner,
                &mut image_tracks,
                &mut tracks,
                &mut attachment,
            )?;
            if attached {
                frontier.push(entry.source_track);
            }
        }
        let mut activation = DynamicActivationUpdate::default();
        dynamic_activate_frame(
            frame,
            frames,
            csr,
            observations,
            image_assignment,
            &frame_poses,
            &image_poses,
            rig,
            features,
            config,
            &mut owner,
            &mut image_tracks,
            &mut tracks,
            &mut activated_rows,
            &mut activation,
        )?;
        work.dynamic_activated_rows += activation.activated_rows;
        work.dynamic_activated_edges += activation.activated_edges;
        work.dynamic_track_creates += activation.creates;
        work.dynamic_track_continues += attachment.continues + activation.continues;
        work.dynamic_owner_conflicts += attachment.owner_conflicts + activation.owner_conflicts;
        work.dynamic_same_image_conflicts +=
            attachment.same_image_conflicts + activation.same_image_conflicts;
        work.dynamic_geometry_rejections += activation.geometry_rejections;
        frontier.extend(activation.frontier);
        frontier.sort_unstable();
        frontier.dedup();
        let frontier_set = frontier.iter().copied().collect::<HashSet<_>>();
        let triangulation = triangulate_frontier(
            rig,
            features,
            image_assignment,
            &image_poses,
            config,
            &mut tracks,
            frontier_set,
        );
        work.triangulation_attempts += triangulation.attempts;
        work.robust_triangulation_tracks += triangulation.robust_tracks;
        work.robust_triangulation_pruned_observations += triangulation.pruned_observations;
        work.robust_triangulation_majority_rejections += triangulation.majority_rejections;
        let insertions = dynamic_expose_positioned_tracks(
            csr,
            observations,
            image_assignment,
            &frame_poses,
            &tracks,
            &frontier,
            &owner,
            features,
            &mut exposed_rows,
            &mut cache_slots,
            &mut frame_correspondences,
            &mut cache_frames_by_track,
            &mut frame_versions,
            &mut candidate_heap,
        )?;
        work.dynamic_pnp_graph_insertions += insertions;
        work.correspondence_cache_insertions += insertions;

        if config.local_ba_every > 0
            && config.local_ba_window_size >= 2
            && registration_order
                .len()
                .is_multiple_of(config.local_ba_every)
        {
            let (active_frames, anchor) = select_local_ba_frames(
                rig,
                frames,
                features,
                image_assignment,
                &image_tracks,
                &image_poses,
                &tracks,
                &registration_order,
                config,
            );
            let local_ba_config = BaConfig {
                max_iterations: config.local_ba_iterations,
                ..config.ba_config
            };
            if !active_frames.is_empty()
                && run_rig_bundle_adjustment(
                    rig,
                    features,
                    image_assignment,
                    config,
                    &active_frames,
                    anchor,
                    &local_ba_config,
                    0,
                    &[],
                    false,
                    &mut frame_poses,
                    &mut image_poses,
                    &mut tracks,
                )?
                .is_some()
            {
                work.local_ba_runs += 1;
                let affected_tracks = active_frames
                    .iter()
                    .flat_map(|frame| frames[*frame].images.iter())
                    .flat_map(|image| image_tracks[image.image_index].iter())
                    .map(|&(_, track)| track)
                    .collect::<HashSet<_>>();
                if config.ba_metric_tracks_only {
                    work.ba_retriangulated_tracks += retriangulate_unanchored_tracks(
                        rig,
                        features,
                        image_assignment,
                        &image_poses,
                        config,
                        &mut tracks,
                        &affected_tracks,
                    );
                }
                let affected_tracks = affected_tracks.into_iter().collect::<Vec<_>>();
                work.ba_requeued_frames += dynamic_requeue_cached_frames(
                    &affected_tracks,
                    &frame_poses,
                    &frame_correspondences,
                    &cache_frames_by_track,
                    &mut frame_versions,
                    &mut candidate_heap,
                );
                let insertions = dynamic_expose_positioned_tracks(
                    csr,
                    observations,
                    image_assignment,
                    &frame_poses,
                    &tracks,
                    &affected_tracks,
                    &owner,
                    features,
                    &mut exposed_rows,
                    &mut cache_slots,
                    &mut frame_correspondences,
                    &mut cache_frames_by_track,
                    &mut frame_versions,
                    &mut candidate_heap,
                )?;
                work.dynamic_pnp_graph_insertions += insertions;
                work.correspondence_cache_insertions += insertions;
            }
        }
    }
    let track_build_stats = dynamic_track_build_stats(pairwise, &tracks);
    Ok(Some(DynamicGrowth {
        frame_poses,
        image_poses,
        tracks,
        track_build_stats,
        work,
        seed_frame_index: seed_frame,
    }))
}

/// Full dynamic entry point, including the existing final rig refinement and
/// output assembly.  Growth itself is owner/CSR based; final BA may still
/// refine the resulting metric tracks exactly as it does for the historical
/// path.
pub(super) fn incremental_rig_sfm_dynamic(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &RigSfmConfig,
) -> Result<RigSfmResult, RigSfmError> {
    validate_dynamic_configuration(config)?;
    let image_assignment = image_assignment(frames, features.len());
    let csr = build_rig_correspondence_csr_from_features(features, pairwise)
        .map_err(|error| RigSfmError::DynamicCorrespondenceBuild(error.to_string()))?;
    let observations = dynamic_observation_lookup(&csr, features)?;
    // Bootstrap membership is intentionally the only legacy topology that
    // crosses into dynamic mode.  The selected track contributes just one
    // same-frame stereo pair; all later ownership and growth still consume
    // the CSR at registration time.
    let mut legacy_config = *config;
    legacy_config.track_builder = RigTrackBuilder::LegacyUnionFind;
    let legacy_tracks =
        build_rig_track_output(features, pairwise, &image_assignment, &legacy_config).tracks;
    let legacy_seed_edges =
        dynamic_membership_seed_edges(&legacy_tracks, &csr, &image_assignment, frames.len());
    let mut legacy_candidates = dynamic_seed_candidates(legacy_seed_edges, false);
    if let Some(frame) = config.seed_frame {
        legacy_candidates.retain(|candidate| candidate.frame == frame);
    }
    let required_seed_landmarks = config.min_pnp_inliers.max(6);
    let mut seed_image_poses = vec![None; features.len()];

    // Evaluate legacy-membership candidates in the historical mapper's
    // support-desc/frame-asc order and accept the first metric-valid seed.
    // Each track/frame contributes at most one pair, and growth is run once.
    let legacy_selection = dynamic_select_seed_candidate(
        rig,
        frames,
        features,
        &observations,
        &image_assignment,
        config,
        &legacy_candidates,
        &mut seed_image_poses,
    );
    let mut selected_candidates = &legacy_candidates;
    let mut selected = legacy_selection;
    let mut candidate_evaluations = legacy_candidates.len();

    // Keep the previous direct same-frame CSR bootstrap as a bounded
    // fallback.  It is only materialized/scored when legacy membership has no
    // sufficient metric result, so normal runs do not pay for a second seed
    // candidate set and direct edges never change the legacy support order.
    let direct_candidates = selected
        .is_none_or(|(_, landmarks)| landmarks < required_seed_landmarks)
        .then(|| {
            let direct_seed_edges =
                dynamic_direct_seed_edges(&csr, &observations, &image_assignment, frames.len());
            let mut candidates = dynamic_seed_candidates(direct_seed_edges, true);
            if let Some(frame) = config.seed_frame {
                candidates.retain(|candidate| candidate.frame == frame);
            }
            candidates
        });
    if let Some(fallback_candidates) = direct_candidates.as_ref() {
        candidate_evaluations += fallback_candidates.len();
        let direct_selection = dynamic_select_seed_candidate(
            rig,
            frames,
            features,
            &observations,
            &image_assignment,
            config,
            fallback_candidates,
            &mut seed_image_poses,
        );
        if direct_selection.is_some_and(|(_, landmarks)| landmarks >= required_seed_landmarks)
            || selected.is_none()
        {
            selected_candidates = fallback_candidates;
            selected = direct_selection;
        } else if let Some((direct_index, direct_landmarks)) = direct_selection {
            let (legacy_index, legacy_landmarks) = selected.expect("legacy selection exists");
            let direct_candidate = &fallback_candidates[direct_index];
            let legacy_candidate = &legacy_candidates[legacy_index];
            let direct_better = direct_landmarks > legacy_landmarks
                || (direct_landmarks == legacy_landmarks
                    && (direct_candidate.support > legacy_candidate.support
                        || (direct_candidate.support == legacy_candidate.support
                            && direct_candidate.frame < legacy_candidate.frame)));
            if direct_better {
                selected_candidates = fallback_candidates;
                selected = Some((direct_index, direct_landmarks));
            }
        }
    }

    let Some((selected_index, selected_landmarks)) = selected else {
        return Err(RigSfmError::NoMetricSeed);
    };
    let selected_candidate = selected_candidates[selected_index].clone();
    let best_seed = selected_candidate.frame;
    if selected_landmarks < required_seed_landmarks {
        return Err(RigSfmError::InsufficientSeedStructure {
            frame: best_seed,
            required: required_seed_landmarks,
            triangulated: selected_landmarks,
        });
    }
    let bootstrap = DynamicBootstrapStats {
        legacy_tracks: legacy_tracks.len(),
        candidates: candidate_evaluations,
        seed_support: selected_candidate.support,
        seed_pairs: selected_candidate.pairs.len(),
        seed_landmarks: selected_landmarks,
        direct_stereo_fallbacks: usize::from(selected_candidate.direct_stereo_fallback),
    };
    if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
        eprintln!(
            "rig-sfm-debug: dynamic-seed-selected source={} frame={best_seed} support={} seed_pairs={} triangulated={selected_landmarks} required={required_seed_landmarks} legacy_tracks={} candidates={} direct_fallbacks={}",
            if selected_candidate.direct_stereo_fallback {
                "direct-stereo"
            } else {
                "legacy-membership"
            },
            selected_candidate.support,
            selected_candidate.pairs.len(),
            bootstrap.legacy_tracks,
            bootstrap.candidates,
            bootstrap.direct_stereo_fallbacks,
        );
    }
    // Bootstrap topology must not overlap the registration-time owner map in
    // memory or semantics. Only the selected frame-local observation pairs
    // survive into growth.
    drop(legacy_tracks);
    drop(legacy_candidates);
    drop(direct_candidates);
    let mut growth = dynamic_grow_from_seed(
        rig,
        frames,
        features,
        pairwise,
        config,
        &csr,
        &observations,
        &image_assignment,
        best_seed,
        &selected_candidate.pairs,
        bootstrap,
    )?
    .ok_or(RigSfmError::InsufficientSeedStructure {
        frame: best_seed,
        required: required_seed_landmarks,
        triangulated: selected_landmarks,
    })?;

    let mut bundle_adjustment = if config.final_bundle_adjustment && config.final_ba_passes > 0 {
        run_windowed_final_ba(
            rig,
            features,
            &image_assignment,
            config,
            &mut growth.frame_poses,
            &mut growth.image_poses,
            &mut growth.tracks,
        )?
    } else {
        None
    };
    growth.work.structure_refined_tracks += refine_rig_structure(
        rig,
        features,
        &image_assignment,
        &growth.image_poses,
        config,
        &mut growth.tracks,
    );
    for _ in 0..config.final_filter_refinement_passes {
        let pruned = filter_positioned_track_observations(
            rig,
            features,
            &image_assignment,
            &growth.image_poses,
            config.max_reprojection_error_px,
            &mut growth.tracks,
        );
        if pruned == 0 {
            break;
        }
        growth.work.final_filter_refinement_passes += 1;
        growth.work.final_filter_refinement_pruned_observations += pruned;
        if config.final_bundle_adjustment && config.final_ba_passes > 0 {
            let refinement = run_windowed_final_ba(
                rig,
                features,
                &image_assignment,
                config,
                &mut growth.frame_poses,
                &mut growth.image_poses,
                &mut growth.tracks,
            )?;
            merge_rig_ba_stats(&mut bundle_adjustment, refinement);
        }
        growth.work.structure_refined_tracks += refine_rig_structure(
            rig,
            features,
            &image_assignment,
            &growth.image_poses,
            config,
            &mut growth.tracks,
        );
    }
    let tracks = public_tracks_from_working_tracks(
        rig,
        features,
        &image_assignment,
        &growth.frame_poses,
        &growth.image_poses,
        config,
        &growth.tracks,
    );
    let (error_sum, error_count) = reprojection_error_registered(
        rig,
        &image_assignment,
        &growth.frame_poses,
        &growth.image_poses,
        &tracks,
    );
    Ok(RigSfmResult {
        registered_frames: growth
            .frame_poses
            .iter()
            .filter(|pose| pose.is_some())
            .count(),
        registered_images: growth
            .image_poses
            .iter()
            .filter(|pose| pose.is_some())
            .count(),
        mean_reprojection_error_px: if error_count == 0 {
            0.0
        } else {
            error_sum / error_count as f64
        },
        frame_poses: growth.frame_poses,
        image_poses: growth.image_poses,
        tracks,
        seed_frame_index: growth.seed_frame_index,
        track_build_stats: growth.track_build_stats,
        work: growth.work,
        bundle_adjustment,
    })
}
