//! Cross-boundary diagnostic types, the frame DSU, DLT checks and
//! connectivity reports.

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DiagnosticCrossBoundaryTrack {
    pub(super) track_id: usize,
    pub(super) left_keys: Vec<ObservationKey>,
    pub(super) right_keys: Vec<ObservationKey>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum DiagnosticDltStatus {
    Short {
        observations: usize,
    },
    Pass {
        observations: usize,
        mean_error: f64,
        max_error: f64,
    },
    Rejected {
        observations: usize,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DiagnosticDltResult {
    pub(super) status: DiagnosticDltStatus,
    pub(super) landmark: Option<LandmarkOutput>,
}

#[derive(Debug, Clone)]
pub(super) struct DiagnosticCandidateObservation {
    pub(super) track_id: usize,
    pub(super) key: ObservationKey,
    pub(super) correspondence: GeneralizedCorrespondence2D3D,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DiagnosticCandidateAudit {
    pub(super) frame_id: u64,
    pub(super) full_cross_success_track_ids: Vec<usize>,
    pub(super) retriangulated_failure_track_ids: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CrossBoundaryDiagnosticSummary {
    pub(super) boundary_frame: u64,
    pub(super) cross_tracks: usize,
    pub(super) left_pass: usize,
    pub(super) left_short: usize,
    pub(super) left_rejected: usize,
    pub(super) right_pass: usize,
    pub(super) right_short: usize,
    pub(super) right_rejected: usize,
    pub(super) both_sides_two_observations: usize,
    pub(super) both_sides_dlt_pass: usize,
    pub(super) candidate_frames: usize,
    pub(super) pnp_reports: usize,
    pub(super) distinct_anchor_frames: usize,
    pub(super) target_support_attempts: usize,
    pub(super) target_support_successes: usize,
    pub(super) full_cross_support_attempts: usize,
    pub(super) full_cross_support_successes: usize,
    pub(super) existing_support_checks: usize,
    pub(super) fixed_xyz_gate_failures: usize,
    pub(super) retriangulated_existing_support_failures: usize,
    pub(super) cap_skips: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiagnosticConnectivityReport {
    pub(super) supported_images: BTreeSet<u64>,
    pub(super) supported_frames: BTreeSet<u64>,
    pub(super) supported_component_sizes: Vec<usize>,
    pub(super) removed_tracks: usize,
    pub(super) removed_observations: usize,
    pub(super) added_tracks: usize,
    pub(super) added_observations: usize,
}

#[derive(Debug, Clone)]
struct DiagnosticDsu {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DiagnosticDsu {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.size[left_root] < self.size[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        self.size[left_root] += self.size[right_root];
    }
}

pub(super) fn diagnostic_dlt_result(
    track_id: usize,
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> DiagnosticDltResult {
    if keys.len() < MIN_TRACK_OBSERVATIONS {
        return DiagnosticDltResult {
            status: DiagnosticDltStatus::Short {
                observations: keys.len(),
            },
            landmark: None,
        };
    }
    let track = GlobalTrack {
        observations: keys.to_vec(),
    };
    match triangulate_track(track_id, &track, observations, images, cameras) {
        Ok(landmark) => DiagnosticDltResult {
            status: DiagnosticDltStatus::Pass {
                observations: keys.len(),
                mean_error: landmark.mean_error,
                max_error: landmark.max_error,
            },
            landmark: Some(landmark),
        },
        Err(reason) => DiagnosticDltResult {
            status: DiagnosticDltStatus::Rejected {
                observations: keys.len(),
                reason,
            },
            landmark: None,
        },
    }
}

pub(super) fn diagnostic_dlt_status_label(status: &DiagnosticDltStatus) -> String {
    match status {
        DiagnosticDltStatus::Short { observations } => format!("short({observations})"),
        DiagnosticDltStatus::Pass {
            observations,
            mean_error,
            max_error,
        } => format!("pass(obs={observations},mean={mean_error:.6},max={max_error:.6})"),
        DiagnosticDltStatus::Rejected {
            observations,
            reason,
        } => format!("reject(obs={observations},reason={reason})"),
    }
}

fn sort_diagnostic_keys(
    keys: &mut [ObservationKey],
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<(), String> {
    for key in keys.iter().copied() {
        if !images.contains_key(&key.global_image_id) {
            return Err("diagnostic track references unknown image".to_owned());
        }
    }
    // Keep the same ObservationKey ordering used by the production
    // triangulator. In particular, bounded endpoint sampling must not change
    // merely because this report groups keys into left/right frame sets.
    keys.sort_unstable();
    Ok(())
}

pub(super) fn diagnostic_frame_range(
    keys: &[ObservationKey],
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<String, String> {
    let mut frames = keys.iter().map(|key| {
        images
            .get(&key.global_image_id)
            .map(|image| image.atlas.frame_id)
            .ok_or_else(|| "diagnostic track references unknown image".to_owned())
    });
    let Some(first) = frames.next() else {
        return Ok("none".to_owned());
    };
    let first = first?;
    let (minimum, maximum) = frames.try_fold((first, first), |(minimum, maximum), frame| {
        let frame = frame?;
        Ok::<_, String>((minimum.min(frame), maximum.max(frame)))
    })?;
    Ok(format!("{minimum}..{maximum}"))
}

pub(super) fn collect_diagnostic_cross_tracks(
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    left_max_frame: u64,
) -> Result<Vec<DiagnosticCrossBoundaryTrack>, String> {
    let mut result = Vec::new();
    let mut cross_observations = 0usize;
    for (track_id, track) in store.tracks.iter().enumerate() {
        let mut keys = track.observations.clone();
        sort_diagnostic_keys(&mut keys, images)?;
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();
        for key in keys {
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "diagnostic track references unknown image".to_owned())?;
            if image.atlas.frame_id <= left_max_frame {
                left_keys.push(key);
            } else {
                right_keys.push(key);
            }
        }
        if !left_keys.is_empty() && !right_keys.is_empty() {
            cross_observations = cross_observations
                .checked_add(left_keys.len() + right_keys.len())
                .ok_or_else(|| "cross-boundary diagnostic observation count overflow".to_owned())?;
            if cross_observations > DIAGNOSTIC_MAX_CROSS_OBSERVATIONS {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} observations",
                    DIAGNOSTIC_MAX_CROSS_OBSERVATIONS
                ));
            }
            result.push(DiagnosticCrossBoundaryTrack {
                track_id,
                left_keys,
                right_keys,
            });
            if result.len() > DIAGNOSTIC_MAX_CROSS_TRACKS {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} tracks",
                    DIAGNOSTIC_MAX_CROSS_TRACKS
                ));
            }
        }
    }
    Ok(result)
}

pub(super) fn diagnostic_distinct_track_ids(
    observations: &[DiagnosticCandidateObservation],
) -> BTreeSet<usize> {
    observations
        .iter()
        .map(|observation| observation.track_id)
        .collect()
}

pub(super) fn diagnostic_inlier_track_ids(
    inliers: &[usize],
    observations: &[DiagnosticCandidateObservation],
) -> BTreeSet<usize> {
    inliers
        .iter()
        .filter_map(|index| observations.get(*index))
        .map(|observation| observation.track_id)
        .collect()
}

pub(super) fn diagnostic_pose_is_finite(pose: &Pose) -> bool {
    pose.world_to_camera
        .translation
        .iter()
        .all(|value| value.is_finite())
        && pose
            .world_to_camera
            .rotation
            .quaternion()
            .coords
            .iter()
            .all(|value| value.is_finite())
}

pub(super) fn validate_diagnostic_rig_pose(
    manifest: &RigManifest,
    frame_id: u64,
    images: &BTreeMap<u64, GlobalImage>,
    rig_pose: &Pose,
) -> Result<usize, String> {
    if !diagnostic_pose_is_finite(rig_pose) {
        return Err("generalized PnP returned a non-finite rig pose".to_owned());
    }
    let mut sensor_count = 0;
    for image in images
        .values()
        .filter(|image| image.atlas.frame_id == frame_id)
    {
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| "diagnostic image references unknown sensor".to_owned())?;
        let sensor_pose = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        if !sensor_pose
            .translation
            .iter()
            .all(|value| value.is_finite())
            || !sensor_pose
                .rotation
                .quaternion()
                .coords
                .iter()
                .all(|value| value.is_finite())
        {
            return Err("generalized PnP sensor pose is non-finite".to_owned());
        }
        let round_trip = sensor.sensor_from_rig.inverse().compose(&sensor_pose);
        let translation_error =
            (round_trip.translation - rig_pose.world_to_camera.translation).norm();
        let rotation_error = round_trip
            .rotation
            .rotation_to(&rig_pose.world_to_camera.rotation)
            .angle();
        if !translation_error.is_finite()
            || !rotation_error.is_finite()
            || translation_error > 1.0e-8
            || rotation_error > 1.0e-8
        {
            return Err(format!(
                "generalized PnP fixed-extrinsic round trip failed for frame {frame_id}"
            ));
        }
        sensor_count += 1;
    }
    if sensor_count == 0 {
        return Err(format!("diagnostic frame {frame_id} has no atlas images"));
    }
    Ok(sensor_count)
}

pub(super) fn diagnostic_landmark_frame_index(
    landmarks: &[LandmarkOutput],
    images: &BTreeMap<u64, GlobalImage>,
    candidate_frames: &BTreeSet<u64>,
) -> Result<BTreeMap<u64, Vec<usize>>, String> {
    let mut indexed = BTreeMap::<u64, BTreeSet<usize>>::new();
    for (landmark_id, landmark) in landmarks.iter().enumerate() {
        let mut seen_frames = BTreeSet::new();
        for key in &landmark.observations {
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "baseline landmark references unknown image".to_owned())?;
            if candidate_frames.contains(&image.atlas.frame_id)
                && seen_frames.insert(image.atlas.frame_id)
            {
                indexed
                    .entry(image.atlas.frame_id)
                    .or_default()
                    .insert(landmark_id);
            }
        }
    }
    Ok(indexed
        .into_iter()
        .map(|(frame, ids)| (frame, ids.into_iter().collect()))
        .collect())
}

pub(super) fn diagnostic_track_with_keys(
    left_keys: &[ObservationKey],
    right_keys: &[ObservationKey],
) -> GlobalTrack {
    let mut keys = left_keys.to_vec();
    keys.extend_from_slice(right_keys);
    keys.sort_unstable();
    keys.dedup();
    GlobalTrack { observations: keys }
}

fn diagnostic_frame_ids(images: &BTreeMap<u64, GlobalImage>) -> Vec<u64> {
    images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn diagnostic_image_ids(images: &BTreeMap<u64, GlobalImage>) -> BTreeSet<u64> {
    images.keys().copied().collect()
}

fn diagnostic_add_track_to_dsu(
    keys: &[ObservationKey],
    images: &BTreeMap<u64, GlobalImage>,
    frame_indices: &BTreeMap<u64, usize>,
    dsu: &mut DiagnosticDsu,
    supported_images: &mut BTreeSet<u64>,
    supported_frames: &mut BTreeSet<u64>,
) -> Result<(), String> {
    let mut track_frames = BTreeSet::new();
    for key in keys {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "connectivity track references unknown image".to_owned())?;
        let frame_id = image.atlas.frame_id;
        let frame_index = *frame_indices
            .get(&frame_id)
            .ok_or_else(|| "connectivity frame index is missing".to_owned())?;
        supported_images.insert(key.global_image_id);
        supported_frames.insert(frame_id);
        track_frames.insert(frame_index);
    }
    let mut indices = track_frames.into_iter();
    if let Some(first) = indices.next() {
        for index in indices {
            dsu.union(first, index);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SupportConnectivity {
    pub(super) supported_images: BTreeSet<u64>,
    pub(super) supported_frames: BTreeSet<u64>,
    pub(super) frame_components: BTreeMap<u64, usize>,
    pub(super) component_count: usize,
}

pub(super) fn support_connectivity_for_landmarks(
    baseline_landmarks: &[LandmarkOutput],
    replacements: &BTreeMap<usize, LandmarkOutput>,
    removed_track_ids: &BTreeSet<usize>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<SupportConnectivity, String> {
    let frame_ids = diagnostic_frame_ids(images);
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    // Compact per-image lookup and flags avoid repeated tree insertions for
    // every observation. No landmark incidence or frame-pair graph is retained.
    let image_slots = images
        .iter()
        .enumerate()
        .map(|(slot, (&id, image))| (id, (slot, frame_indices[&image.atlas.frame_id])))
        .collect::<BTreeMap<_, _>>();
    let mut image_supported = vec![false; images.len()];
    let mut frame_supported = vec![false; frame_ids.len()];
    for (index, baseline) in baseline_landmarks.iter().enumerate() {
        if removed_track_ids.contains(&baseline.track_id) {
            continue;
        }
        let keys = replacements
            .get(&index)
            .map(|landmark| landmark.observations.as_slice())
            .unwrap_or(&baseline.observations);
        let mut first_frame = None;
        for key in keys {
            let &(image_slot, frame_slot) = image_slots
                .get(&key.global_image_id)
                .ok_or_else(|| "connectivity track references unknown image".to_owned())?;
            image_supported[image_slot] = true;
            frame_supported[frame_slot] = true;
            if let Some(first) = first_frame {
                dsu.union(first, frame_slot);
            } else {
                first_frame = Some(frame_slot);
            }
        }
    }
    let supported_images = images
        .keys()
        .enumerate()
        .filter_map(|(slot, &id)| image_supported[slot].then_some(id))
        .collect();
    let supported_frames = frame_ids
        .iter()
        .enumerate()
        .filter_map(|(slot, &id)| frame_supported[slot].then_some(id))
        .collect::<BTreeSet<_>>();
    let mut root_to_component = BTreeMap::<usize, usize>::new();
    let mut frame_components = BTreeMap::new();
    for frame_id in &supported_frames {
        let frame_index = *frame_indices
            .get(frame_id)
            .ok_or_else(|| "supported connectivity frame index is missing".to_owned())?;
        let root = dsu.find(frame_index);
        let next = root_to_component.len();
        let component = *root_to_component.entry(root).or_insert(next);
        frame_components.insert(*frame_id, component);
    }
    Ok(SupportConnectivity {
        supported_images,
        supported_frames,
        component_count: root_to_component.len(),
        frame_components,
    })
}

#[cfg(test)]
pub(super) fn support_connectivity_reference(
    baseline_landmarks: &[LandmarkOutput],
    replacements: &BTreeMap<usize, LandmarkOutput>,
    removed_track_ids: &BTreeSet<usize>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<SupportConnectivity, String> {
    let frame_ids = diagnostic_frame_ids(images);
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    let mut supported_images = BTreeSet::new();
    let mut supported_frames = BTreeSet::new();
    for (index, baseline) in baseline_landmarks.iter().enumerate() {
        if removed_track_ids.contains(&baseline.track_id) {
            continue;
        }
        let keys = replacements
            .get(&index)
            .map(|landmark| landmark.observations.as_slice())
            .unwrap_or(&baseline.observations);
        diagnostic_add_track_to_dsu(
            keys,
            images,
            &frame_indices,
            &mut dsu,
            &mut supported_images,
            &mut supported_frames,
        )?;
    }
    let mut root_to_component = BTreeMap::<usize, usize>::new();
    let mut frame_components = BTreeMap::new();
    for frame_id in &supported_frames {
        let frame_index = *frame_indices
            .get(frame_id)
            .ok_or_else(|| "supported connectivity frame index is missing".to_owned())?;
        let root = dsu.find(frame_index);
        let next = root_to_component.len();
        let component = *root_to_component.entry(root).or_insert(next);
        frame_components.insert(*frame_id, component);
    }
    Ok(SupportConnectivity {
        supported_images,
        supported_frames,
        component_count: root_to_component.len(),
        frame_components,
    })
}

pub(super) fn connectivity_not_split(
    baseline: &SupportConnectivity,
    candidate: &SupportConnectivity,
) -> bool {
    if baseline.supported_images != candidate.supported_images
        || baseline.supported_frames != candidate.supported_frames
        || candidate.component_count > baseline.component_count
    {
        return false;
    }
    let mut component_map = BTreeMap::<usize, usize>::new();
    for frame_id in &baseline.supported_frames {
        let Some(&baseline_component) = baseline.frame_components.get(frame_id) else {
            return false;
        };
        let Some(&candidate_component) = candidate.frame_components.get(frame_id) else {
            return false;
        };
        if let Some(previous) = component_map.insert(baseline_component, candidate_component) {
            if previous != candidate_component {
                return false;
            }
        }
    }
    true
}

pub(super) fn diagnostic_connectivity_report(
    baseline_landmarks: &[LandmarkOutput],
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    removed_track_ids: &BTreeSet<usize>,
    added_track_ids: &BTreeSet<usize>,
) -> Result<DiagnosticConnectivityReport, String> {
    let frame_ids = diagnostic_frame_ids(images);
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let actually_removed = removed_track_ids
        .difference(added_track_ids)
        .copied()
        .collect::<BTreeSet<_>>();

    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    let mut supported_images = BTreeSet::new();
    let mut supported_frames = BTreeSet::new();
    let mut removed_observations = 0;
    let mut removed_seen = BTreeSet::new();
    let mut added_existing_seen = BTreeSet::new();
    for landmark in baseline_landmarks {
        if actually_removed.contains(&landmark.track_id) {
            removed_observations += landmark.observations.len();
            removed_seen.insert(landmark.track_id);
            continue;
        }
        if added_track_ids.contains(&landmark.track_id) {
            added_existing_seen.insert(landmark.track_id);
        }
        diagnostic_add_track_to_dsu(
            &landmark.observations,
            images,
            &frame_indices,
            &mut dsu,
            &mut supported_images,
            &mut supported_frames,
        )?;
    }

    if let Some(track_id) = actually_removed.difference(&removed_seen).next() {
        return Err(format!(
            "connectivity removal references non-baseline track {track_id}"
        ));
    }

    let mut added_observations = 0;
    let mut added_new = 0;
    for track_id in added_track_ids.difference(&added_existing_seen) {
        let track = store
            .tracks
            .get(*track_id)
            .ok_or_else(|| format!("connectivity addition references unknown track {track_id}"))?;
        added_new += 1;
        added_observations += track.observations.len();
        diagnostic_add_track_to_dsu(
            &track.observations,
            images,
            &frame_indices,
            &mut dsu,
            &mut supported_images,
            &mut supported_frames,
        )?;
    }

    let mut component_counts = BTreeMap::<usize, usize>::new();
    for frame_id in &supported_frames {
        let frame_index = *frame_indices
            .get(frame_id)
            .ok_or_else(|| "supported connectivity frame index is missing".to_owned())?;
        let root = dsu.find(frame_index);
        *component_counts.entry(root).or_default() += 1;
    }
    let mut supported_component_sizes = component_counts.into_values().collect::<Vec<_>>();
    supported_component_sizes.sort_unstable();
    Ok(DiagnosticConnectivityReport {
        supported_images,
        supported_frames,
        supported_component_sizes,
        removed_tracks: actually_removed.len(),
        removed_observations,
        added_tracks: added_new,
        added_observations,
    })
}

pub(super) fn diagnostic_format_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn diagnostic_format_track_ids(ids: &[usize]) -> String {
    ids.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn diagnostic_emit_connectivity(
    baseline_landmarks: &[LandmarkOutput],
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    audits: &[DiagnosticCandidateAudit],
) -> Result<(), String> {
    let all_images = diagnostic_image_ids(images);
    let all_frames = diagnostic_frame_ids(images)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let empty = BTreeSet::new();
    let baseline =
        diagnostic_connectivity_report(baseline_landmarks, store, images, &empty, &empty)?;
    let baseline_unsupported_images = all_images
        .difference(&baseline.supported_images)
        .copied()
        .collect::<BTreeSet<_>>();
    let baseline_unsupported_frames = all_frames
        .difference(&baseline.supported_frames)
        .copied()
        .collect::<BTreeSet<_>>();
    println!(
        "cross_boundary_connectivity_baseline supported_images={} unsupported_images={} supported_frames={} unsupported_frames={} supported_component_count={} supported_component_sizes={}",
        baseline.supported_images.len(),
        baseline_unsupported_images.len(),
        baseline.supported_frames.len(),
        baseline_unsupported_frames.len(),
        baseline.supported_component_sizes.len(),
        diagnostic_format_usize_list(&baseline.supported_component_sizes),
    );

    for audit in audits {
        let removed = audit
            .retriangulated_failure_track_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let added = audit
            .full_cross_success_track_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let candidate =
            diagnostic_connectivity_report(baseline_landmarks, store, images, &removed, &added)?;
        let candidate_unsupported_images = all_images
            .difference(&candidate.supported_images)
            .copied()
            .collect::<BTreeSet<_>>();
        let candidate_unsupported_frames = all_frames
            .difference(&candidate.supported_frames)
            .copied()
            .collect::<BTreeSet<_>>();
        let existing_unsupported_images = baseline_unsupported_images
            .intersection(&candidate_unsupported_images)
            .count();
        let existing_unsupported_frames = baseline_unsupported_frames
            .intersection(&candidate_unsupported_frames)
            .count();
        let recovered_existing_images = baseline_unsupported_images
            .difference(&candidate_unsupported_images)
            .count();
        let recovered_existing_frames = baseline_unsupported_frames
            .difference(&candidate_unsupported_frames)
            .count();
        let new_unsupported_images = candidate_unsupported_images
            .difference(&baseline_unsupported_images)
            .count();
        let new_unsupported_frames = candidate_unsupported_frames
            .difference(&baseline_unsupported_frames)
            .count();
        println!(
            "cross_boundary_connectivity frame={} full_cross_success_tracks={} full_cross_success_ids={} retriangulated_failure_tracks={} retriangulated_failure_ids={} removed_tracks={} removed_observations={} added_tracks={} added_observations={} supported_images={} unsupported_images={} supported_frames={} unsupported_frames={} existing_unsupported_images={} existing_unsupported_frames={} recovered_existing_images={} recovered_existing_frames={} new_unsupported_images={} new_unsupported_frames={} supported_component_count={} supported_component_sizes={}",
            audit.frame_id,
            audit.full_cross_success_track_ids.len(),
            diagnostic_format_track_ids(&audit.full_cross_success_track_ids),
            audit.retriangulated_failure_track_ids.len(),
            diagnostic_format_track_ids(&audit.retriangulated_failure_track_ids),
            candidate.removed_tracks,
            candidate.removed_observations,
            candidate.added_tracks,
            candidate.added_observations,
            candidate.supported_images.len(),
            candidate_unsupported_images.len(),
            candidate.supported_frames.len(),
            candidate_unsupported_frames.len(),
            existing_unsupported_images,
            existing_unsupported_frames,
            recovered_existing_images,
            recovered_existing_frames,
            new_unsupported_images,
            new_unsupported_frames,
            candidate.supported_component_sizes.len(),
            diagnostic_format_usize_list(&candidate.supported_component_sizes),
        );
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn diagnostic_component_sizes_from_frame_tracks(
    frame_ids: &[u64],
    track_frames: &[Vec<u64>],
) -> (Vec<usize>, BTreeSet<u64>) {
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    let mut supported = BTreeSet::new();
    for track in track_frames {
        let mut indices = track
            .iter()
            .filter_map(|frame_id| {
                supported.insert(*frame_id);
                frame_indices.get(frame_id).copied()
            })
            .collect::<BTreeSet<_>>()
            .into_iter();
        if let Some(first) = indices.next() {
            for index in indices {
                dsu.union(first, index);
            }
        }
    }
    let mut counts = BTreeMap::<usize, usize>::new();
    for frame_id in &supported {
        if let Some(index) = frame_indices.get(frame_id) {
            *counts.entry(dsu.find(*index)).or_default() += 1;
        }
    }
    let mut sizes = counts.into_values().collect::<Vec<_>>();
    sizes.sort_unstable();
    (sizes, supported)
}
