//! The cross-boundary repair (`--repair-cross-boundary`).

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BoundaryRepairSummary {
    pub(super) status: String,
    pub(super) candidates: usize,
    pub(super) attempted: usize,
    pub(super) accepted_frame: Option<u64>,
    pub(super) added_tracks: usize,
    pub(super) added_observations: usize,
    pub(super) removed_tracks: usize,
    pub(super) removed_observations: usize,
    pub(super) baseline_component_count: usize,
    pub(super) candidate_component_count: usize,
}

pub(super) fn boundary_repair_candidate_is_accepted(
    baseline: &DiagnosticConnectivityReport,
    candidate: &DiagnosticConnectivityReport,
    added_tracks: usize,
) -> bool {
    added_tracks > 0
        && candidate.supported_component_sizes.len() < baseline.supported_component_sizes.len()
        && baseline
            .supported_images
            .is_subset(&candidate.supported_images)
        && baseline
            .supported_frames
            .is_subset(&candidate.supported_frames)
}

fn landmark_output_gate_valid(landmark: &LandmarkOutput) -> bool {
    landmark.observations.len() >= MIN_TRACK_OBSERVATIONS
        && landmark
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        && landmark.errors.len() == landmark.observations.len()
        && landmark.errors.iter().all(|error| error.is_finite())
        && landmark.mean_error.is_finite()
        && landmark.rms_error.is_finite()
        && landmark.max_error.is_finite()
        && landmark.mean_error <= MAX_MEAN_REPROJECTION_PX
        && landmark.max_error <= MAX_REPROJECTION_PX
}

/// Try one deterministic, transactional cross-boundary repair.
///
/// This is intentionally separate from `diagnose_cross_boundary_pnp`: the
/// diagnostic path never mutates a model, while this opt-in path stages one
/// candidate and commits only the first candidate that improves supported
/// frame connectivity without losing any existing support.  Existing
/// landmark observations, rather than the source TrackStore, are authoritative
/// for affected-track re-triangulation.
pub(super) fn repair_cross_boundary(
    manifest: &RigManifest,
    left_max_frame: u64,
    store: &TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
) -> Result<BoundaryRepairSummary, String> {
    let baseline_connectivity = diagnostic_connectivity_report(
        landmarks,
        store,
        images,
        &BTreeSet::new(),
        &BTreeSet::new(),
    )?;
    let cross_tracks = collect_diagnostic_cross_tracks(store, images, left_max_frame)?;
    let rig = build_generalized_rig(manifest, cameras)?;
    let pnp = GeneralizedPnPRansac {
        iterations: RECOVERY_PNP_ITERATIONS,
        reprojection_threshold: RECOVERY_PNP_REPROJECTION_PX,
        seed: RECOVERY_PNP_SEED,
        ..GeneralizedPnPRansac::default()
    };
    let mut cross_by_id = BTreeMap::<usize, DiagnosticCrossBoundaryTrack>::new();
    let mut candidates_by_frame = BTreeMap::<u64, Vec<DiagnosticCandidateObservation>>::new();
    let mut candidate_observations = 0usize;
    for cross in cross_tracks {
        let left = diagnostic_dlt_result(
            cross.track_id,
            &cross.left_keys,
            &store.observations,
            images,
            cameras,
        );
        let Some(anchor) = left.landmark else {
            continue;
        };
        cross_by_id.insert(cross.track_id, cross.clone());
        for key in &cross.right_keys {
            candidate_observations = candidate_observations
                .checked_add(1)
                .ok_or_else(|| "boundary repair observation count overflow".to_owned())?;
            if candidate_observations > DIAGNOSTIC_MAX_CROSS_OBSERVATIONS {
                return Err(format!(
                    "boundary repair exceeds {} candidate observations",
                    DIAGNOSTIC_MAX_CROSS_OBSERVATIONS
                ));
            }
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "boundary repair references unknown image".to_owned())?;
            let observation = store
                .observations
                .get(key)
                .ok_or_else(|| "boundary repair references unknown observation".to_owned())?;
            let frame_candidates = candidates_by_frame.entry(image.atlas.frame_id).or_default();
            frame_candidates.push(DiagnosticCandidateObservation {
                track_id: cross.track_id,
                key: *key,
                correspondence: GeneralizedCorrespondence2D3D {
                    sensor_index: image.atlas.sensor_index,
                    point2d: observation.xy,
                    point3d: anchor.position,
                    confidence: None,
                },
            });
            if candidates_by_frame.len() > DIAGNOSTIC_MAX_CANDIDATE_FRAMES {
                return Err(format!(
                    "boundary repair exceeds {} candidate frames",
                    DIAGNOSTIC_MAX_CANDIDATE_FRAMES
                ));
            }
        }
    }
    for observations in candidates_by_frame.values_mut() {
        observations.sort_by_key(|observation| (observation.track_id, observation.key));
    }

    let mut summary = BoundaryRepairSummary {
        status: "no-accepted-candidate".to_owned(),
        candidates: candidates_by_frame.len(),
        attempted: 0,
        accepted_frame: None,
        added_tracks: 0,
        added_observations: 0,
        removed_tracks: 0,
        removed_observations: 0,
        baseline_component_count: baseline_connectivity.supported_component_sizes.len(),
        candidate_component_count: baseline_connectivity.supported_component_sizes.len(),
    };

    for (frame_id, frame_candidates) in candidates_by_frame {
        summary.attempted += 1;
        let distinct_candidates = diagnostic_distinct_track_ids(&frame_candidates);
        if frame_candidates.len() > DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=correspondence-cap correspondences={} distinct_tracks={}",
                frame_candidates.len(),
                distinct_candidates.len(),
            );
            continue;
        }
        if distinct_candidates.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=insufficient-distinct-candidates correspondences={} distinct_tracks={} required={}",
                frame_candidates.len(),
                distinct_candidates.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        let correspondences = frame_candidates
            .iter()
            .map(|observation| observation.correspondence.clone())
            .collect::<Vec<_>>();
        let Some(report) = pnp.estimate(&rig, &correspondences) else {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=pnp-failed correspondences={} distinct_tracks={}",
                correspondences.len(),
                distinct_candidates.len(),
            );
            continue;
        };
        let distinct_inliers = diagnostic_inlier_track_ids(&report.inliers, &frame_candidates);
        if distinct_inliers.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=insufficient-distinct-inliers pnp_inliers={} distinct_inliers={} required={}",
                report.inliers.len(),
                distinct_inliers.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        // Only candidate ids are retained here.  Scanning the baseline once
        // avoids a second model-sized track-id map for every candidate.
        let baseline_candidate_track_ids = landmarks
            .iter()
            .filter(|landmark| distinct_inliers.contains(&landmark.track_id))
            .map(|landmark| landmark.track_id)
            .collect::<BTreeSet<_>>();
        let mut pose_overrides = BTreeMap::new();
        compose_recovered_frame_poses(
            manifest,
            frame_id,
            &report.pose,
            images,
            &mut pose_overrides,
        )?;
        validate_diagnostic_rig_pose(manifest, frame_id, images, &report.pose)?;

        let mut added_landmarks = Vec::new();
        let mut added_ids = BTreeSet::new();
        let mut added_keys = BTreeSet::new();
        let mut full_cross_failure_count = 0usize;
        let mut full_cross_failure_observations = 0usize;
        for track_id in &distinct_inliers {
            let cross = cross_by_id
                .get(track_id)
                .ok_or_else(|| format!("boundary repair lost cross track {track_id}"))?;
            let full_track = diagnostic_track_with_keys(&cross.left_keys, &cross.right_keys);
            let Ok(landmark) = triangulate_track_with_pose_overrides(
                *track_id,
                &full_track,
                &store.observations,
                images,
                cameras,
                &pose_overrides,
            ) else {
                full_cross_failure_count += 1;
                full_cross_failure_observations += full_track.observations.len();
                continue;
            };
            if baseline_candidate_track_ids.contains(track_id) || !added_ids.insert(*track_id) {
                continue;
            }
            if landmark
                .observations
                .iter()
                .any(|key| !added_keys.insert(*key))
            {
                println!(
                    "boundary_repair_candidate frame={frame_id} status=rejected reason=added-observation-conflict",
                );
                added_landmarks.clear();
                added_ids.clear();
                break;
            }
            added_landmarks.push(landmark);
        }
        if added_landmarks.is_empty() {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=no-full-cross-addition full_cross_failures={} full_cross_failure_observations={}",
                full_cross_failure_count,
                full_cross_failure_observations,
            );
            continue;
        }

        let mut affected = BTreeMap::<usize, LandmarkOutput>::new();
        let mut removed_ids = BTreeSet::new();
        let mut removed_observations = 0usize;
        let mut affected_count = 0usize;
        let mut affected_observations = 0usize;
        let mut affected_cap_exceeded = false;
        for (index, landmark) in landmarks.iter().enumerate() {
            let touches_candidate = landmark.observations.iter().any(|key| {
                images
                    .get(&key.global_image_id)
                    .is_some_and(|image| image.atlas.frame_id == frame_id)
            });
            if !touches_candidate {
                continue;
            }
            affected_count += 1;
            affected_observations = affected_observations
                .checked_add(landmark.observations.len())
                .ok_or_else(|| "boundary repair affected observation count overflow".to_owned())?;
            if affected_count > JOINT_BA_MAX_LANDMARKS
                || affected_observations > JOINT_BA_MAX_OBSERVATIONS
            {
                println!(
                    "boundary_repair_candidate frame={frame_id} status=rejected reason=affected-track-cap affected_landmarks={} affected_observations={} landmark_cap={} observation_cap={}",
                    affected_count,
                    affected_observations,
                    JOINT_BA_MAX_LANDMARKS,
                    JOINT_BA_MAX_OBSERVATIONS,
                );
                affected.clear();
                removed_ids.clear();
                affected_cap_exceeded = true;
                break;
            }
            let original_track = GlobalTrack {
                observations: landmark.observations.clone(),
            };
            match triangulate_track_with_pose_overrides(
                landmark.track_id,
                &original_track,
                &store.observations,
                images,
                cameras,
                &pose_overrides,
            ) {
                Ok(updated) => {
                    affected.insert(index, updated);
                }
                Err(_) => {
                    removed_ids.insert(landmark.track_id);
                    removed_observations += landmark.observations.len();
                }
            }
        }
        if affected_cap_exceeded {
            continue;
        }
        let retained_key_conflict = landmarks.iter().any(|landmark| {
            !removed_ids.contains(&landmark.track_id)
                && landmark
                    .observations
                    .iter()
                    .any(|key| added_keys.contains(key))
        });
        if retained_key_conflict {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=retained-observation-conflict",
            );
            continue;
        }

        let candidate_connectivity =
            diagnostic_connectivity_report(landmarks, store, images, &removed_ids, &added_ids)?;
        if !boundary_repair_candidate_is_accepted(
            &baseline_connectivity,
            &candidate_connectivity,
            added_landmarks.len(),
        ) {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=support-or-connectivity-gate added_tracks={} removed_tracks={} baseline_components={} candidate_components={} baseline_supported_frames={} candidate_supported_frames={}",
                added_landmarks.len(),
                removed_ids.len(),
                baseline_connectivity.supported_component_sizes.len(),
                candidate_connectivity.supported_component_sizes.len(),
                baseline_connectivity.supported_frames.len(),
                candidate_connectivity.supported_frames.len(),
            );
            continue;
        }

        let mut all_remaining_valid = true;
        for (index, landmark) in landmarks.iter().enumerate() {
            if removed_ids.contains(&landmark.track_id) {
                continue;
            }
            let candidate = affected.get(&index).unwrap_or(landmark);
            if !landmark_output_gate_valid(candidate) {
                all_remaining_valid = false;
                break;
            }
            let metrics = evaluate_landmark_metrics(
                candidate,
                &store.observations,
                images,
                cameras,
                &pose_overrides,
            );
            if metrics.is_err()
                || metrics.as_ref().is_ok_and(|metrics| {
                    metrics.mean_error > MAX_MEAN_REPROJECTION_PX
                        || metrics.max_error > MAX_REPROJECTION_PX
                })
            {
                all_remaining_valid = false;
                break;
            }
        }
        if all_remaining_valid {
            for landmark in &added_landmarks {
                if !landmark_output_gate_valid(landmark) {
                    all_remaining_valid = false;
                    break;
                }
            }
        }
        if !all_remaining_valid {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=remaining-landmark-gate removed_tracks={} removed_observations={}",
                removed_ids.len(),
                removed_observations,
            );
            continue;
        }

        let accepted_added_tracks = added_landmarks.len();
        let accepted_added_observations = added_landmarks
            .iter()
            .map(|landmark| landmark.observations.len())
            .sum::<usize>();
        apply_pose_overrides(images, &pose_overrides)?;
        for (index, replacement) in affected {
            landmarks[index] = replacement;
        }
        landmarks.retain(|landmark| !removed_ids.contains(&landmark.track_id));
        landmarks.extend(added_landmarks);
        summary.status = "accepted".to_owned();
        summary.accepted_frame = Some(frame_id);
        summary.added_tracks = accepted_added_tracks;
        summary.added_observations = accepted_added_observations;
        summary.removed_tracks = removed_ids.len();
        summary.removed_observations = removed_observations;
        summary.candidate_component_count = candidate_connectivity.supported_component_sizes.len();
        println!(
            "boundary_repair_candidate frame={frame_id} status=accepted pnp_inliers={} distinct_inliers={} pnp_mean_px={:.6} pnp_max_px={:.6} added_tracks={} added_observations={} removed_tracks={} removed_observations={} supported_component_sizes_before={} supported_component_sizes_after={}",
            report.inliers.len(),
            distinct_inliers.len(),
            report.mean_reprojection_error,
            report.max_reprojection_error,
            summary.added_tracks,
            summary.added_observations,
            summary.removed_tracks,
            summary.removed_observations,
            diagnostic_format_usize_list(&baseline_connectivity.supported_component_sizes),
            diagnostic_format_usize_list(&candidate_connectivity.supported_component_sizes),
        );
        break;
    }
    Ok(summary)
}
