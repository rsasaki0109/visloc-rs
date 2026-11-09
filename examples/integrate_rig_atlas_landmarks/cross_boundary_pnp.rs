//! The non-mutating cross-boundary generalized-PnP diagnostic
//! (`--diagnose-cross-boundary-pnp`).

use super::*;

/// Run the strict, non-mutating cross-boundary generalized-PnP experiment.
///
/// Only tracks whose left-side DLT passes are used as 3D anchors. Each right
/// frame is a separate candidate; there is no pose propagation or whole-gauge
/// transform. The returned pose is used only for diagnostic re-triangulation.
pub(super) fn diagnose_cross_boundary_pnp(
    manifest: &RigManifest,
    left_max_frame: u64,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    baseline_landmarks: &[LandmarkOutput],
) -> Result<CrossBoundaryDiagnosticSummary, String> {
    let cross_tracks = collect_diagnostic_cross_tracks(store, images, left_max_frame)?;
    let rig = build_generalized_rig(manifest, cameras)?;
    let pnp = GeneralizedPnPRansac {
        iterations: RECOVERY_PNP_ITERATIONS,
        reprojection_threshold: RECOVERY_PNP_REPROJECTION_PX,
        seed: RECOVERY_PNP_SEED,
        ..GeneralizedPnPRansac::default()
    };
    let mut summary = CrossBoundaryDiagnosticSummary {
        boundary_frame: left_max_frame,
        cross_tracks: cross_tracks.len(),
        ..CrossBoundaryDiagnosticSummary::default()
    };
    let mut evaluated = Vec::with_capacity(cross_tracks.len());
    for cross in cross_tracks {
        let left = diagnostic_dlt_result(
            cross.track_id,
            &cross.left_keys,
            &store.observations,
            images,
            cameras,
        );
        let right = diagnostic_dlt_result(
            cross.track_id,
            &cross.right_keys,
            &store.observations,
            images,
            cameras,
        );
        match &left.status {
            DiagnosticDltStatus::Pass { .. } => summary.left_pass += 1,
            DiagnosticDltStatus::Short { .. } => summary.left_short += 1,
            DiagnosticDltStatus::Rejected { .. } => summary.left_rejected += 1,
        }
        match &right.status {
            DiagnosticDltStatus::Pass { .. } => summary.right_pass += 1,
            DiagnosticDltStatus::Short { .. } => summary.right_short += 1,
            DiagnosticDltStatus::Rejected { .. } => summary.right_rejected += 1,
        }
        if cross.left_keys.len() >= MIN_TRACK_OBSERVATIONS
            && cross.right_keys.len() >= MIN_TRACK_OBSERVATIONS
        {
            summary.both_sides_two_observations += 1;
        }
        if matches!(&left.status, DiagnosticDltStatus::Pass { .. })
            && matches!(&right.status, DiagnosticDltStatus::Pass { .. })
        {
            summary.both_sides_dlt_pass += 1;
        }
        let left_range = diagnostic_frame_range(&cross.left_keys, images)?;
        let right_range = diagnostic_frame_range(&cross.right_keys, images)?;
        println!(
            "cross_boundary_track track_id={} left_obs={} left_frames={} right_obs={} right_frames={} left_dlt={} right_dlt={}",
            cross.track_id,
            cross.left_keys.len(),
            left_range,
            cross.right_keys.len(),
            right_range,
            diagnostic_dlt_status_label(&left.status),
            diagnostic_dlt_status_label(&right.status),
        );
        evaluated.push((cross, left, right));
    }

    let mut candidates_by_frame = BTreeMap::<u64, Vec<DiagnosticCandidateObservation>>::new();
    let mut candidate_observations = 0usize;
    for (cross, left, _) in &evaluated {
        let Some(anchor) = left.landmark.as_ref() else {
            continue;
        };
        for key in &cross.right_keys {
            candidate_observations = candidate_observations
                .checked_add(1)
                .ok_or_else(|| "diagnostic candidate observation count overflow".to_owned())?;
            if candidate_observations > DIAGNOSTIC_MAX_CROSS_OBSERVATIONS {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} candidate observations",
                    DIAGNOSTIC_MAX_CROSS_OBSERVATIONS
                ));
            }
            let image = images.get(&key.global_image_id).ok_or_else(|| {
                "diagnostic right observation references unknown image".to_owned()
            })?;
            let observation = store
                .observations
                .get(key)
                .ok_or_else(|| "diagnostic right observation is missing".to_owned())?;
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
                    "cross-boundary diagnostic exceeds {} candidate frames",
                    DIAGNOSTIC_MAX_CANDIDATE_FRAMES
                ));
            }
        }
    }
    for observations in candidates_by_frame.values_mut() {
        observations.sort_by_key(|observation| (observation.track_id, observation.key));
    }
    summary.candidate_frames = candidates_by_frame.len();
    let candidate_frames = candidates_by_frame.keys().copied().collect::<BTreeSet<_>>();
    let baseline_by_frame =
        diagnostic_landmark_frame_index(baseline_landmarks, images, &candidate_frames)?;
    let cross_by_id = evaluated
        .iter()
        .map(|(cross, _, _)| (cross.track_id, cross))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_audits = Vec::new();

    for (frame_id, frame_candidates) in candidates_by_frame {
        let distinct_candidates = diagnostic_distinct_track_ids(&frame_candidates);
        if frame_candidates.len() > DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME {
            summary.cap_skips += 1;
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=cap-exceeded correspondences={} distinct_tracks={} cap={}",
                frame_candidates.len(),
                distinct_candidates.len(),
                DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME,
            );
            continue;
        }
        if distinct_candidates.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=insufficient-distinct-candidates correspondences={} distinct_tracks={} required={}",
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
                "cross_boundary_pnp_frame frame={frame_id} status=pnp-failed correspondences={} distinct_tracks={}",
                correspondences.len(),
                distinct_candidates.len(),
            );
            continue;
        };
        summary.pnp_reports += 1;
        let distinct_inliers = diagnostic_inlier_track_ids(&report.inliers, &frame_candidates);
        if distinct_inliers.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=insufficient-distinct-inliers correspondences={} candidate_tracks={} inliers={} distinct_inliers={} required={}",
                correspondences.len(),
                distinct_candidates.len(),
                report.inliers.len(),
                distinct_inliers.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        let mut candidate_pose_overrides = BTreeMap::new();
        compose_recovered_frame_poses(
            manifest,
            frame_id,
            &report.pose,
            images,
            &mut candidate_pose_overrides,
        )?;
        let sensor_count = validate_diagnostic_rig_pose(manifest, frame_id, images, &report.pose)?;
        summary.distinct_anchor_frames += 1;

        let mut target_support_tracks = 0;
        let mut full_cross_support_tracks = 0;
        let mut target_support_observations = 0;
        let mut full_cross_support_observations = 0;
        let mut candidate_audit = DiagnosticCandidateAudit {
            frame_id,
            ..DiagnosticCandidateAudit::default()
        };
        // Only RANSAC-distinct inlier tracks are eligible for the support
        // experiment. Outlier candidate tracks remain visible in the
        // candidate_tracks field but cannot be presented as recovered support.
        for track_id in &distinct_inliers {
            let Some(cross) = cross_by_id.get(track_id) else {
                return Err(format!("missing diagnostic cross track {track_id}"));
            };
            let target_keys = frame_candidates
                .iter()
                .filter(|observation| observation.track_id == *track_id)
                .map(|observation| observation.key)
                .collect::<Vec<_>>();
            let target_track = diagnostic_track_with_keys(&cross.left_keys, &target_keys);
            if let Ok(landmark) = triangulate_track_with_pose_overrides(
                *track_id,
                &target_track,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) {
                target_support_tracks += 1;
                target_support_observations += landmark.observations.len();
            }
            let full_track = diagnostic_track_with_keys(&cross.left_keys, &cross.right_keys);
            if let Ok(landmark) = triangulate_track_with_pose_overrides(
                *track_id,
                &full_track,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) {
                full_cross_support_tracks += 1;
                full_cross_support_observations += landmark.observations.len();
                candidate_audit.full_cross_success_track_ids.push(*track_id);
            }
        }
        summary.target_support_attempts += distinct_inliers.len();
        summary.target_support_successes += target_support_tracks;
        summary.full_cross_support_attempts += distinct_inliers.len();
        summary.full_cross_support_successes += full_cross_support_tracks;

        let mut fixed_xyz_gate_failures = 0;
        let mut retriangulated_existing_support_failures = 0;
        let affected_landmarks = baseline_by_frame
            .get(&frame_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for landmark_id in affected_landmarks {
            let landmark = baseline_landmarks
                .get(*landmark_id)
                .ok_or_else(|| "diagnostic baseline landmark index is invalid".to_owned())?;
            let fixed_xyz_failed = match evaluate_landmark_metrics(
                landmark,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) {
                Ok(metrics) => {
                    metrics.mean_error > MAX_MEAN_REPROJECTION_PX
                        || metrics.max_error > MAX_REPROJECTION_PX
                }
                Err(_) => true,
            };
            if fixed_xyz_failed {
                fixed_xyz_gate_failures += 1;
            }
            let retriangulated_failed = store
                .tracks
                .get(landmark.track_id)
                .map(|track| {
                    triangulate_track_with_pose_overrides(
                        landmark.track_id,
                        track,
                        &store.observations,
                        images,
                        cameras,
                        &candidate_pose_overrides,
                    )
                    .is_err()
                })
                .unwrap_or(true);
            if retriangulated_failed {
                retriangulated_existing_support_failures += 1;
                candidate_audit
                    .retriangulated_failure_track_ids
                    .push(landmark.track_id);
            }
        }
        summary.fixed_xyz_gate_failures += fixed_xyz_gate_failures;
        summary.existing_support_checks += affected_landmarks.len();
        summary.retriangulated_existing_support_failures +=
            retriangulated_existing_support_failures;
        println!(
            "cross_boundary_pnp_frame frame={frame_id} status=anchor-success correspondences={} candidate_tracks={} inliers={} distinct_inliers={} mean_px={:.6} max_px={:.6} sensors={} target_support_attempts={} target_support_successes={} target_support_observations={} full_cross_support_attempts={} full_cross_support_successes={} full_cross_support_observations={} affected_landmarks={} fixed_xyz_gate_failures={} retriangulated_existing_support_failures={}",
            correspondences.len(),
            distinct_candidates.len(),
            report.inliers.len(),
            distinct_inliers.len(),
            report.mean_reprojection_error,
            report.max_reprojection_error,
            sensor_count,
            distinct_inliers.len(),
            target_support_tracks,
            target_support_observations,
            distinct_inliers.len(),
            full_cross_support_tracks,
            full_cross_support_observations,
            affected_landmarks.len(),
            fixed_xyz_gate_failures,
            retriangulated_existing_support_failures,
        );
        candidate_audits.push(candidate_audit);
    }
    diagnostic_emit_connectivity(baseline_landmarks, store, images, &candidate_audits)?;
    println!(
        "cross_boundary_pnp_summary boundary_frame={} cross_tracks={} left_pass={} left_short={} left_rejected={} right_pass={} right_short={} right_rejected={} both_sides_two_observations={} both_sides_dlt_pass={} candidate_frames={} pnp_reports={} distinct_anchor_frames={} target_support_attempts={} target_support_successes={} full_cross_support_attempts={} full_cross_support_successes={} existing_support_checks={} fixed_xyz_gate_failures={} retriangulated_existing_support_failures={} cap_skips={} default_model_unchanged=true",
        summary.boundary_frame,
        summary.cross_tracks,
        summary.left_pass,
        summary.left_short,
        summary.left_rejected,
        summary.right_pass,
        summary.right_short,
        summary.right_rejected,
        summary.both_sides_two_observations,
        summary.both_sides_dlt_pass,
        summary.candidate_frames,
        summary.pnp_reports,
        summary.distinct_anchor_frames,
        summary.target_support_attempts,
        summary.target_support_successes,
        summary.full_cross_support_attempts,
        summary.full_cross_support_successes,
        summary.existing_support_checks,
        summary.fixed_xyz_gate_failures,
        summary.retriangulated_existing_support_failures,
        summary.cap_skips,
    );
    Ok(summary)
}
