//! Recovery of unsupported frames by generalized PnP
//! (`--recover-zero-support-frames`).

use super::*;

#[derive(Debug, Clone)]
pub(super) struct RecoveryResult {
    pub(super) pose_overrides: BTreeMap<u64, Pose>,
    pub(super) landmarks: Vec<LandmarkOutput>,
    pub(super) summary: RecoverySummary,
}

#[derive(Debug, Clone)]
struct RecoveryAnchor {
    track_id: usize,
    correspondence_indices: Vec<usize>,
}

/// Recover initially unsupported rig frames from rejected tracks without
/// letting one unsupported frame bootstrap another.  Anchor points are
/// triangulated only from the frame set supported by the fixed-pose model;
/// the target frame is then estimated with the existing deterministic
/// generalized-PnP implementation and fixed sensor extrinsics.  Everything
/// for one frame is staged and committed only after the six-inlier and
/// six-observation gates pass.
pub(super) fn recover_zero_support_frames(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    baseline_landmarks: &[LandmarkOutput],
) -> Result<RecoveryResult, String> {
    let initial_supported_frames = baseline_landmarks
        .iter()
        .flat_map(|landmark| landmark.observations.iter())
        .filter_map(|key| images.get(&key.global_image_id))
        .map(|image| image.atlas.frame_id)
        .collect::<BTreeSet<_>>();
    let atlas_frames = images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<BTreeSet<_>>();
    let zero_support_frames = atlas_frames
        .difference(&initial_supported_frames)
        .copied()
        .collect::<Vec<_>>();
    let mut summary = RecoverySummary {
        frames_considered: zero_support_frames.len(),
        ..RecoverySummary::default()
    };
    let rig = build_generalized_rig(manifest, cameras)?;
    let pnp = GeneralizedPnPRansac {
        iterations: RECOVERY_PNP_ITERATIONS,
        reprojection_threshold: RECOVERY_PNP_REPROJECTION_PX,
        seed: RECOVERY_PNP_SEED,
        ..GeneralizedPnPRansac::default()
    };
    let baseline_keys = baseline_landmarks
        .iter()
        .flat_map(|landmark| landmark.observations.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut used_track_ids = baseline_landmarks
        .iter()
        .map(|landmark| landmark.track_id)
        .collect::<BTreeSet<_>>();
    let mut used_observations = baseline_keys;
    let mut working_pose_overrides = BTreeMap::<u64, Pose>::new();
    let mut recovered_landmarks = Vec::new();

    for frame_id in zero_support_frames {
        let track_ids = recovery_candidate_track_ids(frame_id, images, store);
        let candidate_track_count = track_ids.len();
        summary.candidate_tracks += candidate_track_count;
        if track_ids.is_empty() {
            continue;
        }
        summary.frames_attempted += 1;

        let mut anchors = Vec::new();
        let mut correspondences = Vec::new();
        for track_id in track_ids {
            let track = &store.tracks[track_id];
            let target_keys = track
                .observations
                .iter()
                .copied()
                .filter(|key| {
                    images
                        .get(&key.global_image_id)
                        .is_some_and(|image| image.atlas.frame_id == frame_id)
                })
                .collect::<Vec<_>>();
            let anchor_keys = track
                .observations
                .iter()
                .copied()
                .filter(|key| {
                    images.get(&key.global_image_id).is_some_and(|image| {
                        initial_supported_frames.contains(&image.atlas.frame_id)
                    })
                })
                .collect::<Vec<_>>();
            if target_keys.is_empty() || anchor_keys.len() < MIN_TRACK_OBSERVATIONS {
                summary.anchor_rejections += 1;
                continue;
            }
            let anchor_track = GlobalTrack {
                observations: anchor_keys,
            };
            let Ok(anchor_landmark) = triangulate_track(
                track_id,
                &anchor_track,
                &store.observations,
                images,
                cameras,
            ) else {
                summary.anchor_rejections += 1;
                continue;
            };
            summary.anchor_tracks += 1;
            if correspondences.len() + target_keys.len() > MAX_RECOVERY_CORRESPONDENCES {
                break;
            }
            let first_correspondence = correspondences.len();
            for key in &target_keys {
                let image = images
                    .get(&key.global_image_id)
                    .ok_or_else(|| "recovery track references unknown image".to_owned())?;
                let observation = store
                    .observations
                    .get(key)
                    .ok_or_else(|| "recovery track references unknown observation".to_owned())?;
                correspondences.push(GeneralizedCorrespondence2D3D {
                    sensor_index: image.atlas.sensor_index,
                    point2d: observation.xy,
                    point3d: anchor_landmark.position,
                    confidence: None,
                });
            }
            anchors.push(RecoveryAnchor {
                track_id,
                correspondence_indices: (first_correspondence..correspondences.len()).collect(),
            });
        }
        summary.pnp_correspondences += correspondences.len();
        if correspondences.len() < RECOVERY_MIN_PNP_INLIERS {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} candidates={} anchors={} correspondences={} status=insufficient-correspondences",
                candidate_track_count,
                anchors.len(),
                correspondences.len(),
            );
            continue;
        }
        let Some(report) = pnp.estimate(&rig, &correspondences) else {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} anchors={} correspondences={} status=pnp-failed",
                anchors.len(),
                correspondences.len(),
            );
            continue;
        };
        summary.pnp_inliers += report.inliers.len();
        if report.inliers.len() < RECOVERY_MIN_PNP_INLIERS {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} anchors={} correspondences={} pnp_inliers={} status=insufficient-inliers",
                anchors.len(),
                correspondences.len(),
                report.inliers.len(),
            );
            continue;
        }
        let inliers = report.inliers.iter().copied().collect::<BTreeSet<_>>();
        let mut candidate_pose_overrides = working_pose_overrides.clone();
        compose_recovered_frame_poses(
            manifest,
            frame_id,
            &report.pose,
            images,
            &mut candidate_pose_overrides,
        )?;
        if !existing_landmarks_non_regressed(
            baseline_landmarks,
            &store.observations,
            images,
            cameras,
            &candidate_pose_overrides,
        )? {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} pnp_inliers={} status=existing-model-regression",
                report.inliers.len(),
            );
            continue;
        }

        let anchor_count = anchors.len();
        let mut frame_landmarks = Vec::new();
        let mut frame_observations = BTreeSet::new();
        for anchor in anchors {
            if !anchor
                .correspondence_indices
                .iter()
                .any(|index| inliers.contains(index))
            {
                summary.track_rejections += 1;
                continue;
            }
            let filtered_keys = recovery_track_keys(
                &store.tracks[anchor.track_id],
                frame_id,
                &initial_supported_frames,
                images,
            );
            if filtered_keys.len() < MIN_TRACK_OBSERVATIONS {
                summary.track_rejections += 1;
                continue;
            }
            if filtered_keys
                .iter()
                .any(|key| used_observations.contains(key) || frame_observations.contains(key))
            {
                summary.track_rejections += 1;
                continue;
            }
            let filtered_track = GlobalTrack {
                observations: filtered_keys,
            };
            let Ok(landmark) = triangulate_track_with_pose_overrides(
                anchor.track_id,
                &filtered_track,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) else {
                summary.track_rejections += 1;
                continue;
            };
            if landmark.observations.len() < MIN_TRACK_OBSERVATIONS
                || used_track_ids.contains(&landmark.track_id)
            {
                summary.track_rejections += 1;
                continue;
            }
            frame_observations.extend(landmark.observations.iter().copied());
            frame_landmarks.push(landmark);
        }
        let accepted_observations = frame_landmarks
            .iter()
            .map(|landmark| landmark.observations.len())
            .sum::<usize>();
        let accepted_target_landmarks = frame_landmarks
            .iter()
            .filter(|landmark| {
                landmark.observations.iter().any(|key| {
                    images
                        .get(&key.global_image_id)
                        .is_some_and(|image| image.atlas.frame_id == frame_id)
                })
            })
            .count();
        let accepted_target_observations = frame_landmarks
            .iter()
            .flat_map(|landmark| landmark.observations.iter())
            .filter(|key| {
                images
                    .get(&key.global_image_id)
                    .is_some_and(|image| image.atlas.frame_id == frame_id)
            })
            .count();
        summary.candidate_target_landmarks += accepted_target_landmarks;
        summary.candidate_target_observations += accepted_target_observations;
        if frame_landmarks.is_empty()
            || accepted_target_landmarks < RECOVERY_MIN_PNP_INLIERS
            || accepted_target_observations < RECOVERY_MIN_PNP_INLIERS
        {
            summary.track_rejections += frame_landmarks.len();
            println!(
                "recovery frame={frame_id} pnp_inliers={} accepted_landmarks={} accepted_observations={} target_landmarks={} target_observations={} status=track-gate-rejected",
                report.inliers.len(),
                frame_landmarks.len(),
                accepted_observations,
                accepted_target_landmarks,
                accepted_target_observations,
            );
            continue;
        }
        for landmark in &frame_landmarks {
            used_track_ids.insert(landmark.track_id);
            used_observations.extend(landmark.observations.iter().copied());
        }
        summary.frames_recovered += 1;
        summary.accepted_landmarks += frame_landmarks.len();
        summary.accepted_observations += accepted_observations;
        summary.accepted_target_landmarks += accepted_target_landmarks;
        summary.accepted_target_observations += accepted_target_observations;
        println!(
            "recovery frame={frame_id} anchors={} correspondences={} pnp_inliers={} pnp_mean_px={:.6} pnp_max_px={:.6} accepted_landmarks={} accepted_observations={} target_landmarks={} target_observations={} status=accepted",
            anchor_count,
            correspondences.len(),
            report.inliers.len(),
            report.mean_reprojection_error,
            report.max_reprojection_error,
            frame_landmarks.len(),
            accepted_observations,
            accepted_target_landmarks,
            accepted_target_observations,
        );
        working_pose_overrides = candidate_pose_overrides;
        recovered_landmarks.extend(frame_landmarks);
    }

    Ok(RecoveryResult {
        pose_overrides: working_pose_overrides,
        landmarks: recovered_landmarks,
        summary,
    })
}

fn recovery_candidate_track_ids(
    frame_id: u64,
    images: &BTreeMap<u64, GlobalImage>,
    store: &TrackStore,
) -> Vec<usize> {
    let image_ids = images
        .values()
        .filter(|image| image.atlas.frame_id == frame_id)
        .map(|image| image.atlas.global_image_id)
        .collect::<Vec<_>>();
    let mut track_ids = BTreeSet::new();
    for global_image_id in image_ids {
        let first = ObservationKey {
            global_image_id,
            keypoint_index: 0,
        };
        let last = ObservationKey {
            global_image_id,
            keypoint_index: usize::MAX,
        };
        track_ids.extend(
            store
                .observations
                .range(first..=last)
                .map(|(_, state)| state.owner_track),
        );
        if track_ids.len() >= MAX_RECOVERY_TRACKS_PER_FRAME {
            break;
        }
    }
    track_ids
        .into_iter()
        .take(MAX_RECOVERY_TRACKS_PER_FRAME)
        .filter(|track_id| {
            store
                .tracks
                .get(*track_id)
                .is_some_and(|track| !track.observations.is_empty())
        })
        .collect()
}

pub(super) fn build_generalized_rig(
    manifest: &RigManifest,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<GeneralizedCameraRig, String> {
    let mut sensors = Vec::with_capacity(manifest.sensors.len());
    for (index, calibration) in &manifest.sensors {
        if *index != sensors.len() {
            return Err("rig sensor indices must be contiguous from zero".to_owned());
        }
        let camera = cameras
            .get(&calibration.camera_id)
            .ok_or_else(|| format!("recovery camera {} is unavailable", calibration.camera_id))?;
        sensors.push(RigSensor {
            camera: camera.clone(),
            sensor_from_rig: calibration.sensor_from_rig.clone(),
        });
    }
    GeneralizedCameraRig::new(sensors)
        .ok_or_else(|| "recovery rig has invalid intrinsics or extrinsics".to_owned())
}

pub(super) fn compose_recovered_frame_poses(
    manifest: &RigManifest,
    frame_id: u64,
    rig_pose: &Pose,
    images: &BTreeMap<u64, GlobalImage>,
    pose_overrides: &mut BTreeMap<u64, Pose>,
) -> Result<(), String> {
    let mut changed = 0;
    for image in images.values() {
        if image.atlas.frame_id != frame_id {
            continue;
        }
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| "recovery image references unknown sensor".to_owned())?;
        let world_to_sensor = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        pose_overrides.insert(
            image.atlas.global_image_id,
            Pose {
                world_to_camera: world_to_sensor,
            },
        );
        changed += 1;
    }
    if changed == 0 {
        return Err(format!("recovery frame {frame_id} has no atlas images"));
    }
    Ok(())
}

pub(super) fn apply_pose_overrides(
    images: &mut BTreeMap<u64, GlobalImage>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<(), String> {
    for (global_image_id, pose) in pose_overrides {
        let image = images
            .get_mut(global_image_id)
            .ok_or_else(|| "recovery pose references unknown image".to_owned())?;
        image.atlas.pose = pose.clone();
    }
    Ok(())
}

pub(super) fn recovery_track_keys(
    track: &GlobalTrack,
    target_frame: u64,
    initially_supported_frames: &BTreeSet<u64>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Vec<ObservationKey> {
    track
        .observations
        .iter()
        .copied()
        .filter(|key| {
            images.get(&key.global_image_id).is_some_and(|image| {
                image.atlas.frame_id == target_frame
                    || initially_supported_frames.contains(&image.atlas.frame_id)
            })
        })
        .collect()
}

fn existing_landmarks_non_regressed(
    landmarks: &[LandmarkOutput],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<bool, String> {
    for landmark in landmarks {
        let (mean, max) =
            evaluate_landmark_position(landmark, observations, images, cameras, pose_overrides)?;
        if mean > landmark.mean_error + 1.0e-9 || max > landmark.max_error + 1.0e-9 {
            return Ok(false);
        }
    }
    Ok(true)
}
