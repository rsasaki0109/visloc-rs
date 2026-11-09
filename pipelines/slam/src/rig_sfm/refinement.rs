//! Fixed-frame-rotation refinement of a finished reconstruction and the
//! working / public state conversions it shares with the drivers.

use super::*;

/// Refine an existing rig reconstruction while holding the supplied frame
/// rotations fixed.
///
/// `desired_rotations` is indexed by rig frame (not image).  A `Some` entry
/// replaces the world-to-rig/world-to-camera rotation of a registered frame;
/// its camera centre is retained when that replacement is installed.  `None`
/// entries and unregistered frames retain their existing rotations.  The
/// subsequent sparse rig BA includes every registered frame and fixes every
/// active rotation, while translations and landmarks remain free.  A final
/// fixed-pose structure refinement updates the landmarks without changing the
/// frame poses.
///
/// The operation is transactional: validation failures, solver failures, and
/// non-finite candidate states restore `result` byte-for-byte (as represented
/// by its clone) and return an error.
pub fn refine_rig_sfm_with_fixed_frame_rotations(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    desired_rotations: &[Option<UnitQuaternion<f64>>],
    config: &RigSfmConfig,
    result: &mut RigSfmResult,
) -> Result<(), RigSfmError> {
    let result_before = result.clone();
    match refine_rig_sfm_with_fixed_frame_rotations_impl(
        rig,
        frames,
        features,
        desired_rotations,
        config,
        result,
    ) {
        Ok(()) => Ok(()),
        Err(error) => {
            *result = result_before;
            Err(error)
        }
    }
}

fn refine_rig_sfm_with_fixed_frame_rotations_impl(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    desired_rotations: &[Option<UnitQuaternion<f64>>],
    config: &RigSfmConfig,
    result: &mut RigSfmResult,
) -> Result<(), RigSfmError> {
    validate_fixed_frame_rotation_refinement_inputs(
        rig,
        frames,
        features,
        desired_rotations,
        config,
        result,
    )?;

    let image_assignment = image_assignment(frames, features.len());
    let active_frames = result
        .frame_poses
        .iter()
        .enumerate()
        .filter_map(|(frame, pose)| pose.as_ref().map(|_| frame))
        .collect::<HashSet<_>>();
    if active_frames.is_empty() {
        return Err(RigSfmError::NoRegisteredFrames);
    }
    if !active_frames.contains(&result.seed_frame_index) {
        return Err(RigSfmError::InvalidResultSeedFrame {
            frame: result.seed_frame_index,
        });
    }

    let mut frame_poses = result.frame_poses.clone();
    let mut image_poses = result.image_poses.clone();
    for (frame, desired) in desired_rotations.iter().enumerate() {
        let (Some(frame_pose), Some(desired)) = (frame_poses[frame].as_mut(), desired.as_ref())
        else {
            continue;
        };
        let centre = frame_pose.camera_center_world();
        frame_pose.world_to_camera.rotation = *desired;
        frame_pose.world_to_camera.translation = -desired.transform_vector(&centre.coords);
    }
    regenerate_registered_image_poses(rig, frames, &frame_poses, &mut image_poses);

    let mut tracks = working_tracks_from_result(result, &image_assignment)?;
    let ba_stats = run_rig_bundle_adjustment(
        rig,
        features,
        &image_assignment,
        config,
        &active_frames,
        result.seed_frame_index,
        &config.ba_config,
        config.final_ba_min_pose_observations,
        &[],
        true,
        &mut frame_poses,
        &mut image_poses,
        &mut tracks,
    )?
    .ok_or(RigSfmError::NoBundleAdjustmentObservations)?;
    if !ba_stats.initial_cost.is_finite() || !ba_stats.final_cost.is_finite() {
        return Err(RigSfmError::NonFiniteResultState);
    }

    // `run_rig_bundle_adjustment` derives image poses after its write-back;
    // keep this explicit here because the following fixed-pose structure pass
    // and the public track conversion both consume the derived image states.
    regenerate_registered_image_poses(rig, frames, &frame_poses, &mut image_poses);
    let mut registered_image_poses = image_poses.clone();
    for (image, &(frame, _)) in image_assignment.iter().enumerate() {
        if frame_poses[frame].is_none() {
            registered_image_poses[image] = None;
        }
    }
    let structure_refined = refine_rig_structure(
        rig,
        features,
        &image_assignment,
        &registered_image_poses,
        config,
        &mut tracks,
    );
    regenerate_registered_image_poses(rig, frames, &frame_poses, &mut image_poses);

    if !working_track_state_is_finite(&tracks) {
        return Err(RigSfmError::NonFiniteResultState);
    }

    let sfm_tracks = public_tracks_from_working_tracks(
        rig,
        features,
        &image_assignment,
        &frame_poses,
        &image_poses,
        config,
        &tracks,
    );
    let (error_sum, error_count) = reprojection_error_registered(
        rig,
        &image_assignment,
        &frame_poses,
        &image_poses,
        &sfm_tracks,
    );
    let mean_reprojection_error_px = if error_count == 0 {
        0.0
    } else {
        error_sum / error_count as f64
    };
    if !mean_reprojection_error_px.is_finite()
        || !refinement_state_is_finite(&frame_poses, &image_poses, &sfm_tracks, Some(&ba_stats))
    {
        return Err(RigSfmError::NonFiniteResultState);
    }

    result.frame_poses = frame_poses;
    result.image_poses = image_poses;
    result.tracks = sfm_tracks;
    result.registered_frames = result
        .frame_poses
        .iter()
        .filter(|pose| pose.is_some())
        .count();
    result.registered_images = result
        .frame_poses
        .iter()
        .enumerate()
        .filter(|(frame, pose)| {
            pose.is_some()
                && frames[*frame]
                    .images
                    .iter()
                    .all(|image| result.image_poses[image.image_index].is_some())
        })
        .map(|frame| frames[frame.0].images.len())
        .sum();
    result.mean_reprojection_error_px = mean_reprojection_error_px;
    result.bundle_adjustment = Some(ba_stats);
    result.work.structure_refined_tracks = result
        .work
        .structure_refined_tracks
        .saturating_add(structure_refined);
    Ok(())
}

fn validate_fixed_frame_rotation_refinement_inputs(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    desired_rotations: &[Option<UnitQuaternion<f64>>],
    config: &RigSfmConfig,
    result: &RigSfmResult,
) -> Result<(), RigSfmError> {
    if rig.sensors().len() < 2 {
        return Err(RigSfmError::TooFewSensors);
    }
    if frames.is_empty() {
        return Err(RigSfmError::NoFrames);
    }
    if desired_rotations.len() != frames.len() {
        return Err(RigSfmError::InvalidFixedFrameRotationCount {
            expected: frames.len(),
            actual: desired_rotations.len(),
        });
    }
    if result.frame_poses.len() != frames.len() {
        return Err(RigSfmError::InvalidResultFramePoseCount {
            expected: frames.len(),
            actual: result.frame_poses.len(),
        });
    }
    if result.image_poses.len() != features.len() {
        return Err(RigSfmError::InvalidResultImagePoseCount {
            expected: features.len(),
            actual: result.image_poses.len(),
        });
    }
    // Reuse the mapper's complete rig/frame/feature validation.  There are no
    // pairwise inputs in this refinement API, so an empty pair stream is
    // sufficient here.  The mapper-only deferred prefix is deliberately
    // disabled for this validation copy: with no pair stream to partition, a
    // perfectly valid mapper config would otherwise fail merely because its
    // prefix is nonzero.
    let validation_config = RigSfmConfig {
        deferred_registration_pair_prefix: None,
        ..*config
    };
    validate_inputs(rig, frames, features, &[], &validation_config)?;

    if features.iter().any(|feature_set| {
        feature_set
            .keypoints
            .iter()
            .any(|point| !point.coords.iter().all(|value| value.is_finite()))
    }) {
        return Err(RigSfmError::NonFiniteResultState);
    }

    for (frame, desired) in desired_rotations.iter().enumerate() {
        let Some(desired) = desired else {
            continue;
        };
        let norm_squared = desired.coords.norm_squared();
        if !desired.coords.iter().all(|value| value.is_finite()) || !norm_squared.is_finite() {
            return Err(RigSfmError::NonFiniteFixedFrameRotation { frame });
        }
        if norm_squared <= f64::EPSILON {
            return Err(RigSfmError::InvalidFixedFrameRotation { frame });
        }
    }
    if !result.mean_reprojection_error_px.is_finite()
        || result
            .bundle_adjustment
            .as_ref()
            .is_some_and(|stats| !stats.initial_cost.is_finite() || !stats.final_cost.is_finite())
        || result.frame_poses.iter().any(|pose| {
            pose.as_ref()
                .is_some_and(|pose| !pose_state_is_finite(pose))
        })
        || result.image_poses.iter().any(|pose| {
            pose.as_ref()
                .is_some_and(|pose| !pose_state_is_finite(pose))
        })
    {
        return Err(RigSfmError::NonFiniteResultState);
    }
    for (track, sfm_track) in result.tracks.iter().enumerate() {
        if !sfm_track
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(RigSfmError::NonFiniteResultTrack { track });
        }
        for &(image, keypoint, pixel) in &sfm_track.observations {
            if image >= features.len() || keypoint >= features[image].keypoints.len() {
                return Err(RigSfmError::InvalidResultTrackObservation {
                    track,
                    image,
                    keypoint,
                });
            }
            if !pixel.coords.iter().all(|value| value.is_finite()) {
                return Err(RigSfmError::NonFiniteResultTrack { track });
            }
        }
    }
    Ok(())
}

fn working_tracks_from_result(
    result: &RigSfmResult,
    image_assignment: &[(usize, usize)],
) -> Result<Vec<WorkingTrack>, RigSfmError> {
    result
        .tracks
        .iter()
        .enumerate()
        .map(|(track, sfm_track)| {
            let observations = sfm_track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
                .collect::<Vec<_>>();
            if observations
                .iter()
                .any(|&(image, _)| image >= image_assignment.len())
            {
                return Err(RigSfmError::InvalidResultTrackObservation {
                    track,
                    image: observations
                        .iter()
                        .find(|&&(image, _)| image >= image_assignment.len())
                        .map_or(usize::MAX, |&(image, _)| image),
                    keypoint: usize::MAX,
                });
            }
            Ok(WorkingTrack {
                metric_anchored: track_is_metric_anchored(&observations, image_assignment),
                observations,
                position: Some(sfm_track.position),
            })
        })
        .collect()
}

fn regenerate_registered_image_poses(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    frame_poses: &[Option<Pose>],
    image_poses: &mut [Option<Pose>],
) {
    for (frame, frame_pose) in frame_poses.iter().enumerate() {
        let Some(frame_pose) = frame_pose.as_ref() else {
            continue;
        };
        install_image_poses(rig, &frames[frame], frame_pose, image_poses);
    }
}

pub(super) fn public_tracks_from_working_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &[WorkingTrack],
) -> Vec<SfmTrack> {
    tracks
        .iter()
        .filter_map(|track| {
            let position = track.position?;
            let observations = track
                .observations
                .iter()
                .filter(|&&(image, keypoint)| {
                    let frame = image_assignment[image].0;
                    if frame_poses[frame].is_none() || image_poses[image].is_none() {
                        return false;
                    }
                    let sensor = &rig.sensors()[image_assignment[image].1];
                    sensor
                        .camera
                        .project(
                            &image_poses[image]
                                .as_ref()
                                .unwrap()
                                .transform_world_point(&position),
                        )
                        .is_some_and(|projected| {
                            (projected - features[image].keypoints[keypoint]).norm()
                                <= config.max_reprojection_error_px
                        })
                })
                .map(|&(image, keypoint)| (image, keypoint, features[image].keypoints[keypoint]))
                .collect::<Vec<_>>();
            (observations.len() >= 2).then_some(SfmTrack {
                position,
                observations,
            })
        })
        .collect()
}

pub(super) fn reprojection_error_registered(
    rig: &GeneralizedCameraRig,
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for track in tracks {
        for (image, _, pixel) in &track.observations {
            let frame = image_assignment[*image].0;
            if frame_poses[frame].is_none() {
                continue;
            }
            let Some(pose) = image_poses[*image].as_ref() else {
                continue;
            };
            let sensor = &rig.sensors()[image_assignment[*image].1];
            if let Some(projected) = sensor
                .camera
                .project(&pose.transform_world_point(&track.position))
            {
                let error = (projected - pixel).norm();
                if error.is_finite() {
                    sum += error;
                    count += 1;
                }
            }
        }
    }
    (sum, count)
}

fn refinement_state_is_finite(
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
    ba_stats: Option<&RigBaStats>,
) -> bool {
    frame_poses
        .iter()
        .all(|pose| pose.as_ref().is_none_or(pose_state_is_finite))
        && image_poses
            .iter()
            .all(|pose| pose.as_ref().is_none_or(pose_state_is_finite))
        && tracks.iter().all(|track| {
            track.position.coords.iter().all(|value| value.is_finite())
                && track
                    .observations
                    .iter()
                    .all(|(_, _, pixel)| pixel.coords.iter().all(|value| value.is_finite()))
        })
        && ba_stats
            .is_none_or(|stats| stats.initial_cost.is_finite() && stats.final_cost.is_finite())
}

fn pose_state_is_finite(pose: &Pose) -> bool {
    let rotation_norm_squared = pose.world_to_camera.rotation.coords.norm_squared();
    pose.world_to_camera
        .rotation
        .coords
        .iter()
        .all(|value| value.is_finite())
        && rotation_norm_squared.is_finite()
        && rotation_norm_squared > f64::EPSILON
        && pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
}

fn working_track_state_is_finite(tracks: &[WorkingTrack]) -> bool {
    tracks.iter().all(|track| {
        track
            .position
            .is_none_or(|point| point.coords.iter().all(|value| value.is_finite()))
    })
}
