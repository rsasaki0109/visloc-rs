//! Incremental structure-from-motion for a calibrated, synchronized camera rig.
//!
//! Each input frame owns one or more images with fixed `sensor <- rig`
//! extrinsics.  A frame is registered from all of its 2D-3D observations at
//! once with generalized PnP; image poses are derived from that single body
//! pose.  This preserves the physical stereo baseline during initialization
//! and avoids treating synchronized sensors as unrelated monocular cameras.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use nalgebra::{Matrix3, Point2, Point3, SMatrix, Vector2, Vector3};
use thiserror::Error;
use visloc_core::geometry::Pose;
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{
    GeneralizedCameraRig, GeneralizedCorrespondence2D3D, GeneralizedPnPRansac,
};

use crate::bundle::{BaConfig, BaRigObservation, BundleAdjustment};
use crate::incremental_sfm::{build_basic_tracks, PairwiseMatches, SfmTrack};
use crate::{LinearSolver, RobustKernel};

/// One image and its calibrated sensor slot within a synchronized rig frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RigFrameImage {
    pub image_index: usize,
    pub sensor_index: usize,
}

/// Images captured at one timestamp and governed by one `world -> rig` pose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigFrame {
    pub images: Vec<RigFrameImage>,
}

/// Conservative controls for generalized-rig incremental reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigSfmConfig {
    pub min_track_length: usize,
    pub min_pnp_inliers: usize,
    pub min_triangulation_angle_deg: f64,
    pub max_reprojection_error_px: f64,
    pub pnp_max_iterations: usize,
    pub ransac_seed: u64,
    pub final_bundle_adjustment: bool,
    pub ba_config: BaConfig,
    pub local_ba_every: usize,
    pub local_ba_window_size: usize,
    pub local_ba_iterations: usize,
    pub final_ba_passes: usize,
    pub final_ba_window_size: usize,
    pub final_ba_fix_window_ends: bool,
    /// Fixed-pose, per-landmark Gauss-Newton passes after registration.  This
    /// is linear in retained observations and never forms a global BA matrix.
    pub structure_refinement_iterations: usize,
}

impl Default for RigSfmConfig {
    fn default() -> Self {
        Self {
            min_track_length: 2,
            min_pnp_inliers: 8,
            min_triangulation_angle_deg: 1.0,
            max_reprojection_error_px: 4.0,
            pnp_max_iterations: 512,
            ransac_seed: 7,
            final_bundle_adjustment: true,
            ba_config: BaConfig {
                linear_solver: LinearSolver::Sparse,
                robust_kernel: RobustKernel::Huber { delta: 6.0 },
                parallel: true,
                ..BaConfig::default()
            },
            local_ba_every: 10,
            local_ba_window_size: 40,
            local_ba_iterations: 8,
            final_ba_passes: 2,
            final_ba_window_size: 60,
            final_ba_fix_window_ends: true,
            structure_refinement_iterations: 5,
        }
    }
}

/// Output poses retain both the physical frame state and derived image states.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSfmResult {
    pub frame_poses: Vec<Option<Pose>>,
    pub image_poses: Vec<Option<Pose>>,
    pub tracks: Vec<SfmTrack>,
    pub registered_frames: usize,
    pub registered_images: usize,
    pub mean_reprojection_error_px: f64,
    pub seed_frame_index: usize,
    pub work: RigSfmWorkStats,
    pub bundle_adjustment: Option<RigBaStats>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigBaStats {
    pub observations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Counters used to prove that frontier growth stays proportional to sparse
/// observation support rather than rescanning the frame/track Cartesian
/// product.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RigSfmWorkStats {
    pub triangulation_attempts: usize,
    pub correspondence_cache_insertions: usize,
    pub pnp_attempts: usize,
    pub local_ba_runs: usize,
    pub structure_refined_tracks: usize,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum RigSfmError {
    #[error("rig reconstruction requires at least two calibrated sensors")]
    TooFewSensors,
    #[error("rig reconstruction contains no frames")]
    NoFrames,
    #[error("frame {frame} contains no images")]
    EmptyFrame { frame: usize },
    #[error("image {image} is outside the feature range 0..{feature_count}")]
    ImageIndex { image: usize, feature_count: usize },
    #[error("sensor {sensor} is outside the calibrated range 0..{sensor_count}")]
    SensorIndex { sensor: usize, sensor_count: usize },
    #[error("image {image} occurs in more than one frame")]
    DuplicateImage { image: usize },
    #[error("sensor {sensor} occurs twice in frame {frame}")]
    DuplicateSensor { frame: usize, sensor: usize },
    #[error("image {image} is not assigned to a rig frame")]
    UnassignedImage { image: usize },
    #[error("invalid pair ({image_i}, {image_j}) for {feature_count} images")]
    PairImageIndex {
        image_i: usize,
        image_j: usize,
        feature_count: usize,
    },
    #[error("pair ({image_i}, {image_j}) references an invalid keypoint")]
    PairKeypointIndex { image_i: usize, image_j: usize },
    #[error("no frame has enough multi-sensor tracks for metric initialization")]
    NoMetricSeed,
    #[error("metric seed frame {frame} triangulated fewer than {required} landmarks")]
    InsufficientSeedStructure {
        frame: usize,
        required: usize,
        triangulated: usize,
    },
    #[error("rig bundle adjustment failed: {0}")]
    BundleAdjustment(String),
}

#[derive(Debug, Clone)]
struct WorkingTrack {
    observations: Vec<(usize, usize)>,
    position: Option<Point3<f64>>,
}

#[derive(Debug, Clone, Copy)]
struct CachedRigCorrespondence {
    sensor_index: usize,
    point2d: Point2<f64>,
    track_index: usize,
}

/// Reconstruct synchronized rig frames with one generalized body pose per
/// timestamp.  The first pose fixes the world gauge; the calibrated inter-
/// sensor baseline fixes metric scale.
pub fn incremental_rig_sfm(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &RigSfmConfig,
) -> Result<RigSfmResult, RigSfmError> {
    if rig.sensors().len() < 2 {
        return Err(RigSfmError::TooFewSensors);
    }
    if frames.is_empty() {
        return Err(RigSfmError::NoFrames);
    }
    validate_inputs(rig, frames, features, pairwise)?;

    let image_assignment = image_assignment(frames, features.len());
    let raw_tracks = build_basic_tracks(features.len(), pairwise, config.min_track_length.max(2));
    let mut tracks = raw_tracks
        .into_iter()
        .map(|observations| WorkingTrack {
            observations,
            position: None,
        })
        .collect::<Vec<_>>();
    let image_tracks = image_track_index(features.len(), &tracks);

    let seed_frame_index =
        choose_metric_seed(frames, &tracks, &image_assignment).ok_or(RigSfmError::NoMetricSeed)?;
    let mut frame_poses = vec![None; frames.len()];
    frame_poses[seed_frame_index] = Some(Pose::identity());
    let mut registration_order = vec![seed_frame_index];

    let mut image_poses = vec![None; features.len()];
    install_image_poses(
        rig,
        &frames[seed_frame_index],
        frame_poses[seed_frame_index].as_ref().unwrap(),
        &mut image_poses,
    );
    let seed_frontier = frames[seed_frame_index]
        .images
        .iter()
        .flat_map(|image| {
            image_tracks[image.image_index]
                .iter()
                .map(|(_, track)| *track)
        })
        .collect::<HashSet<_>>();
    let seed_triangulation = triangulate_frontier(
        rig,
        features,
        &image_assignment,
        &image_poses,
        config,
        &mut tracks,
        seed_frontier,
    );
    let seed_landmarks = tracks
        .iter()
        .filter(|track| track.position.is_some())
        .count();
    if seed_landmarks < config.min_pnp_inliers.max(6) {
        return Err(RigSfmError::InsufficientSeedStructure {
            frame: seed_frame_index,
            required: config.min_pnp_inliers.max(6),
            triangulated: seed_landmarks,
        });
    }

    let pnp = GeneralizedPnPRansac {
        iterations: config.pnp_max_iterations,
        reprojection_threshold: config.max_reprojection_error_px,
        seed: config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    };

    // Each landmark enters each observing frame's cache exactly once. Heap
    // versions make stale support counts cheap to discard, avoiding the
    // all-unregistered-frame rescan that becomes quadratic at 10k scale.
    let mut frame_correspondences: Vec<Vec<CachedRigCorrespondence>> =
        vec![Vec::new(); frames.len()];
    let mut frame_versions = vec![0usize; frames.len()];
    let mut attempted_versions = vec![None; frames.len()];
    let mut candidate_heap = BinaryHeap::new();
    let mut work = RigSfmWorkStats {
        triangulation_attempts: seed_triangulation.attempts,
        ..RigSfmWorkStats::default()
    };
    work.correspondence_cache_insertions += append_landmark_correspondences(
        &seed_triangulation.landmarks,
        &tracks,
        features,
        &image_assignment,
        &mut frame_correspondences,
        &mut frame_versions,
        &mut candidate_heap,
    );

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
        let correspondences = frame_correspondences[frame]
            .iter()
            .filter_map(|cached| {
                Some(GeneralizedCorrespondence2D3D {
                    sensor_index: cached.sensor_index,
                    point2d: cached.point2d,
                    point3d: tracks[cached.track_index].position?,
                    confidence: None,
                })
            })
            .collect::<Vec<_>>();
        let distinct_sensors = correspondences
            .iter()
            .map(|correspondence| correspondence.sensor_index)
            .collect::<HashSet<_>>()
            .len();
        if distinct_sensors < 2 {
            continue;
        }
        let Some(report) = pnp.estimate(rig, &correspondences) else {
            continue;
        };
        if report.inliers.len() < config.min_pnp_inliers.max(6) {
            continue;
        }
        frame_poses[frame] = Some(report.pose);
        registration_order.push(frame);
        install_image_poses(
            rig,
            &frames[frame],
            frame_poses[frame].as_ref().unwrap(),
            &mut image_poses,
        );
        let frontier = frames[frame]
            .images
            .iter()
            .flat_map(|image| {
                image_tracks[image.image_index]
                    .iter()
                    .map(|(_, track)| *track)
            })
            .collect::<HashSet<_>>();
        let triangulation = triangulate_frontier(
            rig,
            features,
            &image_assignment,
            &image_poses,
            config,
            &mut tracks,
            frontier,
        );
        work.triangulation_attempts += triangulation.attempts;
        work.correspondence_cache_insertions += append_landmark_correspondences(
            &triangulation.landmarks,
            &tracks,
            features,
            &image_assignment,
            &mut frame_correspondences,
            &mut frame_versions,
            &mut candidate_heap,
        );
        if config.local_ba_every > 0
            && config.local_ba_window_size >= 2
            && registration_order.len() % config.local_ba_every == 0
        {
            let start = registration_order
                .len()
                .saturating_sub(config.local_ba_window_size);
            let active_frames = registration_order[start..]
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            let anchor = registration_order[start];
            let local_ba_config = BaConfig {
                max_iterations: config.local_ba_iterations,
                ..config.ba_config
            };
            if run_rig_bundle_adjustment(
                rig,
                features,
                &image_assignment,
                config,
                &active_frames,
                anchor,
                &local_ba_config,
                &[],
                &mut frame_poses,
                &mut image_poses,
                &mut tracks,
            )?
            .is_some()
            {
                work.local_ba_runs += 1;
            }
        }
    }

    let bundle_adjustment = if config.final_bundle_adjustment && config.final_ba_passes > 0 {
        run_windowed_final_ba(
            rig,
            features,
            &image_assignment,
            config,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
        )?
    } else {
        None
    };

    work.structure_refined_tracks = refine_rig_structure(
        rig,
        features,
        &image_assignment,
        &image_poses,
        config,
        &mut tracks,
    );

    let sfm_tracks = tracks
        .into_iter()
        .filter_map(|track| {
            let position = track.position?;
            let observations = track
                .observations
                .into_iter()
                .filter(|(image, keypoint)| {
                    let Some(pose) = image_poses[*image].as_ref() else {
                        return false;
                    };
                    let sensor = &rig.sensors()[image_assignment[*image].1];
                    sensor
                        .camera
                        .project(&pose.transform_world_point(&position))
                        .is_some_and(|projected| {
                            (projected - features[*image].keypoints[*keypoint]).norm()
                                <= config.max_reprojection_error_px
                        })
                })
                .map(|(image, keypoint)| (image, keypoint, features[image].keypoints[keypoint]))
                .collect::<Vec<_>>();
            (observations.len() >= 2).then_some(SfmTrack {
                position,
                observations,
            })
        })
        .collect::<Vec<_>>();
    let (error_sum, error_count) =
        reprojection_error(rig, &image_assignment, &image_poses, &sfm_tracks);

    Ok(RigSfmResult {
        registered_frames: frame_poses.iter().filter(|pose| pose.is_some()).count(),
        registered_images: image_poses.iter().filter(|pose| pose.is_some()).count(),
        mean_reprojection_error_px: if error_count == 0 {
            0.0
        } else {
            error_sum / error_count as f64
        },
        frame_poses,
        image_poses,
        tracks: sfm_tracks,
        seed_frame_index,
        work,
        bundle_adjustment,
    })
}

fn refine_rig_structure(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
) -> usize {
    if config.structure_refinement_iterations == 0 {
        return 0;
    }
    tracks
        .iter_mut()
        .filter_map(|track| {
            let mut point = track.position?;
            let initial = point;
            let mut accepted = false;
            for _ in 0..config.structure_refinement_iterations {
                let mut hessian = Matrix3::zeros();
                let mut gradient = Vector3::zeros();
                let mut observations = 0usize;
                let Some(current_cost) = rig_point_cost(
                    rig,
                    features,
                    image_assignment,
                    image_poses,
                    track,
                    &point,
                    config.max_reprojection_error_px,
                ) else {
                    break;
                };
                let step = 1.0e-6 * (1.0 + point.coords.norm());
                for &(image, keypoint) in &track.observations {
                    let Some(pose) = image_poses[image].as_ref() else {
                        continue;
                    };
                    let sensor = &rig.sensors()[image_assignment[image].1];
                    let measured = features[image].keypoints[keypoint];
                    let Some(projected) =
                        sensor.camera.project(&pose.transform_world_point(&point))
                    else {
                        continue;
                    };
                    let residual = projected - measured;
                    let norm = residual.norm();
                    if !norm.is_finite() || norm > 2.0 * config.max_reprojection_error_px {
                        continue;
                    }
                    let mut jacobian = SMatrix::<f64, 2, 3>::zeros();
                    let mut valid = true;
                    for axis in 0..3 {
                        let mut plus = point;
                        let mut minus = point;
                        plus[axis] += step;
                        minus[axis] -= step;
                        let projections = sensor
                            .camera
                            .project(&pose.transform_world_point(&plus))
                            .zip(sensor.camera.project(&pose.transform_world_point(&minus)));
                        let Some((plus_pixel, minus_pixel)) = projections else {
                            valid = false;
                            break;
                        };
                        jacobian.set_column(axis, &((plus_pixel - minus_pixel) / (2.0 * step)));
                    }
                    if !valid || !jacobian.iter().all(|value| value.is_finite()) {
                        continue;
                    }
                    let weight = huber_weight(norm, config.max_reprojection_error_px);
                    hessian += jacobian.transpose() * jacobian * weight;
                    gradient +=
                        jacobian.transpose() * Vector2::new(residual.x, residual.y) * weight;
                    observations += 1;
                }
                if observations < 2 {
                    break;
                }
                let damping = 1.0e-8_f64.max(1.0e-6 * hessian.diagonal().amax());
                hessian += Matrix3::identity() * damping;
                let Some(delta) = hessian.lu().solve(&(-gradient)) else {
                    break;
                };
                if !delta.iter().all(|value| value.is_finite()) {
                    break;
                }
                let candidate = point + delta;
                let Some(candidate_cost) = rig_point_cost(
                    rig,
                    features,
                    image_assignment,
                    image_poses,
                    track,
                    &candidate,
                    config.max_reprojection_error_px,
                ) else {
                    break;
                };
                if candidate_cost + 1.0e-12 >= current_cost {
                    break;
                }
                point = candidate;
                accepted = true;
                if delta.norm() <= 1.0e-8 * (1.0 + point.coords.norm()) {
                    break;
                }
            }
            if accepted && (point - initial).norm() > 0.0 {
                track.position = Some(point);
                Some(())
            } else {
                None
            }
        })
        .count()
}

fn rig_point_cost(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    track: &WorkingTrack,
    point: &Point3<f64>,
    huber_delta: f64,
) -> Option<f64> {
    let mut cost = 0.0;
    let mut count = 0usize;
    for &(image, keypoint) in &track.observations {
        let Some(pose) = image_poses[image].as_ref() else {
            continue;
        };
        let sensor = &rig.sensors()[image_assignment[image].1];
        let Some(projected) = sensor.camera.project(&pose.transform_world_point(point)) else {
            continue;
        };
        let norm = (projected - features[image].keypoints[keypoint]).norm();
        if !norm.is_finite() || norm > 2.0 * huber_delta {
            continue;
        }
        cost += if norm <= huber_delta {
            0.5 * norm * norm
        } else {
            huber_delta * (norm - 0.5 * huber_delta)
        };
        count += 1;
    }
    (count >= 2).then_some(cost)
}

fn huber_weight(norm: f64, delta: f64) -> f64 {
    if norm <= delta || norm <= f64::EPSILON {
        1.0
    } else {
        delta / norm
    }
}

#[allow(clippy::too_many_arguments)]
fn run_windowed_final_ba(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
    tracks: &mut [WorkingTrack],
) -> Result<Option<RigBaStats>, RigSfmError> {
    let window = config.final_ba_window_size.max(2);
    let stride = (window / 2).max(1);
    let mut starts = (0..frame_poses.len()).step_by(stride).collect::<Vec<_>>();
    if let Some(tail_start) = frame_poses.len().checked_sub(window) {
        starts.push(tail_start);
    }
    starts.sort_unstable();
    starts.dedup();
    let ba_config = BaConfig {
        max_iterations: config.local_ba_iterations,
        ..config.ba_config
    };
    let mut aggregate: Option<RigBaStats> = None;
    for pass in 0..config.final_ba_passes {
        starts.sort_unstable();
        if pass % 2 == 1 {
            starts.reverse();
        }
        for &start in &starts {
            let end = (start + window).min(frame_poses.len());
            let active_frames = (start..end)
                .filter(|frame| frame_poses[*frame].is_some())
                .collect::<HashSet<_>>();
            if active_frames.len() < 2 {
                continue;
            }
            let anchor = if pass % 2 == 0 {
                *active_frames.iter().min().unwrap()
            } else {
                *active_frames.iter().max().unwrap()
            };
            let other_end = if pass % 2 == 0 {
                *active_frames.iter().max().unwrap()
            } else {
                *active_frames.iter().min().unwrap()
            };
            let extra_fixed = config
                .final_ba_fix_window_ends
                .then_some(other_end)
                .into_iter()
                .collect::<Vec<_>>();
            if let Some(stats) = run_rig_bundle_adjustment(
                rig,
                features,
                image_assignment,
                config,
                &active_frames,
                anchor,
                &ba_config,
                &extra_fixed,
                frame_poses,
                image_poses,
                tracks,
            )? {
                let current = aggregate.get_or_insert(RigBaStats {
                    observations: 0,
                    initial_cost: 0.0,
                    final_cost: 0.0,
                    iterations: 0,
                    converged: true,
                });
                current.observations += stats.observations;
                current.initial_cost += stats.initial_cost;
                current.final_cost += stats.final_cost;
                current.iterations += stats.iterations;
                current.converged &= stats.converged;
            }
        }
    }
    Ok(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn run_rig_bundle_adjustment(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    active_frames: &HashSet<usize>,
    anchor_frame_index: usize,
    ba_config: &BaConfig,
    extra_fixed_frames: &[usize],
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
    tracks: &mut [WorkingTrack],
) -> Result<Option<RigBaStats>, RigSfmError> {
    let mut problem = BundleAdjustment::new(rig.sensors()[0].camera.clone());
    for (frame, pose) in frame_poses.iter().enumerate() {
        if !active_frames.contains(&frame) {
            continue;
        }
        let Some(pose) = pose else {
            continue;
        };
        problem.add_pose(frame as u64, pose.clone());
    }
    problem.fix_pose(anchor_frame_index as u64);
    for &frame in extra_fixed_frames {
        problem.fix_pose(frame as u64);
    }

    let mut visual_observations = 0usize;
    for (track_index, track) in tracks.iter().enumerate() {
        let Some(position) = track.position else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, keypoint) in &track.observations {
            let Some(pose) = image_poses[image].as_ref() else {
                continue;
            };
            let (frame, sensor_index) = image_assignment[image];
            if !active_frames.contains(&frame) {
                continue;
            }
            let sensor = &rig.sensors()[sensor_index];
            let pixel = features[image].keypoints[keypoint];
            let usable = sensor
                .camera
                .project(&pose.transform_world_point(&position))
                .is_some_and(|projected| {
                    (projected - pixel).norm() <= 2.0 * config.max_reprojection_error_px
                });
            if usable {
                observations.push(BaRigObservation {
                    keyframe_id: frame as u64,
                    landmark_id: track_index as u64,
                    xy: pixel,
                    camera: sensor.camera.clone(),
                    sensor_from_rig: sensor.sensor_from_rig.clone(),
                });
            }
        }
        if observations.len() < 2 {
            continue;
        }
        problem.add_landmark(track_index as u64, position);
        visual_observations += observations.len();
        for observation in observations {
            problem.add_rig_observation(observation);
        }
    }
    if visual_observations == 0 {
        return Ok(None);
    }
    let result = problem
        .optimize(ba_config)
        .map_err(|error| RigSfmError::BundleAdjustment(error.to_string()))?;
    for (frame, pose) in frame_poses.iter_mut().enumerate() {
        let Some(reference_pose) = problem.poses.get(&(frame as u64)) else {
            continue;
        };
        *pose = Some(reference_pose.clone());
    }
    for (track_index, track) in tracks.iter_mut().enumerate() {
        if let Some(position) = problem.landmarks.get(&(track_index as u64)) {
            track.position = Some(*position);
        }
    }
    for (image, pose) in image_poses.iter_mut().enumerate() {
        let (frame, sensor_index) = image_assignment[image];
        let Some(frame_pose) = frame_poses[frame].as_ref() else {
            continue;
        };
        *pose = Some(Pose {
            world_to_camera: rig.sensors()[sensor_index]
                .sensor_from_rig
                .compose(&frame_pose.world_to_camera),
        });
    }
    Ok(Some(RigBaStats {
        observations: visual_observations,
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        iterations: result.iterations.len(),
        converged: result.converged,
    }))
}

fn validate_inputs(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
) -> Result<(), RigSfmError> {
    let mut assigned = vec![false; features.len()];
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.images.is_empty() {
            return Err(RigSfmError::EmptyFrame { frame: frame_index });
        }
        let mut sensors = HashSet::new();
        for image in &frame.images {
            if image.image_index >= features.len() {
                return Err(RigSfmError::ImageIndex {
                    image: image.image_index,
                    feature_count: features.len(),
                });
            }
            if image.sensor_index >= rig.sensors().len() {
                return Err(RigSfmError::SensorIndex {
                    sensor: image.sensor_index,
                    sensor_count: rig.sensors().len(),
                });
            }
            if assigned[image.image_index] {
                return Err(RigSfmError::DuplicateImage {
                    image: image.image_index,
                });
            }
            assigned[image.image_index] = true;
            if !sensors.insert(image.sensor_index) {
                return Err(RigSfmError::DuplicateSensor {
                    frame: frame_index,
                    sensor: image.sensor_index,
                });
            }
        }
    }
    if let Some(image) = assigned.iter().position(|assigned| !assigned) {
        return Err(RigSfmError::UnassignedImage { image });
    }
    for pair in pairwise {
        if pair.image_i >= features.len() || pair.image_j >= features.len() {
            return Err(RigSfmError::PairImageIndex {
                image_i: pair.image_i,
                image_j: pair.image_j,
                feature_count: features.len(),
            });
        }
        if pair.matches.iter().any(|(left, right)| {
            *left >= features[pair.image_i].len() || *right >= features[pair.image_j].len()
        }) {
            return Err(RigSfmError::PairKeypointIndex {
                image_i: pair.image_i,
                image_j: pair.image_j,
            });
        }
    }
    Ok(())
}

fn image_assignment(frames: &[RigFrame], image_count: usize) -> Vec<(usize, usize)> {
    let mut assignment = vec![(usize::MAX, usize::MAX); image_count];
    for (frame_index, frame) in frames.iter().enumerate() {
        for image in &frame.images {
            assignment[image.image_index] = (frame_index, image.sensor_index);
        }
    }
    assignment
}

fn image_track_index(image_count: usize, tracks: &[WorkingTrack]) -> Vec<Vec<(usize, usize)>> {
    let mut by_image = vec![Vec::new(); image_count];
    for (track_index, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in &track.observations {
            by_image[image].push((keypoint, track_index));
        }
    }
    by_image
}

fn choose_metric_seed(
    frames: &[RigFrame],
    tracks: &[WorkingTrack],
    image_assignment: &[(usize, usize)],
) -> Option<usize> {
    (0..frames.len())
        .map(|frame| {
            let supported = tracks
                .iter()
                .filter(|track| {
                    track
                        .observations
                        .iter()
                        .filter(|(image, _)| image_assignment[*image].0 == frame)
                        .map(|(image, _)| image_assignment[*image].1)
                        .collect::<HashSet<_>>()
                        .len()
                        >= 2
                })
                .count();
            (frame, supported)
        })
        .filter(|(_, supported)| *supported >= 6)
        .max_by_key(|(frame, supported)| (*supported, std::cmp::Reverse(*frame)))
        .map(|(frame, _)| frame)
}

fn install_image_poses(
    rig: &GeneralizedCameraRig,
    frame: &RigFrame,
    frame_pose: &Pose,
    image_poses: &mut [Option<Pose>],
) {
    for image in &frame.images {
        let sensor = &rig.sensors()[image.sensor_index];
        image_poses[image.image_index] = Some(Pose {
            world_to_camera: sensor.sensor_from_rig.compose(&frame_pose.world_to_camera),
        });
    }
}

fn append_landmark_correspondences(
    landmark_indices: &[usize],
    tracks: &[WorkingTrack],
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    frame_correspondences: &mut [Vec<CachedRigCorrespondence>],
    frame_versions: &mut [usize],
    heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> usize {
    let mut changed_frames = HashSet::new();
    let mut insertions = 0;
    for &track_index in landmark_indices {
        let track = &tracks[track_index];
        if track.position.is_none() {
            continue;
        }
        for &(image, keypoint) in &track.observations {
            let (frame, sensor) = image_assignment[image];
            frame_correspondences[frame].push(CachedRigCorrespondence {
                sensor_index: sensor,
                point2d: features[image].keypoints[keypoint],
                track_index,
            });
            insertions += 1;
            changed_frames.insert(frame);
        }
    }
    for frame in changed_frames {
        frame_versions[frame] += 1;
        heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    insertions
}

#[derive(Debug, Default)]
struct TriangulationUpdate {
    landmarks: Vec<usize>,
    attempts: usize,
}

fn triangulate_frontier(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
    frontier: HashSet<usize>,
) -> TriangulationUpdate {
    let mut update = TriangulationUpdate::default();
    // Hash iteration order is process-randomized.  The order in which newly
    // triangulated landmarks enter each frame cache changes the indexed
    // RANSAC samples, so canonicalize it before any numeric work.
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    frontier.sort_unstable();
    for track_index in frontier {
        let track = &mut tracks[track_index];
        if track.position.is_some() {
            continue;
        }
        update.attempts += 1;
        let rays = track
            .observations
            .iter()
            .filter_map(|(image, keypoint)| {
                let pose = image_poses[*image].as_ref()?;
                let sensor = &rig.sensors()[image_assignment[*image].1];
                let normalized = sensor
                    .camera
                    .normalize_pixel(&features[*image].keypoints[*keypoint])?;
                let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
                let direction_world = pose
                    .camera_to_world()
                    .rotation
                    .transform_vector(&bearing_camera)
                    .normalize();
                Some((
                    *image,
                    *keypoint,
                    pose.camera_center_world(),
                    direction_world,
                ))
            })
            .collect::<Vec<_>>();
        let mut best = None;
        for left in 0..rays.len() {
            for right in (left + 1)..rays.len() {
                let cosine = rays[left].3.dot(&rays[right].3).clamp(-1.0, 1.0);
                let angle = cosine.acos();
                if best
                    .as_ref()
                    .is_none_or(|(_, _, best_angle): &(usize, usize, f64)| angle > *best_angle)
                {
                    best = Some((left, right, angle));
                }
            }
        }
        let Some((left, right, angle)) = best else {
            continue;
        };
        if angle.to_degrees() < config.min_triangulation_angle_deg {
            continue;
        }
        let Some(point) =
            closest_ray_midpoint(&rays[left].2, &rays[left].3, &rays[right].2, &rays[right].3)
        else {
            continue;
        };
        if track.observations.iter().all(|(image, keypoint)| {
            let Some(pose) = image_poses[*image].as_ref() else {
                return true;
            };
            let sensor = &rig.sensors()[image_assignment[*image].1];
            let point_camera = pose.transform_world_point(&point);
            point_camera.z > 0.0
                && sensor.camera.project(&point_camera).is_some_and(|pixel| {
                    (pixel - features[*image].keypoints[*keypoint]).norm()
                        <= config.max_reprojection_error_px
                })
        }) {
            track.position = Some(point);
            update.landmarks.push(track_index);
        }
    }
    update
}

fn closest_ray_midpoint(
    origin_left: &Point3<f64>,
    direction_left: &Vector3<f64>,
    origin_right: &Point3<f64>,
    direction_right: &Vector3<f64>,
) -> Option<Point3<f64>> {
    let offset = origin_left - origin_right;
    let cosine = direction_left.dot(direction_right);
    let denominator = 1.0 - cosine * cosine;
    if denominator <= 1.0e-12 {
        return None;
    }
    let left_projection = direction_left.dot(&offset);
    let right_projection = direction_right.dot(&offset);
    let left_depth = (cosine * right_projection - left_projection) / denominator;
    let right_depth = (right_projection - cosine * left_projection) / denominator;
    if left_depth <= 0.0 || right_depth <= 0.0 {
        return None;
    }
    let point_left = origin_left + left_depth * direction_left;
    let point_right = origin_right + right_depth * direction_right;
    Some(Point3::from((point_left.coords + point_right.coords) * 0.5))
}

fn reprojection_error(
    rig: &GeneralizedCameraRig,
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for track in tracks {
        for (image, _, pixel) in &track.observations {
            let Some(pose) = image_poses[*image].as_ref() else {
                continue;
            };
            let sensor = &rig.sensors()[image_assignment[*image].1];
            if let Some(projected) = sensor
                .camera
                .project(&pose.transform_world_point(&track.position))
            {
                sum += (projected - pixel).norm();
                count += 1;
            }
        }
    }
    (sum, count)
}

#[cfg(test)]
mod tests {
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;
    use visloc_core::types::Camera;
    use visloc_vision::pnp::RigSensor;

    use super::*;

    #[test]
    fn reconstructs_metric_synchronized_rig_frames() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let world_points = (0..24)
            .map(|index| {
                Point3::new(
                    (index % 6) as f64 * 0.28 - 0.7,
                    (index / 6) as f64 * 0.24 - 0.35,
                    4.0 + (index % 5) as f64 * 0.17,
                )
            })
            .collect::<Vec<_>>();
        let truth = (0..8)
            .map(|frame| {
                Pose::from_world_to_camera(
                    UnitQuaternion::from_euler_angles(
                        0.002 * frame as f64,
                        -0.01 * frame as f64,
                        0.003 * frame as f64,
                    ),
                    Vector3::new(-0.08 * frame as f64, 0.01 * frame as f64, 0.0),
                )
            })
            .collect::<Vec<_>>();
        let mut frames = Vec::new();
        let mut features = Vec::new();
        for (frame_index, frame_pose) in truth.iter().enumerate() {
            let mut frame_images = Vec::new();
            for sensor_index in 0..2 {
                let image_index = features.len();
                let image_pose = rig.sensors()[sensor_index]
                    .sensor_from_rig
                    .compose(&frame_pose.world_to_camera);
                let keypoints = world_points
                    .iter()
                    .map(|point| {
                        rig.sensors()[sensor_index]
                            .camera
                            .project(&image_pose.transform_point(point))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                features
                    .push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
                frame_images.push(RigFrameImage {
                    image_index,
                    sensor_index,
                });
            }
            assert_eq!(frame_images[0].image_index, frame_index * 2);
            frames.push(RigFrame {
                images: frame_images,
            });
        }
        let identity_matches = (0..world_points.len())
            .map(|index| (index, index))
            .collect::<Vec<_>>();
        let mut pairwise = Vec::new();
        for frame in 0..8 {
            pairwise.push(PairwiseMatches::new(
                frame * 2,
                frame * 2 + 1,
                identity_matches.clone(),
            ));
            if frame + 1 < 8 {
                pairwise.push(PairwiseMatches::new(
                    frame * 2,
                    (frame + 1) * 2,
                    identity_matches.clone(),
                ));
                pairwise.push(PairwiseMatches::new(
                    frame * 2 + 1,
                    (frame + 1) * 2 + 1,
                    identity_matches.clone(),
                ));
            }
        }

        let result = incremental_rig_sfm(
            &rig,
            &frames,
            &features,
            &pairwise,
            &RigSfmConfig::default(),
        )
        .unwrap();
        assert_eq!(result.registered_frames, 8);
        assert_eq!(result.registered_images, 16);
        assert_eq!(result.tracks.len(), world_points.len());
        assert!(result.mean_reprojection_error_px < 1.0e-4);
        let total_observations = result
            .tracks
            .iter()
            .map(|track| track.observations.len())
            .sum::<usize>();
        assert_eq!(
            result.work.correspondence_cache_insertions,
            total_observations
        );
        assert!(result.work.triangulation_attempts <= total_observations);
        assert!(result.work.pnp_attempts <= frames.len() - 1);
        for (estimated, expected) in result.frame_poses.iter().zip(&truth) {
            let estimated = estimated.as_ref().unwrap();
            assert!(
                (estimated.camera_center_world() - expected.camera_center_world()).norm() < 1.0e-4
            );
            assert!(
                estimated
                    .world_to_camera
                    .rotation
                    .angle_to(&expected.world_to_camera.rotation)
                    < 1.0e-4
            );
        }
        let baseline = (result.image_poses[0]
            .as_ref()
            .unwrap()
            .camera_center_world()
            - result.image_poses[1]
                .as_ref()
                .unwrap()
                .camera_center_world())
        .norm();
        assert!((baseline - 0.20).abs() < 1.0e-9);
    }
}
