//! Rig bundle adjustment: local-window selection, structure-only refinement,
//! windowed final BA and gauge anchoring.

use super::*;

/// Visit only indexed incident tracks; never materialize a frame-pair graph.
pub(super) fn covisible_local_frames(
    newest: usize,
    cap: usize,
    incident_tracks: impl IntoIterator<Item = usize>,
    tracks: &[WorkingTrack],
    image_assignment: &[(usize, usize)],
    mut usable: impl FnMut(usize, usize, &WorkingTrack) -> bool,
) -> HashSet<usize> {
    if cap < 2 {
        return HashSet::new();
    }
    let mut seen_tracks = HashSet::new();
    let mut counts = HashMap::<usize, usize>::new();
    let mut frames = HashSet::new();
    for id in incident_tracks {
        if !seen_tracks.insert(id) {
            continue;
        }
        let track = &tracks[id];
        if track.position.is_none() {
            continue;
        }
        frames.clear();
        for &(image, keypoint) in &track.observations {
            if usable(image, keypoint, track) {
                frames.insert(image_assignment[image].0);
            }
        }
        if !frames.remove(&newest) {
            continue;
        }
        for &frame in &frames {
            *counts.entry(frame).or_default() += 1;
        }
    }
    // The heap root is the worst retained candidate: low count, then high ID.
    let mut best = BinaryHeap::new();
    for (frame, count) in counts {
        best.push(Reverse((count, Reverse(frame))));
        if best.len() > cap - 1 {
            best.pop();
        }
    }
    if best.is_empty() {
        return HashSet::new();
    }
    let mut selected = best
        .into_iter()
        .map(|Reverse((_, Reverse(frame)))| frame)
        .collect::<HashSet<_>>();
    selected.insert(newest);
    selected
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_local_ba_frames(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_tracks: &[Vec<(usize, usize)>],
    image_poses: &[Option<Pose>],
    tracks: &[WorkingTrack],
    registration_order: &[usize],
    config: &RigSfmConfig,
) -> (HashSet<usize>, usize) {
    let start = registration_order
        .len()
        .saturating_sub(config.local_ba_window_size);
    if !config.local_ba_covisibility {
        return (
            registration_order[start..].iter().copied().collect(),
            registration_order[start],
        );
    }
    let newest = *registration_order
        .last()
        .expect("local BA requires registration");
    let active = covisible_local_frames(
        newest,
        config.local_ba_window_size,
        frames[newest]
            .images
            .iter()
            .flat_map(|image| image_tracks[image.image_index].iter().map(|&(_, id)| id)),
        tracks,
        image_assignment,
        |image, keypoint, track| {
            if config.ba_metric_tracks_only && !track.metric_anchored {
                return false;
            }
            let Some(pose) = &image_poses[image] else {
                return false;
            };
            let sensor = &rig.sensors()[image_assignment[image].1];
            sensor
                .camera
                .project(&pose.transform_world_point(&track.position.unwrap()))
                .is_some_and(|pixel| {
                    (pixel - features[image].keypoints[keypoint]).norm()
                        <= 2.0 * config.max_reprojection_error_px
                })
        },
    );
    let anchor = registration_order
        .iter()
        .copied()
        .find(|frame| active.contains(frame))
        .unwrap_or(newest);
    (active, anchor)
}

pub(super) fn merge_rig_ba_stats(target: &mut Option<RigBaStats>, source: Option<RigBaStats>) {
    let Some(source) = source else {
        return;
    };
    let target = target.get_or_insert(RigBaStats {
        observations: 0,
        initial_cost: 0.0,
        final_cost: 0.0,
        iterations: 0,
        converged: true,
    });
    target.observations += source.observations;
    target.initial_cost += source.initial_cost;
    target.final_cost += source.final_cost;
    target.iterations += source.iterations;
    target.converged &= source.converged;
}

pub(super) fn refine_rig_structure(
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
pub(super) fn run_windowed_final_ba(
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
                config.final_ba_min_pose_observations,
                &extra_fixed,
                false,
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

fn rig_ba_pose_edges(
    observations: impl IntoIterator<Item = (u64, u64)>,
) -> BTreeMap<u64, Vec<u64>> {
    // A landmark star preserves connectivity without a pose-pair clique.
    // Storage is O(observations + landmarks + poses), never O(poses squared).
    let mut first = BTreeMap::new();
    let mut edges: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (pose, landmark) in observations {
        let representative = *first.entry(landmark).or_insert(pose);
        if pose != representative {
            edges.entry(pose).or_default().push(representative);
            edges.entry(representative).or_default().push(pose);
        }
    }
    edges
}

pub(super) fn rig_ba_anchor_reachable(
    observations: impl IntoIterator<Item = (u64, u64)>,
    fixed: impl IntoIterator<Item = u64>,
) -> BTreeSet<u64> {
    let edges = rig_ba_pose_edges(observations);
    let mut reached: BTreeSet<_> = fixed.into_iter().collect();
    let mut pending: Vec<_> = reached.iter().copied().collect();
    while let Some(pose) = pending.pop() {
        if let Some(neighbors) = edges.get(&pose) {
            for &neighbor in neighbors {
                if reached.insert(neighbor) {
                    pending.push(neighbor);
                }
            }
        }
    }
    reached
}

pub(super) fn rig_ba_component_anchors(
    observations: impl IntoIterator<Item = (u64, u64)>,
    poses: impl IntoIterator<Item = u64>,
    fixed: &BTreeSet<u64>,
) -> Vec<u64> {
    let edges = rig_ba_pose_edges(observations);
    let poses: BTreeSet<_> = poses.into_iter().collect();
    let mut visited = BTreeSet::new();
    let mut anchors = Vec::new();
    for pose in poses {
        if !visited.insert(pose) {
            continue;
        }
        let mut has_fixed = false;
        let mut pending = vec![pose];
        while let Some(current) = pending.pop() {
            has_fixed |= fixed.contains(&current);
            if let Some(neighbors) = edges.get(&current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        pending.push(neighbor);
                    }
                }
            }
        }
        if !has_fixed {
            anchors.push(pose);
        }
    }
    anchors
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_rig_bundle_adjustment(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    active_frames: &HashSet<usize>,
    anchor_frame_index: usize,
    ba_config: &BaConfig,
    min_pose_observations: usize,
    extra_fixed_frames: &[usize],
    fix_active_rotations: bool,
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
    if fix_active_rotations {
        for &frame in active_frames {
            if frame != anchor_frame_index && frame_poses[frame].is_some() {
                problem.fix_pose_rotation(frame as u64);
            }
        }
    }

    let mut visual_observations = 0usize;
    let mut observations_per_frame = vec![0usize; frame_poses.len()];
    for (track_index, track) in tracks.iter().enumerate() {
        if config.ba_metric_tracks_only && !track.metric_anchored {
            continue;
        }
        let Some(position) = track.position else {
            continue;
        };
        let mut observations = Vec::new();
        let mut has_active_observation = false;
        for &(image, keypoint) in &track.observations {
            let Some(pose) = image_poses[image].as_ref() else {
                continue;
            };
            let (frame, sensor_index) = image_assignment[image];
            let active = active_frames.contains(&frame);
            if !active && (!config.ba_fixed_boundary_observations || frame_poses[frame].is_none()) {
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
                has_active_observation |= active;
                // Preserve the historical counter path when disabled.
                if !config.ba_fixed_boundary_observations {
                    observations_per_frame[frame] += 1;
                }
                observations.push(BaRigObservation {
                    keyframe_id: frame as u64,
                    landmark_id: track_index as u64,
                    xy: pixel,
                    camera: sensor.camera.clone(),
                    sensor_from_rig: sensor.sensor_from_rig.clone(),
                });
            }
        }
        if observations.len() < 2 || !has_active_observation {
            continue;
        }
        problem.add_landmark(track_index as u64, position);
        visual_observations += observations.len();
        for observation in observations {
            if config.ba_fixed_boundary_observations {
                let frame = observation.keyframe_id as usize;
                observations_per_frame[frame] += 1;
                if !active_frames.contains(&frame) {
                    if !problem.poses.contains_key(&observation.keyframe_id) {
                        problem.add_pose(
                            observation.keyframe_id,
                            frame_poses[frame].as_ref().unwrap().clone(),
                        );
                    }
                    problem.fix_pose(observation.keyframe_id);
                }
            }
            problem.add_rig_observation(observation);
        }
    }
    if visual_observations == 0 {
        if config.ba_backend != RigBaBackend::Legacy
            && problem
                .poses
                .keys()
                .any(|id| !problem.fixed_poses.contains(id))
        {
            let reason = "matrix-free backend has variable poses but no visual observations";
            eprintln!(
                "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
            );
            return Err(RigSfmError::BundleAdjustment(reason.into()));
        }
        return Ok(None);
    }
    if min_pose_observations > 0 {
        for &frame in active_frames {
            if observations_per_frame[frame] < min_pose_observations {
                problem.fix_pose(frame as u64);
            }
        }
    }
    let mut matrix_free_report = None;
    if config.ba_anchor_disconnected_components {
        let anchors = rig_ba_component_anchors(
            problem
                .rig_observations
                .iter()
                .map(|o| (o.keyframe_id, o.landmark_id)),
            problem.poses.keys().copied(),
            &problem.fixed_poses,
        );
        for id in anchors {
            problem.fix_pose(id);
            eprintln!("rig-ba-component-anchor: frame={id} original_anchor={anchor_frame_index}");
        }
    }
    if std::env::var_os("VISLOC_SFM_TRACE_BA_CONNECTIVITY").is_some() {
        let reached = rig_ba_anchor_reachable(
            problem
                .rig_observations
                .iter()
                .map(|o| (o.keyframe_id, o.landmark_id)),
            problem.fixed_poses.iter().copied(),
        );
        for id in problem.poses.keys() {
            eprintln!(
                "rig-ba-connectivity: frame={id} anchor={anchor_frame_index} fixed={} connected_to_fixed={}",
                problem.fixed_poses.contains(id), reached.contains(id),
            );
        }
    }
    // Scalar-only context for pairing native windows with existing LM debug
    // records. No state copy, solve-policy change, or global pair graph.
    if std::env::var_os("VISLOC_SFM_DEBUG_BA").is_some()
        && std::env::var_os("VISLOC_SFM_DEBUG_BA_STEPS").is_some()
    {
        let variable_poses = problem
            .poses
            .keys()
            .filter(|id| !problem.fixed_poses.contains(id))
            .count();
        eprintln!(
            "rig-ba-context: backend={:?} poses={} variable_poses={} landmarks={} observations={} first_frame={:?} last_frame={:?} anchor={} fixed_rotations={}",
            config.ba_backend, problem.poses.len(), variable_poses,
            problem.landmarks.len(), visual_observations,
            problem.poses.keys().min(), problem.poses.keys().max(),
            anchor_frame_index, fix_active_rotations,
        );
    }
    let selected_backend = select_bounded_rig_backend(
        config.ba_backend,
        problem
            .poses
            .keys()
            .filter(|id| !problem.fixed_poses.contains(id))
            .count(),
    );
    if config.ba_backend == RigBaBackend::BoundedDirect64Qr {
        let validation = if problem
            .poses
            .keys()
            .any(|id| !problem.fixed_poses.contains(id))
        {
            problem.validate_matrix_free_entry(ba_config, MatrixFreeBaOptions::default())
        } else {
            problem.validate_matrix_free_landmark_only(ba_config, MatrixFreeBaOptions::default())
        };
        validation.map_err(|error| RigSfmError::BundleAdjustment(error.to_string()))?;
        eprintln!("rig-ba-policy: requested=bounded-direct64-qr selected={selected_backend:?} direct_pose_cap=64");
    }
    let (initial_cost, final_cost, iterations, converged) = match selected_backend {
        RigBaBackend::BoundedDirect64Qr => unreachable!("bounded policy resolves before solve"),
        RigBaBackend::Legacy => {
            // Keep the historical call and caller-provided BaConfig entirely
            // unchanged when the new selector is absent.
            let result = problem
                .optimize(ba_config)
                .map_err(|error| RigSfmError::BundleAdjustment(error.to_string()))?;
            (
                result.initial_cost,
                result.final_cost,
                result.iterations.len(),
                result.converged,
            )
        }
        RigBaBackend::MatrixFreeStrict
        | RigBaBackend::MatrixFreeCluster8
        | RigBaBackend::MatrixFreeCluster8Restart1
        | RigBaBackend::MatrixFreeQr => {
            let has_variable_pose = problem
                .poses
                .keys()
                .any(|id| !problem.fixed_poses.contains(id));
            let has_variable_landmark = problem
                .landmarks
                .keys()
                .any(|id| !problem.fixed_landmarks.contains(id));

            if !has_variable_pose && has_variable_landmark {
                // The public matrix-free validator intentionally requires a
                // variable pose.  A native window can nevertheless have all
                // poses fixed while retaining variable landmarks.  This is a
                // bounded landmark-only visual solve: its pose Schur dimension
                // is exactly zero, so using the existing optimizer here does
                // not create a hidden large legacy pose solve or change the
                // observation/loss semantics.  Explicitly reject any
                // calibration refinement rather than silently changing the
                // requested problem.
                if let Err(error) = problem
                    .validate_matrix_free_landmark_only(ba_config, MatrixFreeBaOptions::default())
                {
                    eprintln!(
                        "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={error}"
                    );
                    return Err(RigSfmError::BundleAdjustment(error.to_string()));
                }
                let result = problem
                    .optimize(ba_config)
                    .map_err(|error| {
                        eprintln!(
                            "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={error}"
                        );
                        RigSfmError::BundleAdjustment(error.to_string())
                    })?;
                matrix_free_report = Some((
                    "landmark-only",
                    result.iterations.len(),
                    result
                        .iterations
                        .iter()
                        .filter(|step| step.step_accepted)
                        .count(),
                    0usize,
                ));
                (
                    result.initial_cost,
                    result.final_cost,
                    result.iterations.len(),
                    result.converged,
                )
            } else if has_variable_pose {
                let result = match selected_backend {
                    RigBaBackend::MatrixFreeQr => problem.optimize_rig_qr(
                        ba_config, MatrixFreeBaOptions::default()),
                    RigBaBackend::MatrixFreeCluster8Restart1 => problem.optimize_matrix_free_cluster8_with_restart(
                        ba_config, MatrixFreeBaOptions::default(), MatrixFreeBaRestartOptions { max_restarts_per_solve: 1 })
                        .map(|result| {
                            eprintln!("rig-ba-restart: limit=1 restarts={} max_pcg_iterations={} rechecks={} failed_rechecks={}",
                                result.restart_iterations.iter().map(|s| s.restarts).sum::<usize>(),
                                result.restart_iterations.iter().filter_map(|s| s.pcg_iterations).max().unwrap_or(0),
                                result.restart_iterations.iter().map(|s| s.true_residual_rechecks).sum::<usize>(),
                                result.restart_iterations.iter().map(|s| s.failed_true_residual_rechecks).sum::<usize>());
                            result.ba
                        }),
                    RigBaBackend::MatrixFreeCluster8 => problem.optimize_matrix_free_cluster8(
                        ba_config, MatrixFreeBaOptions::default()),
                    _ => problem.optimize_matrix_free(ba_config, MatrixFreeBaOptions::default()),
                }
                    .map_err(|error| {
                        eprintln!(
                            "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={error}"
                        );
                        RigSfmError::BundleAdjustment(error.to_string())
                    })?;
                matrix_free_report = Some((
                    if selected_backend == RigBaBackend::MatrixFreeQr {
                        "matrix-free-qr"
                    } else if config.ba_backend == RigBaBackend::MatrixFreeCluster8Restart1 {
                        "matrix-free-cluster8-restart1"
                    } else if config.ba_backend == RigBaBackend::MatrixFreeCluster8 {
                        "matrix-free-cluster8"
                    } else {
                        "matrix-free"
                    },
                    result.iterations.len(),
                    result
                        .iterations
                        .iter()
                        .filter(|step| step.step_accepted)
                        .count(),
                    result
                        .matrix_free_iterations
                        .iter()
                        .filter(|step| step.pcg_failure.is_some())
                        .count(),
                ));
                (
                    result.initial_cost,
                    result.final_cost,
                    result.iterations.len(),
                    result.converged,
                )
            } else {
                // This can only occur when the assembled problem has no
                // variable state at all.  It is a genuine no-op, not a
                // numerical failure, and has no mapper write-back to do.
                if let Err(error) = problem
                    .validate_matrix_free_landmark_only(ba_config, MatrixFreeBaOptions::default())
                {
                    eprintln!(
                        "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={error}"
                    );
                    return Err(RigSfmError::BundleAdjustment(error.to_string()));
                }
                let cost = problem.robust_cost(&ba_config.robust_kernel);
                if !cost.is_finite() {
                    return Err(RigSfmError::BundleAdjustment(
                        "matrix-free no-variable cost is non-finite".into(),
                    ));
                }
                matrix_free_report = Some(("no-op", 0, 0, 0));
                (cost, cost, 0, true)
            }
        }
    };
    if config.ba_backend != RigBaBackend::Legacy {
        for &fixed_pose_id in &problem.fixed_poses {
            let Some(original) = frame_poses
                .get(fixed_pose_id as usize)
                .and_then(Option::as_ref)
            else {
                let reason = format!("fixed pose {fixed_pose_id} disappeared during BA");
                eprintln!(
                    "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
                );
                return Err(RigSfmError::BundleAdjustment(reason));
            };
            let Some(refined) = problem.poses.get(&fixed_pose_id) else {
                let reason = format!("fixed pose {fixed_pose_id} disappeared from BA state");
                eprintln!(
                    "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
                );
                return Err(RigSfmError::BundleAdjustment(reason));
            };
            if refined != original {
                let reason = format!("fixed pose {fixed_pose_id} changed during matrix-free BA");
                eprintln!(
                    "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
                );
                return Err(RigSfmError::BundleAdjustment(reason));
            }
        }
        for &fixed_pose_id in &problem.fixed_pose_rotations {
            let Some(original) = frame_poses
                .get(fixed_pose_id as usize)
                .and_then(Option::as_ref)
            else {
                let reason = format!("fixed-rotation pose {fixed_pose_id} disappeared during BA");
                eprintln!(
                    "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
                );
                return Err(RigSfmError::BundleAdjustment(reason));
            };
            let Some(refined) = problem.poses.get(&fixed_pose_id) else {
                let reason =
                    format!("fixed-rotation pose {fixed_pose_id} disappeared from BA state");
                eprintln!(
                    "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
                );
                return Err(RigSfmError::BundleAdjustment(reason));
            };
            if refined.world_to_camera.rotation != original.world_to_camera.rotation {
                let reason =
                    format!("fixed rotation pose {fixed_pose_id} changed during matrix-free BA");
                eprintln!(
                    "rig-ba-backend: requested=matrix-free used=none eligibility=error reason={reason}"
                );
                return Err(RigSfmError::BundleAdjustment(reason));
            }
        }
        if let Some((used, iterations, accepted, pcg_failures)) = matrix_free_report {
            eprintln!(
                "rig-ba-backend: requested=matrix-free used={used} eligibility=eligible iterations={iterations} accepted={accepted} pcg_failures={pcg_failures}"
            );
        }
    }
    // Diagnostic only: compare within one solve against its fixed anchor.
    // Stream scalar records without retaining pose histories or pair graphs.
    if std::env::var_os("VISLOC_SFM_TRACE_BA_POSE_MOTION").is_some() {
        if let (Some(before_anchor), Some(after_anchor)) = (
            frame_poses[anchor_frame_index].as_ref(),
            problem.poses.get(&(anchor_frame_index as u64)),
        ) {
            for (&id, refined) in &problem.poses {
                if let Some(original) = frame_poses[id as usize].as_ref() {
                    let before = (original.camera_center_world()
                        - before_anchor.camera_center_world())
                    .norm();
                    let after =
                        (refined.camera_center_world() - after_anchor.camera_center_world()).norm();
                    eprintln!(
                        "rig-ba-pose-motion: frame={id} anchor={anchor_frame_index} fixed={} anchor_distance_before={before:.17e} anchor_distance_after={after:.17e}",
                        problem.fixed_poses.contains(&id),
                    );
                }
            }
        }
    }
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
        initial_cost,
        final_cost,
        iterations,
        converged,
    }))
}
