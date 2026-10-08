//! Bundle-adjustment driving, global/local refinement and landmark BA diagnostics.

use super::*;

/// Global BA over all registered poses + triangulated landmarks. Seed pose
/// (the lowest-index registered image) is fixed for gauge. Writes refined
/// poses and points back in place. When `refine_intrinsics` is set, the BA also
/// refines the pinhole intrinsics (alternating) and the refined camera is
/// returned as `Some` (the caller propagates it); otherwise the second tuple
/// element is `None` and the camera is untouched.
pub(crate) fn run_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
) -> Result<(BaResult, Option<Camera>), BaError> {
    run_bundle_adjustment_impl(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        refine_intrinsics,
        false,
    )
}

/// Summary of the optional camera-fixed landmark warm start that precedes a
/// joint global/periodic BA. All fields are scalar so this remains cheap to
/// report from the registration loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LandmarkBaWarmStartStats {
    pub(crate) requested_iterations: usize,
    pub(crate) attempted: bool,
    pub(crate) accepted: bool,
    pub(crate) points: usize,
    pub(crate) observations: usize,
    pub(crate) initial_cost: f64,
    pub(crate) final_cost: f64,
    pub(crate) solver_iterations: usize,
    pub(crate) accepted_steps: usize,
    pub(crate) rejected_steps: usize,
    pub(crate) converged: bool,
    pub(crate) max_displacement: f64,
    pub(crate) median_displacement: f64,
}

/// Optimize only the currently triangulated landmarks while keeping every
/// registered camera and the intrinsics fixed. This is deliberately separate
/// from [`run_bundle_adjustment_impl`]: it uses the same robust residuals and
/// solver, but never exposes a pose variable to the Schur system. A candidate
/// result is copied back only when the robust cost and every point are finite
/// and the cost is non-increasing; otherwise the caller's points are left
/// untouched and the subsequent ordinary joint BA still runs.
pub(super) fn run_landmark_ba_warm_start(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> LandmarkBaWarmStartStats {
    let requested_iterations = config.landmark_ba_warm_start_iterations;
    let mut stats = LandmarkBaWarmStartStats {
        requested_iterations,
        attempted: false,
        accepted: false,
        points: 0,
        observations: 0,
        initial_cost: f64::NAN,
        final_cost: f64::NAN,
        solver_iterations: 0,
        accepted_steps: 0,
        rejected_steps: 0,
        converged: false,
        max_displacement: 0.0,
        median_displacement: 0.0,
    };
    if requested_iterations == 0 {
        return stats;
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    for (image, pose) in poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        ba.add_pose(image as u64, pose.clone());
        ba.fix_pose(image as u64);
    }

    // Keep the input order (track id, then observation order) exactly as the
    // ordinary BA builder. The point-only solve therefore has no alternate
    // ordering or matching policy hidden behind the opt-in switch.
    let mut point_ids = Vec::new();
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, keypoint) in track {
            if poses.get(image).and_then(Option::as_ref).is_none() {
                continue;
            }
            let Some(xy) = features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint))
            else {
                continue;
            };
            observations.push(BaObservation {
                keyframe_id: image as u64,
                landmark_id: track_id as u64,
                xy: *xy,
            });
        }
        if observations.len() < 2 {
            continue;
        }
        ba.add_landmark(track_id as u64, *point);
        for observation in observations {
            ba.add_observation(observation);
        }
        point_ids.push(track_id);
    }
    stats.points = point_ids.len();
    stats.observations = ba.observations.len();
    if point_ids.is_empty() || ba.poses.is_empty() || ba.observations.is_empty() {
        return stats;
    }
    stats.attempted = true;

    let mut warm_config = config.ba_config;
    warm_config.max_iterations = requested_iterations;
    warm_config.refine_intrinsics = false;
    let result = match ba.optimize(&warm_config) {
        Ok(result) => result,
        Err(error) => {
            if sfm_ba_debug_enabled() {
                eprintln!(
                    "sfm-debug-ba-warm-start: solver error={error:?}; keeping input landmarks"
                );
            }
            return stats;
        }
    };
    stats.initial_cost = result.initial_cost;
    stats.final_cost = result.final_cost;
    stats.solver_iterations = result.iterations.len();
    stats.accepted_steps = result
        .iterations
        .iter()
        .filter(|iteration| iteration.step_accepted)
        .count();
    stats.rejected_steps = result.iterations.len().saturating_sub(stats.accepted_steps);
    stats.converged = result.converged;

    let mut displacements = Vec::with_capacity(point_ids.len());
    let mut finite_points = true;
    for &track_id in &point_ids {
        let Some(before) = track_point.get(track_id).and_then(Option::as_ref) else {
            finite_points = false;
            break;
        };
        let Some(after) = ba.landmarks.get(&(track_id as u64)) else {
            finite_points = false;
            break;
        };
        if !after.coords.iter().all(|value| value.is_finite()) {
            finite_points = false;
            break;
        }
        let displacement = (after.coords - before.coords).norm();
        if !displacement.is_finite() {
            finite_points = false;
            break;
        }
        displacements.push(displacement);
    }
    if !displacements.is_empty() {
        stats.max_displacement = displacements.iter().copied().fold(0.0, f64::max);
        stats.median_displacement = sfm_oracle_median(&mut displacements);
    }
    stats.accepted = finite_points
        && stats.initial_cost.is_finite()
        && stats.final_cost.is_finite()
        && stats.final_cost <= stats.initial_cost;
    if stats.accepted {
        for &track_id in &point_ids {
            track_point[track_id] = ba.landmarks.get(&(track_id as u64)).copied();
        }
    }
    if sfm_ba_debug_enabled() {
        eprintln!(
            concat!(
                "sfm-debug-ba-warm-start: requested_iterations={} attempted={} accepted={} ",
                "points={} observations={} initial_cost={:.9e} final_cost={:.9e} ",
                "solver_iterations={} accepted_steps={} rejected_steps={} converged={} ",
                "point_delta_max/median=({:.3e},{:.3e})m"
            ),
            stats.requested_iterations,
            stats.attempted,
            stats.accepted,
            stats.points,
            stats.observations,
            stats.initial_cost,
            stats.final_cost,
            stats.solver_iterations,
            stats.accepted_steps,
            stats.rejected_steps,
            stats.converged,
            stats.max_displacement,
            stats.median_displacement,
        );
    }
    stats
}

/// Implementation shared by the historical BA path and the opt-in final
/// geometry-weighted solve. Keeping the switch here means all ordinary BA
/// callers continue to use the exact legacy `optimize` path.
#[allow(clippy::too_many_arguments)]
fn run_bundle_adjustment_impl(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
    geometry_weighted: bool,
) -> Result<(BaResult, Option<Camera>), BaError> {
    run_bundle_adjustment_impl_with_fixed_rotations(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        refine_intrinsics,
        geometry_weighted,
        None,
    )
}

/// Implementation shared by ordinary BA and the opt-in fixed-rotation
/// diagnostic.  The latter supplies image indices whose pose rotations are
/// constrained while translations/landmarks remain ordinary BA variables.
#[allow(clippy::too_many_arguments)]
fn run_bundle_adjustment_impl_with_fixed_rotations(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
    geometry_weighted: bool,
    fixed_rotation_images: Option<&BTreeSet<usize>>,
) -> Result<(BaResult, Option<Camera>), BaError> {
    let timing_enabled = std::env::var_os("VISLOC_SFM_TIMING").is_some();
    let total_started = std::time::Instant::now();
    let oracle_poses_before = config.debug_oracle_poses.as_ref().map(|_| poses.to_vec());
    let ba_debug = sfm_ba_debug_enabled();
    let ba_step_debug = sfm_ba_step_debug_enabled();
    let support_before = ba_debug.then(|| {
        (
            poses.iter().filter(|pose| pose.is_some()).count(),
            track_point.iter().filter(|point| point.is_some()).count(),
            count_observations(tracks, poses, track_point),
        )
    });
    let poses_before = ba_debug.then(|| poses.to_vec());
    let points_before = ba_debug.then(|| track_point.to_vec());
    let registered_before = poses.iter().filter(|pose| pose.is_some()).count();
    let warm_start_started = std::time::Instant::now();
    let warm_start_stats = if config.landmark_ba_warm_start_iterations > 0
        && registered_before >= config.landmark_ba_warm_start_min_registered_images
    {
        Some(run_landmark_ba_warm_start(
            camera,
            features,
            tracks,
            config,
            poses,
            track_point,
        ))
    } else {
        if sfm_ba_debug_enabled() && config.landmark_ba_warm_start_iterations > 0 {
            eprintln!(
                "sfm-debug-ba-warm-start: skipped registered={} minimum_registered={}",
                registered_before, config.landmark_ba_warm_start_min_registered_images,
            );
        }
        None
    };
    let warm_start_seconds = warm_start_started.elapsed().as_secs_f64();
    let assembly_started = std::time::Instant::now();
    process_memory::log("ba-before-assembly");
    let mut ba = BundleAdjustment::new(camera.clone());
    let ba_config = BaConfig {
        refine_intrinsics,
        ..config.ba_config
    };

    for (image, pose) in poses.iter().enumerate() {
        if let Some(pose) = pose {
            ba.add_pose(image as u64, pose.clone());
            if fixed_rotation_images.is_some_and(|images| images.contains(&image)) {
                ba.fix_pose_rotation(image as u64);
            }
        }
    }

    // Gauge fixing. A monocular reconstruction (no stereo residual) has 7 gauge
    // freedoms: 6 for the rigid SE(3) frame plus **1 for global scale**. Fixing
    // a single pose pins only the 6 rigid DoF and leaves scale unconstrained, so
    // the BA's normal equations are singular along the scale direction. A single
    // solve from a perturbed state tolerates that (the damping holds the null
    // direction), but **re-optimising from an already-converged state lets the
    // scale drift and the reconstruction collapse**. Pin scale too by also
    // fixing the registered pose whose camera centre is farthest from the
    // anchor — the longest, best-conditioned baseline.
    let anchor = poses.iter().position(|p| p.is_some());
    let mut scale_anchor = None;
    if let Some(anchor) = anchor {
        ba.fix_pose(anchor as u64);
        let anchor_center = poses[anchor]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .translation;
        let mut farthest = None;
        let mut best_d2 = 0.0;
        for (image, pose) in poses.iter().enumerate() {
            if image == anchor {
                continue;
            }
            if let Some(pose) = pose {
                let d2 = (pose.camera_to_world().translation - anchor_center).norm_squared();
                if d2 > best_d2 {
                    best_d2 = d2;
                    farthest = Some(image);
                }
            }
        }
        if let Some(scale_anchor_image) = farthest {
            ba.fix_pose(scale_anchor_image as u64);
            scale_anchor = Some(scale_anchor_image);
        }
    }

    let collect_landmark_geometry =
        config.freeze_ill_conditioned_landmarks || sfm_ba_landmark_debug_enabled();
    let mut landmark_diagnostics = Vec::new();
    let mut excluded_landmarks = 0usize;
    let mut excluded_observations = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        let mut obs = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: px,
                });
            }
        }
        if obs.len() >= 2 {
            let (geometry, excluded) = if collect_landmark_geometry {
                let geometry = ba_landmark_geometry(
                    camera,
                    features,
                    poses,
                    track,
                    &point,
                    &ba_config.robust_kernel,
                );
                let excluded = config.freeze_ill_conditioned_landmarks
                    && ba_landmark_should_exclude(
                        &geometry,
                        config.min_triangulation_angle_deg,
                        config.max_reprojection_error_px,
                    );
                (Some(geometry), excluded)
            } else {
                (None, false)
            };
            if excluded {
                // A fixed point with a large pre-BA residual would still pull
                // the camera Schur system through its observation rows. Such a
                // point is not a trustworthy camera constraint, so this
                // conditioning mode drops its rows for this solve instead of
                // retaining mathematically misleading residual influence.
                excluded_landmarks += 1;
                excluded_observations += obs.len();
            } else {
                ba.add_landmark(track_id as u64, point);
                for o in obs {
                    ba.add_observation(o);
                }
            }
            if collect_landmark_geometry {
                let geometry = geometry.expect("geometry collected for every BA landmark");
                landmark_diagnostics.push(LandmarkBaDiagnostic {
                    id: track_id as u64,
                    geometry,
                    displacement: 0.0,
                    excluded,
                });
            }
        }
    }

    if sfm_ba_jacobian_audit_enabled() {
        let audit = crate::bundle::audit_bundle_visual_jacobians(&ba, 64);
        eprintln!(
            "sfm-debug-ba-jacobian: observations_seen={} samples_audited={} invalid_samples={} normal={:?} far_depth={:?} low_parallax={:?} high_residual={:?}",
            audit.observations_seen,
            audit.samples_audited,
            audit.invalid_samples,
            audit.normal,
            audit.far_depth,
            audit.low_parallax,
            audit.high_residual,
        );
    }

    process_memory::log("ba-after-assembly");
    let initial_l2 = ba_debug.then(|| ba.robust_cost(&RobustKernel::None));
    let initial_robust = ba_debug.then(|| ba.robust_cost(&ba_config.robust_kernel));
    let observation_weights = if geometry_weighted && !refine_intrinsics {
        Some(track_geometry_observation_weights(
            poses,
            tracks,
            track_point,
            &ba.observations,
        ))
    } else {
        None
    };
    if let Some(path) = std::env::var_os("VISLOC_SFM_BA_DUMP") {
        // Replayable copy of this solve for solver benchmarks (overwritten
        // by every global BA, so the file ends with the last one).
        if fixed_rotation_images.is_none() && ba.camera.params.len() >= 4 {
            if let Err(error) = crate::ba_problem_io::write_ba_problem(
                &ba,
                observation_weights.as_deref(),
                std::path::Path::new(&path),
            ) {
                eprintln!("sfm-ba-dump: failed: {error}");
            }
        }
    }
    let pre_optimize_seconds = assembly_started.elapsed().as_secs_f64();
    let optimize_started = std::time::Instant::now();
    let use_matrix_free = sfm_matrix_free_ba_enabled(&ba_config) && fixed_rotation_images.is_none();
    if use_matrix_free && sfm_timing_or_debug_enabled() && !ba_config.matrix_free_ba {
        eprintln!("sfm-matrix-free: enabled via VISLOC_SFM_BA_MATRIX_FREE");
    }
    let accelerated = if observation_weights.is_none() && fixed_rotation_images.is_none() {
        crate::ba_accel::ba_accelerator()
            .and_then(|a| a.optimize(&mut ba, &ba_config, crate::ba_accel::BaScope::Global))
    } else {
        None
    };
    let result = if let Some(result) = accelerated {
        result?
    } else if use_matrix_free {
        ba.optimize_honoring_matrix_free(&ba_config, observation_weights.as_deref())?
    } else if let Some(weights) = observation_weights.as_deref() {
        ba.optimize_with_observation_weights(&ba_config, weights)?
    } else {
        ba.optimize(&ba_config)?
    };
    let optimize_seconds = optimize_started.elapsed().as_secs_f64();
    process_memory::log("ba-after-optimize");

    if !landmark_diagnostics.is_empty() {
        for diagnostic in &mut landmark_diagnostics {
            let Some(before) = track_point
                .get(diagnostic.id as usize)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let Some(after) = ba.landmarks.get(&diagnostic.id) else {
                continue;
            };
            diagnostic.displacement = (after.coords - before.coords).norm();
        }
    }

    if ba_debug {
        let accepted = result
            .iterations
            .iter()
            .filter(|iteration| iteration.step_accepted)
            .count();
        let rejected = result.iterations.len().saturating_sub(accepted);
        let last = result.iterations.last();
        let final_l2 = ba.robust_cost(&RobustKernel::None);
        let robust_kernel_cost = ba.robust_cost(&ba_config.robust_kernel);
        let (pose_center_max, pose_center_median, pose_rotation_max, pose_rotation_median) =
            if let Some(before) = poses_before.as_ref() {
                let mut center_displacements = Vec::new();
                let mut rotation_displacements = Vec::new();
                for (image, old_pose) in before.iter().enumerate() {
                    let (Some(old_pose), Some(new_pose)) =
                        (old_pose.as_ref(), ba.poses.get(&(image as u64)))
                    else {
                        continue;
                    };
                    let center_delta = (new_pose.camera_center_world().coords
                        - old_pose.camera_center_world().coords)
                        .norm();
                    let rotation_delta = (old_pose.camera_to_world().rotation.inverse()
                        * new_pose.camera_to_world().rotation)
                        .angle()
                        .to_degrees();
                    if center_delta.is_finite() {
                        center_displacements.push(center_delta);
                    }
                    if rotation_delta.is_finite() {
                        rotation_displacements.push(rotation_delta);
                    }
                }
                (
                    center_displacements.iter().copied().fold(0.0, f64::max),
                    sfm_oracle_median(&mut center_displacements),
                    rotation_displacements.iter().copied().fold(0.0, f64::max),
                    sfm_oracle_median(&mut rotation_displacements),
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            };
        let (point_max, point_median) = if let Some(before) = points_before.as_ref() {
            let mut displacements = Vec::new();
            for (track_id, old_point) in before.iter().enumerate() {
                let (Some(old_point), Some(new_point)) =
                    (old_point.as_ref(), ba.landmarks.get(&(track_id as u64)))
                else {
                    continue;
                };
                let displacement = (new_point.coords - old_point.coords).norm();
                if displacement.is_finite() {
                    displacements.push(displacement);
                }
            }
            (
                displacements.iter().copied().fold(0.0, f64::max),
                sfm_oracle_median(&mut displacements),
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        let (support_poses_before, support_tracks_before, support_observations_before) =
            support_before.unwrap_or((0, 0, 0));
        let support_poses_after = ba.poses.len();
        let support_tracks_after = ba.landmarks.len();
        let support_observations_after = ba.observations.len();
        eprintln!(
            concat!(
                "sfm-debug-ba: poses={} landmarks={} observations={} max_iterations={} ",
                "geometry_weighted={} ",
                "initial_lambda={:?} kernel={:?} iterations={} accepted={} rejected={} ",
                "converged={} initial_cost={:.9e} final_cost={:.9e} final_l2={:.9e} ",
                "last_step=({:.3e},{:.3e}) last_lambda={:.3e} ",
                "initial_robust={:.9e} initial_l2={:.9e} robust_cost={:.9e} ",
                "support_input=({},{},{}) support_ba=({},{},{}) pruning=none ",
                "conditioned_excluded_landmarks={} excluded_observations={} ",
                "landmark_warm_start=(attempted={},accepted={}) ",
                "pose_delta_max/median=({:.3e},{:.3e})m ",
                "pose_rot_delta_max/median=({:.3e},{:.3e})deg ",
                "point_delta_max/median=({:.3e},{:.3e})m ",
                "gauge_anchor={:?} scale_anchor={:?} ",
                "camera_before={:?} camera_after={:?}"
            ),
            ba.poses.len(),
            ba.landmarks.len(),
            ba.observations.len(),
            ba_config.max_iterations,
            observation_weights.is_some(),
            ba_config.initial_lambda,
            ba_config.robust_kernel,
            result.iterations.len(),
            accepted,
            rejected,
            result.converged,
            result.initial_cost,
            result.final_cost,
            final_l2,
            last.map_or(0.0, |iteration| iteration.max_pose_step),
            last.map_or(0.0, |iteration| iteration.max_landmark_step),
            last.map_or(0.0, |iteration| iteration.lambda),
            initial_robust.unwrap_or(f64::NAN),
            initial_l2.unwrap_or(f64::NAN),
            robust_kernel_cost,
            support_poses_before,
            support_tracks_before,
            support_observations_before,
            support_poses_after,
            support_tracks_after,
            support_observations_after,
            excluded_landmarks,
            excluded_observations,
            warm_start_stats.is_some_and(|stats| stats.attempted),
            warm_start_stats.is_some_and(|stats| stats.accepted),
            pose_center_max,
            pose_center_median,
            pose_rotation_max,
            pose_rotation_median,
            point_max,
            point_median,
            anchor,
            scale_anchor,
            camera.params,
            ba.camera.params,
        );
        if ba_step_debug {
            for iteration in &result.iterations {
                eprintln!(
                    concat!(
                        "sfm-debug-ba-step: iteration={} accepted={} ",
                        "cost={:.9e}->{:.9e} delta={:+.9e} lambda={:.3e} ",
                        "step=({:.3e},{:.3e})"
                    ),
                    iteration.iteration,
                    iteration.step_accepted,
                    iteration.cost_before,
                    iteration.cost_after,
                    iteration.cost_after - iteration.cost_before,
                    iteration.lambda,
                    iteration.max_pose_step,
                    iteration.max_landmark_step,
                );
            }
        }
        if sfm_ba_landmark_debug_enabled() {
            sfm_debug_ba_landmarks(&landmark_diagnostics, config.min_triangulation_angle_deg);
        }
    }

    let writeback_started = std::time::Instant::now();
    for (image, pose) in poses.iter_mut().enumerate() {
        if pose.is_some() {
            if let Some(refined) = ba.poses.get(&(image as u64)) {
                *pose = Some(refined.clone());
            }
        }
    }
    for (track_id, point) in track_point.iter_mut().enumerate() {
        if point.is_some() {
            if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
                *point = Some(*refined);
            }
        }
    }
    if timing_enabled {
        let accepted = result
            .iterations
            .iter()
            .filter(|iteration| iteration.step_accepted)
            .count();
        eprintln!(
            "sfm-timing-ba: registered={} landmarks={} observations={} warm_start={:.3}s assemble={:.3}s solve={:.3}s writeback={:.3}s total={:.3}s iterations={} accepted={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.observations.len(),
            warm_start_seconds,
            pre_optimize_seconds,
            optimize_seconds,
            writeback_started.elapsed().as_secs_f64(),
            total_started.elapsed().as_secs_f64(),
            result.iterations.len(),
            accepted,
        );
    }
    sfm_debug_oracle_transition(
        &format!(
            "ba weighted={} refine_intrinsics={} poses={} observations={}",
            geometry_weighted,
            refine_intrinsics,
            ba.poses.len(),
            ba.observations.len(),
        ),
        oracle_poses_before.as_deref(),
        poses,
        config.debug_oracle_poses.as_deref(),
    );
    let refined_camera = refine_intrinsics.then(|| ba.camera.clone());
    Ok((result, refined_camera))
}

/// Run one ordinary fixed-support BA solve on an externally supplied pose
/// basin and an already assembled [`SfmTrack`] support.
///
/// This is intentionally a diagnostic API: it does not participate in the
/// incremental path, never changes track membership or observations, and
/// leaves the caller's track positions untouched when the solve fails. It is
/// useful for separating mapper-basin errors from BA errors, for example by
/// injecting a COLMAP sparse-model pose set while retaining our own tracks.
pub fn run_fixed_support_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &mut [SfmTrack],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
) -> Result<(BaResult, Option<Camera>), BaError> {
    let local_tracks: Vec<Vec<(usize, usize)>> = tracks
        .iter()
        .map(|track| {
            track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
                .collect()
        })
        .collect();
    let mut points: Vec<Option<Point3<f64>>> =
        tracks.iter().map(|track| Some(track.position)).collect();
    let points_before = points.clone();
    let result = run_bundle_adjustment_impl(
        camera,
        features,
        &local_tracks,
        config,
        poses,
        &mut points,
        config.refine_intrinsics,
        config.geometry_weighted_ba,
    );
    let (result, refined_camera) = match result {
        Ok(result) => result,
        Err(error) => {
            for (track, point) in tracks.iter_mut().zip(points_before) {
                track.position = point.expect("diagnostic track point is present");
            }
            return Err(error);
        }
    };
    for (track, point) in tracks.iter_mut().zip(points) {
        if let Some(point) = point {
            track.position = point;
        }
    }
    Ok((result, refined_camera))
}

/// Run one fixed-support BA solve while pinning the rotations supplied by
/// `fixed_rotations`.  The entries are index-aligned with `poses`; a `Some`
/// entry replaces only that pose's rotation before solving, while its current
/// translation is retained.  Translation and landmark variables remain free,
/// and the ordinary monocular gauge anchors are rebuilt by the same BA path.
///
/// This is an opt-in diagnostic API for separating rotation from translation
/// error.  It does not change track membership or observations and no caller
/// needs to provide entries for unregistered images.  An empty/all-`None`
/// vector is equivalent to [`run_fixed_support_bundle_adjustment`].
pub fn run_fixed_rotation_support_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &mut [SfmTrack],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    fixed_rotations: &[Option<Pose>],
) -> Result<(BaResult, Option<Camera>), BaError> {
    if fixed_rotations.len() != poses.len() {
        return Err(BaError::InvalidFixedRotationCount {
            expected: poses.len(),
            actual: fixed_rotations.len(),
        });
    }
    let poses_before = poses.to_vec();
    let points_before: Vec<Point3<f64>> = tracks.iter().map(|track| track.position).collect();
    let mut fixed_rotation_images = BTreeSet::new();
    for (image, (pose, desired)) in poses.iter_mut().zip(fixed_rotations).enumerate() {
        let (Some(pose), Some(desired)) = (pose.as_mut(), desired.as_ref()) else {
            continue;
        };
        pose.world_to_camera.rotation = desired.world_to_camera.rotation;
        fixed_rotation_images.insert(image);
    }

    let local_tracks: Vec<Vec<(usize, usize)>> = tracks
        .iter()
        .map(|track| {
            track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
                .collect()
        })
        .collect();
    let mut points: Vec<Option<Point3<f64>>> =
        tracks.iter().map(|track| Some(track.position)).collect();
    let result = run_bundle_adjustment_impl_with_fixed_rotations(
        camera,
        features,
        &local_tracks,
        config,
        poses,
        &mut points,
        config.refine_intrinsics,
        config.geometry_weighted_ba,
        Some(&fixed_rotation_images),
    );
    let (result, refined_camera) = match result {
        Ok(result) => result,
        Err(error) => {
            poses.clone_from_slice(&poses_before);
            for (track, point) in tracks.iter_mut().zip(points_before) {
                track.position = point;
            }
            return Err(error);
        }
    };
    for (track, point) in tracks.iter_mut().zip(points) {
        if let Some(point) = point {
            track.position = point;
        }
    }
    Ok((result, refined_camera))
}

/// COLMAP `IncrementalMapper::AdjustLocalBundle`. After registering `new_image`,
/// bundle-adjust only it and its `local_ba_num_images` most-covisible registered
/// neighbours (sharing the most triangulated tracks) plus the points they see —
/// every *other* registered image that observes one of those points is added as a
/// **fixed** pose, so it constrains the local solve without being moved. This
/// keeps the freshly grown geometry tight after every step at a fraction of a
/// global solve's cost, the schedule that lets COLMAP hold sub-centimetre
/// accuracy as the reconstruction grows. Poses/points outside the variable set
/// are untouched.
pub(super) fn adjust_local_bundle(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    new_image: usize,
) -> Result<(), BaError> {
    // Covisible registered images: how many triangulated tracks each shares with
    // the newly registered one.
    let mut covis: HashMap<usize, usize> = HashMap::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        if !track
            .iter()
            .any(|&(img, _)| img == new_image && poses[img].is_some())
        {
            continue;
        }
        for &(img, _) in track {
            if img != new_image && poses[img].is_some() {
                *covis.entry(img).or_insert(0) += 1;
            }
        }
    }
    let mut neighbours: Vec<(usize, usize)> = covis.into_iter().collect();
    // Most-covisible first; break ties by index for determinism.
    neighbours.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut variable: HashSet<usize> = neighbours
        .into_iter()
        .take(config.local_ba_num_images)
        .map(|(img, _)| img)
        .collect();
    variable.insert(new_image);

    bundle_adjust_local(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        &variable,
    )
}

#[allow(clippy::too_many_arguments)]
/// With every camera and every pre-existing landmark fixed, refine only the
/// landmarks created by a tentative structure-less insertion. This is the
/// bounded local-submap solve used after projecting the new camera into the
/// independent relative-geometry feasible region.
pub(super) fn refine_structureless_new_landmarks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    new_image: usize,
    preexisting_points: &[bool],
) -> Result<(), BaError> {
    let mut landmark_ids = Vec::new();
    let mut used_images = HashSet::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if preexisting_points.get(track_id).copied().unwrap_or(false)
            || track_point[track_id].is_none()
            || !track.iter().any(|&(image, kp)| {
                image == new_image
                    && poses[image].is_some()
                    && features[image].keypoints.get(kp).is_some()
            })
        {
            continue;
        }
        let observers: Vec<usize> = track
            .iter()
            .filter_map(|&(image, kp)| {
                (poses[image].is_some() && features[image].keypoints.get(kp).is_some())
                    .then_some(image)
            })
            .collect();
        if observers.len() < 2 {
            continue;
        }
        used_images.extend(observers);
        landmark_ids.push(track_id);
    }
    if landmark_ids.is_empty() {
        return Ok(());
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    for image in used_images {
        ba.add_pose(image as u64, poses[image].clone().unwrap());
        ba.fix_pose(image as u64);
    }
    for &track_id in &landmark_ids {
        ba.add_landmark(track_id as u64, track_point[track_id].unwrap());
        for &(image, kp) in &tracks[track_id] {
            if !ba.poses.contains_key(&(image as u64)) {
                continue;
            }
            if let Some(pixel) = features[image].keypoints.get(kp).copied() {
                ba.add_observation(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: pixel,
                });
            }
        }
    }
    ba.optimize(&config.ba_config)?;
    for track_id in landmark_ids {
        if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
            track_point[track_id] = Some(*refined);
        }
    }
    Ok(())
}

/// Bundle-adjust a chosen `variable` set of poses plus every triangulated track
/// they observe. Other registered images observing those tracks join as fixed
/// poses (constraints). The gauge: with ≥2 fixed observers their baseline pins
/// the 7-DoF monocular gauge for free; otherwise (an early, loosely connected
/// neighbourhood) the variable set's own anchor + farthest pose are fixed, as in
/// the global solve. Only variable poses and the solved landmarks are written back.
pub(super) fn bundle_adjust_local(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    variable: &HashSet<usize>,
) -> Result<(), BaError> {
    let timing = std::env::var_os("VISLOC_SFM_LOCAL_BA_TIMING").is_some();
    let build_started = std::time::Instant::now();
    let mut ba = BundleAdjustment::new(camera.clone());

    // Landmarks touching ≥1 variable image, and the images that participate.
    let mut used: HashSet<usize> = HashSet::new();
    let mut lm_ids: Vec<usize> = Vec::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        let mut obs_images: Vec<usize> = Vec::new();
        let mut touches_variable = false;
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if features[image].keypoints.get(kp).is_none() {
                continue;
            }
            obs_images.push(image);
            if variable.contains(&image) {
                touches_variable = true;
            }
        }
        if obs_images.len() < 2 || !touches_variable {
            continue;
        }
        for image in obs_images {
            used.insert(image);
        }
        lm_ids.push(track_id);
    }
    if lm_ids.is_empty() {
        return Ok(());
    }

    for &image in &used {
        ba.add_pose(image as u64, poses[image].clone().unwrap());
        if !variable.contains(&image) {
            ba.fix_pose(image as u64);
        }
    }

    // Need ≥2 fixed poses to pin metric scale; otherwise fix the variable gauge.
    let n_fixed = used.iter().filter(|i| !variable.contains(i)).count();
    if n_fixed < 2 {
        let var_used: Vec<usize> = used
            .iter()
            .copied()
            .filter(|i| variable.contains(i))
            .collect();
        fix_monocular_scale_gauge(&mut ba, poses, &var_used);
    }

    for &track_id in &lm_ids {
        ba.add_landmark(track_id as u64, track_point[track_id].unwrap());
        for &(image, kp) in &tracks[track_id] {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                ba.add_observation(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: px,
                });
            }
        }
    }
    if let Some(path) = std::env::var_os("VISLOC_SFM_BA_DUMP_LOCAL") {
        // Replayable copy of this local solve (overwritten by every call).
        if let Err(error) =
            crate::ba_problem_io::write_ba_problem(&ba, None, std::path::Path::new(&path))
        {
            eprintln!("sfm-ba-dump: failed: {error}");
        }
    }
    let build_seconds = build_started.elapsed().as_secs_f64();
    let optimize_started = std::time::Instant::now();
    let local_config = BaConfig {
        relative_cost_tolerance: config
            .local_ba_relative_cost_tolerance
            .or(config.ba_config.relative_cost_tolerance),
        ..config.ba_config
    };
    let iterations = match crate::ba_accel::ba_accelerator()
        .and_then(|a| a.optimize(&mut ba, &local_config, crate::ba_accel::BaScope::Local))
    {
        Some(result) => result?.iterations.len(),
        None => ba.optimize(&local_config)?.iterations.len(),
    };
    if timing {
        eprintln!(
            "sfm-local-ba: build={build_seconds:.4}s optimize={:.4}s iterations={iterations} poses={} variable={} landmarks={} observations={}",
            optimize_started.elapsed().as_secs_f64(),
            ba.poses.len(),
            variable.len(),
            ba.landmarks.len(),
            ba.observations.len()
        );
    }

    for &image in &used {
        if variable.contains(&image) {
            if let Some(refined) = ba.poses.get(&(image as u64)) {
                poses[image] = Some(refined.clone());
            }
        }
    }
    for &track_id in &lm_ids {
        if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
            track_point[track_id] = Some(*refined);
        }
    }
    Ok(())
}

/// Pin the 7-DoF monocular gauge (6 rigid + scale) by fixing two of `candidates`:
/// the lowest-index pose (rigid anchor) and the one farthest from it (the scale
/// anchor — longest, best-conditioned baseline). Mirrors the global solve's gauge
/// handling; used by a local solve that lacks two fixed-observer poses of its own.
fn fix_monocular_scale_gauge(
    ba: &mut BundleAdjustment,
    poses: &[Option<Pose>],
    candidates: &[usize],
) {
    let Some(&anchor) = candidates.iter().min() else {
        return;
    };
    ba.fix_pose(anchor as u64);
    let anchor_center = poses[anchor]
        .as_ref()
        .unwrap()
        .camera_to_world()
        .translation;
    let mut farthest = None;
    let mut best_d2 = 0.0;
    for &image in candidates {
        if image == anchor {
            continue;
        }
        let d2 = (poses[image].as_ref().unwrap().camera_to_world().translation - anchor_center)
            .norm_squared();
        if d2 > best_d2 {
            best_d2 = d2;
            farthest = Some(image);
        }
    }
    if let Some(scale_anchor) = farthest {
        ba.fix_pose(scale_anchor as u64);
    }
}

/// COLMAP `IncrementalMapper::IterativeGlobalRefinement`: a global BA, then a loop
/// of {re-triangulate/complete tracks, filter outliers, global BA} until the
/// changed-observation fraction falls below `global_ba_change_rate` (or
/// `global_ba_max_refinements` rounds run). Re-triangulation is forced on here
/// regardless of `config.retriangulate` — completing tracks between global solves
/// is integral to COLMAP's schedule, not the opt-in density lever of the simple
/// path.
pub(super) fn iterative_global_refinement(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &mut [Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<BaResult, BaError> {
    // Refine intrinsics on each global solve when enabled; the refined camera is
    // carried forward so the next round's filter / re-triangulation / BA all use
    // it (and the caller reads the final camera back from `*camera`).
    let refine = config.refine_intrinsics;
    let run_ba = |cam: &mut Camera,
                  tr: &[Vec<(usize, usize)>],
                  p: &mut [Option<Pose>],
                  tp: &mut [Option<Point3<f64>>]|
     -> Result<BaResult, BaError> {
        let (res, refined) = run_bundle_adjustment(cam, features, tr, config, p, tp, refine)?;
        if let Some(c) = refined {
            *cam = c;
        }
        Ok(res)
    };

    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: final global refinement begin registered={} points={} observations={} max_followup_rounds={}",
            poses.iter().filter(|pose| pose.is_some()).count(),
            track_point.iter().filter(|point| point.is_some()).count(),
            count_observations(tracks, poses, track_point),
            config.global_ba_max_refinements,
        );
    }
    let mut ba_started = std::time::Instant::now();
    let mut result = run_ba(camera, tracks, poses, track_point)?;
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: final global BA round=0 completed seconds={:.3}",
            ba_started.elapsed().as_secs_f64(),
        );
    }
    for followup in 0..config.global_ba_max_refinements {
        let round = followup + 1;
        let total_obs = count_observations(tracks, poses, track_point).max(1);
        // Filter outlier observations, then complete/re-triangulate tracks the
        // tightened frame can now place. Completing between solves is integral to
        // the schedule — it gives the next global BA more constraints and, on this
        // metric video, measurably beats filter-only (1.64 cm vs 2.21 cm); the
        // forward-motion low-parallax churn it induces against the filter is the
        // price, and the track-density ceiling it leaves is the next lever.
        let support_before = sfm_ba_debug_enabled().then(|| {
            (
                poses.iter().filter(|pose| pose.is_some()).count(),
                track_point.iter().filter(|point| point.is_some()).count(),
                count_observations(tracks, poses, track_point),
            )
        });
        let mut changed =
            filter_outlier_observations(camera, features, tracks, config, poses, track_point);
        let support_after_filter = sfm_ba_debug_enabled().then(|| {
            (
                poses.iter().filter(|pose| pose.is_some()).count(),
                track_point.iter().filter(|point| point.is_some()).count(),
                count_observations(tracks, poses, track_point),
            )
        });
        changed += retriangulate_tracks(camera, features, tracks, config, poses, track_point);
        let support_after_retriangulation = sfm_ba_debug_enabled().then(|| {
            (
                poses.iter().filter(|pose| pose.is_some()).count(),
                track_point.iter().filter(|point| point.is_some()).count(),
                count_observations(tracks, poses, track_point),
            )
        });
        let change_rate = changed as f64 / total_obs as f64;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: final global refinement round={round} changed={changed}/{total_obs} rate={change_rate:.6} threshold={:.6}",
                config.global_ba_change_rate,
            );
        }
        if let (Some(before), Some(after_filter), Some(after_retriangulation)) = (
            support_before,
            support_after_filter,
            support_after_retriangulation,
        ) {
            eprintln!(
                "sfm-debug-ba-support: stage=final_refinement round={round} \
                 before=({},{},{}) after_filter=({},{},{}) \
                 after_retriangulation=({},{},{}) changed={changed}",
                before.0,
                before.1,
                before.2,
                after_filter.0,
                after_filter.1,
                after_filter.2,
                after_retriangulation.0,
                after_retriangulation.1,
                after_retriangulation.2,
            );
        }
        if change_rate < config.global_ba_change_rate {
            if sfm_debug_enabled() {
                eprintln!("sfm-debug: final global refinement converged before BA round={round}");
            }
            break;
        }
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: final global BA round={round} begin");
        }
        ba_started = std::time::Instant::now();
        result = run_ba(camera, tracks, poses, track_point)?;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: final global BA round={round} completed seconds={:.3}",
                ba_started.elapsed().as_secs_f64(),
            );
        }
    }
    Ok(result)
}

/// In-growth global refinement, used during the seed search where `tracks` is
/// shared read-only across trials: global BA, then up to a couple rounds of
/// {re-triangulate/complete, global BA} while it keeps completing tracks. The
/// completion is what keeps registration moving — a freshly tightened global
/// frame lets [`retriangulate_tracks`] triangulate tracks the narrow growth-time
/// baseline had missed, and those new 3D points give the next PnP enough
/// 2D-3D matches to register (without it, registration stalls well short of full
/// coverage and the trajectory develops ATE-wrecking gaps). The track-membership
/// *filter* (which would mutate the shared tracks) is deferred to the final
/// [`iterative_global_refinement`] after a seed is committed.
pub(super) fn growth_global_refinement(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<(), BaError> {
    // When intrinsics refinement is on, co-evolve them in these periodic global
    // passes (COLMAP's IterativeGlobalRefinement keeps the camera moving with the
    // structure, so a wrong focal is corrected while the model is still small
    // enough to expose it — the well-conditioned global solve, not the narrow
    // per-registration local one, is where the focal is observable). Otherwise the
    // intrinsics stay fixed and the refined slot is always None.
    let refine = config.refine_intrinsics;
    let run_global = |cam: &mut Camera,
                      p: &mut [Option<Pose>],
                      tp: &mut [Option<Point3<f64>>]|
     -> Result<(), BaError> {
        let (_, refined) = run_bundle_adjustment(cam, features, tracks, config, p, tp, refine)?;
        if let Some(c) = refined {
            *cam = c;
        }
        Ok(())
    };

    run_global(camera, poses, track_point)?;
    for _ in 0..config.global_ba_max_refinements.min(2) {
        let changed = retriangulate_tracks(camera, features, tracks, config, poses, track_point);
        if changed == 0 {
            break;
        }
        run_global(camera, poses, track_point)?;
    }
    Ok(())
}

/// Summary of an optional final fixed-support BA polish.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FinalBaPolishStats {
    pub(crate) requested_iterations: usize,
    pub(crate) accepted: bool,
    pub(crate) initial_sse: f64,
    pub(crate) final_sse: f64,
    pub(crate) solver_iterations: usize,
    pub(crate) accepted_steps: usize,
    pub(crate) rejected_steps: usize,
    pub(crate) converged: bool,
    pub(crate) max_pose_step: f64,
    pub(crate) max_landmark_step: f64,
    pub(crate) final_lambda: f64,
    /// Number of non-empty final BA landmarks (the output-track support).
    pub(crate) tracks_before: usize,
    pub(crate) tracks_after: usize,
    pub(crate) observations_before: usize,
    pub(crate) observations_after: usize,
}

/// Run an optional fixed-support BA solve on the final support without allowing
/// the solve to change track membership, observation membership, or camera
/// intrinsics. An explicit `final_ba_polish_iterations` uses the historical
/// pure-L2 objective; `geometry_weighted_ba` instead keeps the ordinary robust
/// objective and changes only the fixed pre-BA observation weights. The
/// ordinary robust/refinement schedule has already completed before this
/// function is called. We snapshot all mutable state and commit only a finite,
/// non-increasing weighted-cost result.
pub(crate) fn final_fixed_support_ba_polish(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<(FinalBaPolishStats, Option<BaResult>), BaError> {
    let requested_iterations = if config.final_ba_polish_iterations > 0 {
        config.final_ba_polish_iterations
    } else if config.geometry_weighted_ba {
        // The geometry-weighted mode is itself a final fixed-support solve. If
        // no separate polish cap is supplied, use the ordinary BA cap rather
        // than inventing another tuning knob.
        config.ba_config.max_iterations
    } else {
        0
    };
    let tracks_before = track_point.iter().filter(|point| point.is_some()).count();
    let observations_before = count_observations(tracks, poses, track_point);
    let mut stats = FinalBaPolishStats {
        requested_iterations,
        accepted: false,
        initial_sse: f64::NAN,
        final_sse: f64::NAN,
        solver_iterations: 0,
        accepted_steps: 0,
        rejected_steps: 0,
        converged: false,
        max_pose_step: 0.0,
        max_landmark_step: 0.0,
        final_lambda: 0.0,
        tracks_before,
        tracks_after: tracks_before,
        observations_before,
        observations_after: observations_before,
    };
    if requested_iterations == 0 {
        return Ok((stats, None));
    }

    let poses_before = poses.to_vec();
    let track_point_before = track_point.to_vec();
    let mut polish_config = config.clone();
    polish_config.refine_intrinsics = false;
    polish_config.ba_config = if config.final_ba_polish_iterations > 0 {
        // Preserve the historical explicit fixed-support polish contract:
        // pure L2 with the caller's requested cap.
        BaConfig {
            max_iterations: requested_iterations,
            robust_kernel: RobustKernel::None,
            refine_intrinsics: false,
            ..config.ba_config
        }
    } else {
        // Geometry weighting is a controlled observation-information A/B. Keep
        // the ordinary final objective (usually Huber) so the only changed
        // factor is the fixed pre-BA track weight.
        BaConfig {
            max_iterations: requested_iterations,
            refine_intrinsics: false,
            ..config.ba_config
        }
    };

    let result = match run_bundle_adjustment_impl(
        camera,
        features,
        tracks,
        &polish_config,
        poses,
        track_point,
        false,
        config.geometry_weighted_ba,
    ) {
        Ok(result) => result,
        Err(error) => {
            poses.clone_from_slice(&poses_before);
            track_point.clone_from_slice(&track_point_before);
            return Err(error);
        }
    };
    let (result, _) = result;
    stats.initial_sse = result.initial_cost;
    stats.final_sse = result.final_cost;
    stats.solver_iterations = result.iterations.len();
    stats.accepted_steps = result
        .iterations
        .iter()
        .filter(|iteration| iteration.step_accepted)
        .count();
    stats.rejected_steps = result.iterations.len().saturating_sub(stats.accepted_steps);
    stats.converged = result.converged;
    if let Some(last) = result.iterations.last() {
        stats.max_pose_step = last.max_pose_step;
        stats.max_landmark_step = last.max_landmark_step;
        stats.final_lambda = last.lambda;
    }

    let finite_state = poses.iter().all(|pose| {
        pose.as_ref().is_none_or(|pose| {
            pose.world_to_camera
                .translation
                .iter()
                .all(|value| value.is_finite())
                && pose
                    .world_to_camera
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        })
    }) && track_point.iter().all(|point| {
        point
            .as_ref()
            .is_none_or(|point| point.coords.iter().all(|value| value.is_finite()))
    });
    stats.tracks_after = track_point.iter().filter(|point| point.is_some()).count();
    stats.observations_after = count_observations(tracks, poses, track_point);
    let support_unchanged = stats.tracks_after == tracks_before
        && stats.observations_after == observations_before
        && poses
            .iter()
            .zip(poses_before.iter())
            .all(|(after, before)| after.is_some() == before.is_some())
        && track_point
            .iter()
            .zip(track_point_before.iter())
            .all(|(after, before)| after.is_some() == before.is_some());
    let cost_nonincreasing = stats.initial_sse.is_finite()
        && stats.final_sse.is_finite()
        && stats.final_sse <= stats.initial_sse;
    stats.accepted = finite_state && support_unchanged && cost_nonincreasing;

    if !stats.accepted {
        poses.clone_from_slice(&poses_before);
        track_point.clone_from_slice(&track_point_before);
        stats.observations_after = observations_before;
    }
    if sfm_ba_debug_enabled() {
        eprintln!(
            concat!(
                "sfm-debug-ba-polish: requested_iterations={} accepted={} ",
                "geometry_weighted={} ",
                "support_tracks={}=>{} support_observations={}=>{} ",
                "initial_sse={:.9e} final_sse={:.9e} solver_iterations={} ",
                "accepted_steps={} rejected_steps={} converged={} ",
                "last_step=({:.3e},{:.3e}) last_lambda={:.3e}"
            ),
            stats.requested_iterations,
            stats.accepted,
            config.geometry_weighted_ba,
            stats.tracks_before,
            stats.tracks_after,
            stats.observations_before,
            stats.observations_after,
            stats.initial_sse,
            stats.final_sse,
            stats.solver_iterations,
            stats.accepted_steps,
            stats.rejected_steps,
            stats.converged,
            stats.max_pose_step,
            stats.max_landmark_step,
            stats.final_lambda,
        );
    }
    Ok((stats, stats.accepted.then_some(result)))
}

/// Total triangulated observations: for every track with a 3D point, the number
/// of its registered observations. The denominator for the refinement-loop
/// change-rate stop test.
pub(super) fn count_observations(
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> usize {
    let mut n = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        n += track
            .iter()
            .filter(|&&(img, _)| poses[img].is_some())
            .count();
    }
    n
}

/// Whether a plain-growth periodic BA is due at the current registration
/// boundary. Keeping this decision in a pure helper makes the opt-in schedule
/// explicit and testable: `minimum_registered == 0` is exactly the historical
/// behavior, while a positive minimum only defers a due solve and never
/// disables the final BA.
pub(super) const fn periodic_ba_due(
    ba_every: usize,
    minimum_registered: usize,
    registrations_since_ba: usize,
    registered: usize,
) -> bool {
    ba_every > 0
        && registrations_since_ba >= ba_every
        && (minimum_registered == 0 || registered >= minimum_registered)
}

/// Number of post-BA filter/retriangulation rounds for the plain final pass.
/// A disabled final BA also disables this post-BA stage: running it anyway can
/// discover outliers and launch a second global solve after the caller asked
/// for a growth-only result. The default final-BA schedule is unchanged.
pub(super) fn simple_final_refinement_rounds(config: &IncrementalSfmConfig) -> usize {
    if !config.final_global_ba {
        0
    } else if config.retriangulate {
        config.track_filter_iterations.max(1)
    } else {
        config.track_filter_iterations
    }
}

/// Numerical cutoff for the geometry-only point block condition proxy used by
/// [`ba_landmark_should_freeze`].  A condition number above 1e8 leaves fewer
/// than eight reliable decimal digits in a 3-D point solve, which is the
/// conventional double-precision boundary for treating a normal-equation
/// block as ill-conditioned.  This is deliberately a fixed numerical
/// criterion, not a dataset/accuracy knob.
pub(super) const BA_POINT_BLOCK_MAX_CONDITION: f64 = 1.0e8;

/// Geometry captured at the beginning of one BA solve for one landmark.  The
/// point-block condition is the condition number of the accumulated 3×3
/// landmark Jacobian block (with the configured robust observation weights),
/// before any LM step.  It is a local Schur-block proxy: the full reduced
/// camera system is intentionally not approximated here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct LandmarkBaGeometry {
    pub(super) track_length: usize,
    pub(super) baseline_depth_ratio: f64,
    pub(super) max_parallax_deg: f64,
    pub(super) median_reprojection_px: f64,
    pub(super) point_condition: f64,
    pub(super) point_min_eigenvalue: f64,
    pub(super) point_max_eigenvalue: f64,
    pub(super) invalid_depth: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LandmarkBaDiagnostic {
    id: u64,
    geometry: LandmarkBaGeometry,
    displacement: f64,
    excluded: bool,
}

/// Projection Jacobian with respect to a camera-frame 3-D point.  The
/// pinhole/no-distortion path mirrors the BA normal-equation Jacobian exactly.
/// For a camera carrying radial distortion, a deterministic central difference
/// keeps this diagnostic conservative without pretending that the pinhole
/// formula is exact for a distorted projection.
pub(super) fn ba_point_projection_jacobian(
    camera: &Camera,
    point_camera: &Point3<f64>,
) -> Option<Matrix2x3<f64>> {
    let (fx, fy, _, _) = camera.intrinsics()?;
    if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
        return None;
    }
    let has_distortion = camera
        .radial_distortion()
        .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0)
        || camera.tangential_distortion().is_some();
    if !has_distortion {
        let z_inv = 1.0 / point_camera.z;
        let mut jacobian = Matrix2x3::<f64>::zeros();
        jacobian[(0, 0)] = fx * z_inv;
        jacobian[(0, 2)] = -fx * point_camera.x * z_inv * z_inv;
        jacobian[(1, 1)] = fy * z_inv;
        jacobian[(1, 2)] = -fy * point_camera.y * z_inv * z_inv;
        return jacobian
            .iter()
            .all(|value| value.is_finite())
            .then_some(jacobian);
    }

    let epsilon = (point_camera.coords.norm().max(1.0) * 1.0e-6).max(1.0e-8);
    let mut jacobian = Matrix2x3::<f64>::zeros();
    for axis in 0..3 {
        let mut plus = point_camera.coords;
        let mut minus = point_camera.coords;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        let plus = camera.project(&Point3::from(plus))?;
        let minus = camera.project(&Point3::from(minus))?;
        if !plus.coords.iter().all(|value| value.is_finite())
            || !minus.coords.iter().all(|value| value.is_finite())
        {
            return None;
        }
        let derivative = (plus - minus) / (2.0 * epsilon);
        jacobian[(0, axis)] = derivative.x;
        jacobian[(1, axis)] = derivative.y;
    }
    jacobian
        .iter()
        .all(|value| value.is_finite())
        .then_some(jacobian)
}

/// Compute the fixed, pre-BA geometry used both by the diagnostic dump and by
/// the opt-in landmark freeze gate.  Invalid/behind-camera observations mark
/// the point invalid; valid observations still contribute to the condition
/// proxy so the report explains how the gate was reached.
pub(super) fn ba_landmark_geometry(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    track: &[(usize, usize)],
    point: &Point3<f64>,
    kernel: &RobustKernel,
) -> LandmarkBaGeometry {
    let point_finite = point.coords.iter().all(|value| value.is_finite());
    let mut registered_images = HashSet::new();
    let mut centres = Vec::new();
    let mut depths = Vec::new();
    let mut reprojections = Vec::new();
    let mut hessian = Matrix3::<f64>::zeros();
    let mut invalid_depth = !point_finite;

    if point_finite {
        for &(image, keypoint) in track {
            let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                continue;
            };
            registered_images.insert(image);
            let centre = pose.camera_center_world().coords;
            if centre.iter().all(|value| value.is_finite()) {
                centres.push(centre);
            }
            let point_camera = pose.transform_world_point(point);
            if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
                invalid_depth = true;
                continue;
            }
            depths.push(point_camera.z);
            let Some(pixel) = features
                .get(image)
                .and_then(|feature| feature.keypoints.get(keypoint))
                .copied()
            else {
                continue;
            };
            let Some(error) = reprojection_error_px(camera, pose, point, &pixel) else {
                invalid_depth = true;
                continue;
            };
            if error.is_finite() {
                reprojections.push(error);
            }
            let Some(j_projection) = ba_point_projection_jacobian(camera, &point_camera) else {
                invalid_depth = true;
                continue;
            };
            let rotation = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let j_landmark = j_projection * rotation;
            let weight = kernel.weight(error * error);
            if weight.is_finite() && weight > 0.0 {
                hessian += weight * (j_landmark.transpose() * j_landmark);
            }
        }
    }

    depths.sort_by(f64::total_cmp);
    reprojections.sort_by(f64::total_cmp);
    let median_depth = depths
        .get(depths.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::NAN);
    let median_reprojection = reprojections
        .get(reprojections.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::NAN);
    let mut max_baseline = 0.0f64;
    for i in 0..centres.len() {
        for j in (i + 1)..centres.len() {
            let baseline = (centres[i] - centres[j]).norm();
            if baseline.is_finite() {
                max_baseline = max_baseline.max(baseline);
            }
        }
    }
    let baseline_depth_ratio = if median_depth.is_finite() && median_depth > 0.0 {
        max_baseline / median_depth
    } else {
        f64::NAN
    };
    let max_parallax_deg = if point_finite {
        track_max_parallax(poses, track, point).to_degrees()
    } else {
        f64::NAN
    };

    let eigenvalues = hessian.symmetric_eigen().eigenvalues;
    let point_min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let point_max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let point_condition = if point_min_eigenvalue.is_finite()
        && point_max_eigenvalue.is_finite()
        && point_min_eigenvalue > 0.0
        && point_max_eigenvalue > 0.0
    {
        point_max_eigenvalue / point_min_eigenvalue
    } else {
        f64::INFINITY
    };

    LandmarkBaGeometry {
        track_length: registered_images.len(),
        baseline_depth_ratio,
        max_parallax_deg,
        median_reprojection_px: median_reprojection,
        point_condition,
        point_min_eigenvalue,
        point_max_eigenvalue,
        invalid_depth,
    }
}

pub(super) fn ba_landmark_is_ill_conditioned(
    geometry: &LandmarkBaGeometry,
    min_parallax_deg: f64,
) -> bool {
    geometry.invalid_depth
        || !geometry.point_condition.is_finite()
        || geometry.point_condition > BA_POINT_BLOCK_MAX_CONDITION
        || (min_parallax_deg.is_finite()
            && min_parallax_deg > 0.0
            && geometry.max_parallax_deg.is_finite()
            && geometry.max_parallax_deg < min_parallax_deg)
}

/// A weak point with a small reprojection residual can still be a useful
/// camera observation, whereas a weak point whose current residual is already
/// outside the ordinary reprojection gate is not a trustworthy fixed camera
/// constraint.  The opt-in safeguard therefore excludes only the latter; the
/// classification is computed once before the solve and never changes during
/// LM iterations.
pub(super) fn ba_landmark_should_exclude(
    geometry: &LandmarkBaGeometry,
    min_parallax_deg: f64,
    max_reprojection_error_px: f64,
) -> bool {
    if !ba_landmark_is_ill_conditioned(geometry, min_parallax_deg) {
        return false;
    }
    geometry.invalid_depth
        || !geometry.median_reprojection_px.is_finite()
        || !max_reprojection_error_px.is_finite()
        || geometry.median_reprojection_px > max_reprojection_error_px
}

fn sfm_pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    (variance_x > 0.0 && variance_y > 0.0 && covariance.is_finite())
        .then_some(covariance / (variance_x * variance_y).sqrt())
}

/// Print a compact per-landmark report and correlations for one BA solve.
/// Rows are sorted by the *observed* post-solve displacement, making the
/// potentially tiny set driving an extreme point step immediately visible.
fn sfm_debug_ba_landmarks(records: &[LandmarkBaDiagnostic], min_parallax_deg: f64) {
    if records.is_empty() {
        return;
    }
    let mut rows = records.to_vec();
    rows.sort_by(|a, b| {
        b.displacement
            .total_cmp(&a.displacement)
            .then(a.id.cmp(&b.id))
    });
    let displacements: Vec<f64> = rows
        .iter()
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .collect();
    let total_displacement = displacements.iter().sum::<f64>();
    let top_n = rows.len().min(10);
    let top_displacement = rows
        .iter()
        .take(top_n)
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let low_parallax = rows.iter().filter(|row| {
        min_parallax_deg.is_finite()
            && min_parallax_deg > 0.0
            && row.geometry.max_parallax_deg.is_finite()
            && row.geometry.max_parallax_deg < min_parallax_deg
    });
    let low_parallax_count = low_parallax.clone().count();
    let low_parallax_displacement = low_parallax
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let near_condition = rows
        .iter()
        .filter(|row| row.geometry.point_condition > BA_POINT_BLOCK_MAX_CONDITION);
    let near_condition_count = near_condition.clone().count();
    let near_condition_displacement = near_condition
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let invalid_depth_count = rows.iter().filter(|row| row.geometry.invalid_depth).count();
    let mut median_displacement = displacements;
    let median_displacement = sfm_oracle_median(&mut median_displacement);
    let finite_rows: Vec<&LandmarkBaDiagnostic> = rows
        .iter()
        .filter(|row| {
            row.displacement.is_finite()
                && row.geometry.baseline_depth_ratio.is_finite()
                && row.geometry.max_parallax_deg.is_finite()
                && row.geometry.median_reprojection_px.is_finite()
                && row.geometry.point_condition.is_finite()
                && row.geometry.point_condition > 0.0
        })
        .collect();
    let correlation = |value: fn(&LandmarkBaGeometry) -> f64| {
        let mut x = Vec::with_capacity(finite_rows.len());
        let mut y = Vec::with_capacity(finite_rows.len());
        for row in &finite_rows {
            let feature = value(&row.geometry);
            if feature.is_finite() && feature > 0.0 && row.displacement > 0.0 {
                x.push(feature.ln());
                y.push(row.displacement.ln());
            }
        }
        sfm_pearson(&x, &y).unwrap_or(f64::NAN)
    };
    let excluded = rows.iter().filter(|row| row.excluded).count();
    eprintln!(
        concat!(
            "sfm-debug-ba-landmarks: count={} excluded={} condition_limit={:.3e} ",
            "disp_max={:.3e} disp_median={:.3e} top10_fraction={:.6} ",
            "low_parallax(<{:.3}deg)={}/{} fraction={:.6} ",
            "near_condition(>{:.3e})={}/{} fraction={:.6} invalid_depth={} ",
            "corr_log_disp=(baseline_depth={:.4},parallax_deg={:.4},reproj={:.4},condition={:.4})"
        ),
        rows.len(),
        excluded,
        BA_POINT_BLOCK_MAX_CONDITION,
        rows.first().map_or(f64::NAN, |row| row.displacement),
        median_displacement,
        if total_displacement > 0.0 {
            top_displacement / total_displacement
        } else {
            f64::NAN
        },
        min_parallax_deg,
        low_parallax_count,
        rows.len(),
        if total_displacement > 0.0 {
            low_parallax_displacement / total_displacement
        } else {
            f64::NAN
        },
        BA_POINT_BLOCK_MAX_CONDITION,
        near_condition_count,
        rows.len(),
        if total_displacement > 0.0 {
            near_condition_displacement / total_displacement
        } else {
            f64::NAN
        },
        invalid_depth_count,
        correlation(|geometry| geometry.baseline_depth_ratio),
        correlation(|geometry| geometry.max_parallax_deg),
        correlation(|geometry| geometry.median_reprojection_px),
        correlation(|geometry| geometry.point_condition),
    );
    for row in rows.iter().take(10) {
        let geometry = row.geometry;
        eprintln!(
            concat!(
                "sfm-debug-ba-landmark: track={} excluded={} len={} baseline_depth={:.6e} ",
                "parallax_deg={:.6e} reproj_px={:.6e} condition={:.6e} ",
                "eig=({:.6e},{:.6e}) displacement={:.6e} invalid_depth={}"
            ),
            row.id,
            row.excluded,
            geometry.track_length,
            geometry.baseline_depth_ratio,
            geometry.max_parallax_deg,
            geometry.median_reprojection_px,
            geometry.point_condition,
            geometry.point_min_eigenvalue,
            geometry.point_max_eigenvalue,
            row.displacement,
            geometry.invalid_depth,
        );
    }
}

/// Widest angle (radians) subtended at `point` by any pair of registered camera
/// centres that observe it — the post-BA triangulation angle. Zero if fewer than
/// two registered views remain.
pub(super) fn track_max_parallax(
    poses: &[Option<Pose>],
    track: &[(usize, usize)],
    point: &Point3<f64>,
) -> f64 {
    let dirs: Vec<Vector3<f64>> = track
        .iter()
        .filter_map(|&(image, _)| poses[image].as_ref())
        .filter_map(|pose| {
            let v = pose.camera_to_world().translation - point.coords;
            (v.norm() > f64::EPSILON).then(|| v.normalize())
        })
        .collect();
    let mut max_angle = 0.0;
    for a in 0..dirs.len() {
        for b in (a + 1)..dirs.len() {
            let angle = dirs[a].dot(&dirs[b]).clamp(-1.0, 1.0).acos();
            if angle > max_angle {
                max_angle = angle;
            }
        }
    }
    max_angle
}

/// Reprojection error (px) of `point_world` against pixel `px` in a camera.
/// `None` if the point is behind the camera or projection is degenerate.
pub(crate) fn reprojection_error_px(
    camera: &Camera,
    pose: &Pose,
    point_world: &Point3<f64>,
    px: &Point2<f64>,
) -> Option<f64> {
    let cam = pose.transform_world_point(point_world);
    if !cam.z.is_finite() || cam.z <= 0.0 {
        return None;
    }
    let projected = camera.project(&cam)?;
    Some((projected - px).norm())
}
