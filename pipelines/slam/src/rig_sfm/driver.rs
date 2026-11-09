//! The `incremental_rig_sfm` entry point and its per-frame registration
//! correspondence helpers.

use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct CachedRigCorrespondence {
    pub(super) sensor_index: usize,
    pub(super) point2d: Point2<f64>,
    track_index: Option<usize>,
    pub(super) direct_point3d: Option<Point3<f64>>,
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
    validate_inputs(rig, frames, features, pairwise, config)?;

    // Keep the historical all-pairs track construction path entirely
    // untouched unless the caller explicitly opts into the registration-time
    // correspondence map.  In particular, do not build the CSR as a preview
    // side effect of the default path: its allocation and its deterministic
    // duplicate policy are part of the opt-in behavior only.
    if config.dynamic_correspondence_tracking {
        return incremental_rig_sfm_dynamic(rig, frames, features, pairwise, config);
    }

    let rig_timing = rig_timing_enabled();
    let rig_timing_started = Instant::now();
    let trace_registration = std::env::var_os("VISLOC_SFM_TRACE_REGISTRATION").is_some();
    let image_assignment = image_assignment(frames, features.len());
    let temporal_support_bin_frames = temporal_support_bin_frames_from_env();
    let registration_prefix = config
        .deferred_registration_pair_prefix
        .unwrap_or(pairwise.len());
    let mapping_pairwise = &pairwise[..registration_prefix];
    let deferred_pairwise = config
        .deferred_retriangulation_pair_prefix
        .map_or(&pairwise[registration_prefix..], |prefix| {
            &pairwise[registration_prefix..prefix]
        });
    let retriangulation_pairwise = config
        .deferred_retriangulation_pair_prefix
        .map_or(deferred_pairwise, |prefix| &pairwise[prefix..]);
    if rig_timing {
        eprintln!(
            "rig-timing: track-build-start frames={} images={} pairs={} matches={}",
            frames.len(),
            features.len(),
            mapping_pairwise.len(),
            mapping_pairwise
                .iter()
                .map(|p| p.matches.len())
                .sum::<usize>(),
        );
    }
    let track_build_started = Instant::now();
    let track_output =
        build_rig_track_output(features, mapping_pairwise, &image_assignment, config);
    let track_build_stats = track_output.stats;
    let conflicting_components = track_output.conflicting_components;
    if rig_timing {
        let largest_conflict = conflicting_components
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        let total_conflict_obs: usize = conflicting_components.iter().map(Vec::len).sum();
        eprintln!(
            "rig-timing: track-build-done elapsed={:.3}s tracks={} conflicting_components={} \
             conflicting_observations={} largest_conflicting_component={}",
            track_build_started.elapsed().as_secs_f64(),
            track_output.tracks.len(),
            conflicting_components.len(),
            total_conflict_obs,
            largest_conflict,
        );
    }
    let raw_tracks = track_output.tracks;
    if let Some(bin_frames) = temporal_support_bin_frames {
        log_temporal_track_support(
            "constructed",
            raw_tracks
                .iter()
                .map(|track| track.iter().map(|&(image, _)| image).collect()),
            &image_assignment,
            bin_frames,
        );
        log_temporal_track_support(
            "conflicting",
            conflicting_components
                .iter()
                .map(|track| track.iter().map(|&(image, _)| image).collect()),
            &image_assignment,
            bin_frames,
        );
    }
    let mut tracks = raw_tracks
        .into_iter()
        .map(|observations| {
            let metric_anchored = track_is_metric_anchored(&observations, &image_assignment);
            WorkingTrack {
                observations,
                position: None,
                metric_anchored,
            }
        })
        .collect::<Vec<_>>();
    let mut image_tracks = image_track_index(features.len(), &tracks);
    for index in &mut image_tracks {
        index.sort_unstable_by_key(|&(keypoint, _)| keypoint);
    }
    let metric_supports = metric_frame_supports(frames.len(), &tracks, &image_assignment);
    let seed_candidates = if let Some(frame) = config.seed_frame {
        (metric_supports[frame] > 0)
            .then_some(frame)
            .into_iter()
            .collect()
    } else {
        metric_seed_candidates(&metric_supports)
    };
    if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
        let metric_tracks = tracks.iter().filter(|track| track.metric_anchored).count();
        let max_frame_support = metric_supports.iter().copied().max().unwrap_or(0);
        eprintln!(
            "rig-sfm-debug: tracks={} metric_anchored_tracks={} metric_seed_candidates={} max_frame_support={max_frame_support}",
            tracks.len(),
            metric_tracks,
            seed_candidates.len(),
        );
    }
    let mut image_poses = vec![None; features.len()];
    let required_seed_landmarks = config.min_pnp_inliers.max(6);
    let mut total_seed_attempts = 0usize;
    let mut total_seed_robust_tracks = 0usize;
    let mut total_seed_pruned_observations = 0usize;
    let mut total_seed_majority_rejections = 0usize;
    let mut best_failed_seed = (
        seed_candidates
            .first()
            .copied()
            .ok_or(RigSfmError::NoMetricSeed)?,
        0usize,
    );
    let mut accepted_seed = None;
    // Undo only what the *previous* failed attempt actually touched (its own
    // triangulated landmarks and its own seed frame's images), instead of an
    // O(#tracks) full rescan of `tracks` plus an O(#images) `image_poses.fill`
    // on every single candidate. On a heavily conflict-fragmented graph
    // (many near-duplicate/aliased frames yield a plausible two-view seed
    // that still falls short of `required_seed_landmarks`) the candidate list
    // can run to thousands of entries, turning the old blanket reset into an
    // O(candidates x tracks) blowup. `triangulate_frontier` only ever sets a
    // *new* `Some` position for a track already `None` (it skips tracks whose
    // position is already `Some`), and only for tracks reachable from this
    // seed frame's own frontier, so tracking exactly the returned `landmarks`
    // list is sufficient to undo it byte-for-byte -- this changes nothing
    // observable when `robust_triangulation_pruning` is off (the default):
    // it resets the same fields to the same values, just without touching
    // tracks/images the previous attempt never wrote to. When pruning is on,
    // it can also mutate existing observations beyond just `position`, so
    // that opt-in path keeps the original full clone/restore for
    // correctness.
    let mut previous_touched_tracks: Vec<usize> = Vec::new();
    let mut previous_touched_images: Vec<usize> = Vec::new();
    let seed_search_started = if rig_timing {
        Some(Instant::now())
    } else {
        None
    };
    let mut seed_search_last_tick = Instant::now();
    let candidate_count = seed_candidates.len();
    let seed_iter: Box<dyn Iterator<Item = usize>> = if config.max_seed_attempts > 0 {
        Box::new(seed_candidates.into_iter().take(config.max_seed_attempts))
    } else {
        Box::new(seed_candidates.into_iter())
    };
    let mut seed_attempts_tried = 0usize;
    for seed_frame_index in seed_iter {
        seed_attempts_tried += 1;
        if let Some(started) = seed_search_started {
            if seed_search_last_tick.elapsed().as_secs_f64() >= 5.0 {
                eprintln!(
                    "rig-timing: seed-search-tick elapsed={:.3}s attempts={}/{} \
                     best_failed_landmarks={} required={}",
                    started.elapsed().as_secs_f64(),
                    seed_attempts_tried,
                    candidate_count,
                    best_failed_seed.1,
                    required_seed_landmarks,
                );
                seed_search_last_tick = Instant::now();
            }
        }
        let tracks_before_seed = config.robust_triangulation_pruning.then(|| tracks.clone());
        if tracks_before_seed.is_some() {
            for track in &mut tracks {
                track.position = None;
            }
        } else {
            for &track_index in &previous_touched_tracks {
                tracks[track_index].position = None;
            }
        }
        previous_touched_tracks.clear();
        for &image_index in &previous_touched_images {
            image_poses[image_index] = None;
        }
        previous_touched_images.clear();
        install_image_poses(
            rig,
            &frames[seed_frame_index],
            &Pose::identity(),
            &mut image_poses,
        );
        previous_touched_images = frames[seed_frame_index]
            .images
            .iter()
            .map(|image| image.image_index)
            .collect();
        let seed_frontier = frames[seed_frame_index]
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
            seed_frontier,
        );
        previous_touched_tracks = triangulation.landmarks.clone();
        total_seed_attempts += triangulation.attempts;
        total_seed_robust_tracks += triangulation.robust_tracks;
        total_seed_pruned_observations += triangulation.pruned_observations;
        total_seed_majority_rejections += triangulation.majority_rejections;
        let seed_landmarks = triangulation.landmarks.len();
        if seed_landmarks > best_failed_seed.1 {
            best_failed_seed = (seed_frame_index, seed_landmarks);
        }
        if seed_landmarks >= required_seed_landmarks {
            accepted_seed = Some((seed_frame_index, triangulation.landmarks));
            break;
        }
        if let Some(tracks_before_seed) = tracks_before_seed {
            tracks = tracks_before_seed;
            previous_touched_tracks.clear();
        }
    }
    if let Some(started) = seed_search_started {
        eprintln!(
            "rig-timing: seed-search-done elapsed={:.3}s attempts={}/{} accepted={}",
            started.elapsed().as_secs_f64(),
            seed_attempts_tried,
            candidate_count,
            accepted_seed.is_some(),
        );
    }
    let Some((seed_frame_index, seed_landmarks)) = accepted_seed else {
        return Err(RigSfmError::InsufficientSeedStructure {
            frame: best_failed_seed.0,
            required: required_seed_landmarks,
            triangulated: best_failed_seed.1,
        });
    };
    let seed_triangulation = TriangulationUpdate {
        landmarks: seed_landmarks,
        attempts: total_seed_attempts,
        robust_tracks: total_seed_robust_tracks,
        pruned_observations: total_seed_pruned_observations,
        majority_rejections: total_seed_majority_rejections,
    };
    let mut frame_poses = vec![None; frames.len()];
    frame_poses[seed_frame_index] = Some(Pose::identity());
    let mut registration_order = vec![seed_frame_index];
    emit_registration_trace(
        trace_registration,
        RegistrationTrace {
            mode: "legacy",
            order: 0,
            frame: seed_frame_index,
            support: seed_triangulation.landmarks.len(),
            inliers: seed_triangulation.landmarks.len(),
            sensors: rig.sensors().len(),
        },
        frame_poses[seed_frame_index]
            .as_ref()
            .expect("seed pose installed"),
    );

    let pnp = GeneralizedPnPRansac {
        iterations: config.pnp_max_iterations,
        reprojection_threshold: config.max_reprojection_error_px,
        seed: config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    };

    // Each landmark enters each observing frame's cache exactly once. Heap
    // versions make stale support counts cheap to discard, avoiding the
    // all-unregistered-frame rescan that becomes quadratic at 10k scale.
    macro_rules! rig_step {
        ($label:expr, $started:expr) => {
            if rig_timing {
                eprintln!(
                    "rig-timing: setup-step {}={:.3}s total_elapsed={:.3}s",
                    $label,
                    $started.elapsed().as_secs_f64(),
                    rig_timing_started.elapsed().as_secs_f64(),
                );
            }
        };
    }
    let step_started = Instant::now();
    let mut frame_correspondences: Vec<Vec<CachedRigCorrespondence>> =
        vec![Vec::new(); frames.len()];
    let mut frame_versions = vec![0usize; frames.len()];
    let mut attempted_versions = vec![None; frames.len()];
    let mut candidate_heap = BinaryHeap::new();
    let mut work = RigSfmWorkStats {
        triangulation_attempts: seed_triangulation.attempts,
        robust_triangulation_tracks: seed_triangulation.robust_tracks,
        robust_triangulation_pruned_observations: seed_triangulation.pruned_observations,
        robust_triangulation_majority_rejections: seed_triangulation.majority_rejections,
        ..RigSfmWorkStats::default()
    };
    rig_step!("alloc-frame-caches", step_started);
    if rig_timing {
        eprintln!(
            "rig-timing: setup-step-start append-landmark-correspondences \
             seed_landmarks={}",
            seed_triangulation.landmarks.len(),
        );
    }
    let step_started = Instant::now();
    work.correspondence_cache_insertions += append_landmark_correspondences(
        &seed_triangulation.landmarks,
        &tracks,
        features,
        &image_assignment,
        &mut frame_correspondences,
        &mut frame_versions,
        &mut candidate_heap,
    );
    rig_step!(
        format!(
            "append-landmark-correspondences insertions={}",
            work.correspondence_cache_insertions
        ),
        step_started
    );
    // A deferred pair is forbidden from changing the established track
    // structure, but it is still valid input to the opt-in direct bridge:
    // direct points are temporary source-stereo triangulations consumed by
    // robust PnP and never enter union-find or BA. This lets low-support local
    // edges cross a registration gap without letting them corrupt the map.
    let step_started = Instant::now();
    let stereo_links = (config.direct_stereo_pnp_max_frame_gap > 0)
        .then(|| build_verified_stereo_links(pairwise, &image_assignment));
    rig_step!("build-stereo-links", step_started);
    let step_started = Instant::now();
    let direct_pair_adjacency = (config.direct_stereo_pnp_max_frame_gap > 0)
        .then(|| build_frame_pair_adjacency(pairwise, &image_assignment, frames.len()));
    rig_step!("build-direct-pair-adjacency", step_started);
    let step_started = Instant::now();
    let motion_pair_adjacency = (config.motion_bridge_max_frame_gap > 0)
        .then(|| build_frame_pair_adjacency(mapping_pairwise, &image_assignment, frames.len()));
    rig_step!("build-motion-pair-adjacency", step_started);
    let step_started = Instant::now();
    let mut direct_pairs_visited = vec![false; pairwise.len()];
    let mut direct_target_observations = vec![HashSet::new(); frames.len()];
    let mut direct_source_queue = registration_order.clone();
    let mut motion_pairs_visited = vec![false; mapping_pairwise.len()];
    let mut motion_source_queue = registration_order.clone();
    let mut motion_pending = Vec::new();
    rig_step!("alloc-bridge-state", step_started);
    if rig_timing {
        eprintln!(
            "rig-timing: setup-done total_elapsed={:.3}s entering-growth-loop",
            rig_timing_started.elapsed().as_secs_f64(),
        );
    }
    let mut rig_timing_last_tick = Instant::now();
    let mut rig_timing_outer_iterations = 0u64;
    loop {
        rig_timing_outer_iterations += 1;
        if rig_timing && rig_timing_last_tick.elapsed().as_secs_f64() >= 5.0 {
            eprintln!(
                "rig-timing: growth-tick elapsed={:.3}s outer_iterations={} heap_len={} \
                 registered={} pnp_attempts={} triangulation_attempts={} \
                 direct_bridge_pair_visits={} motion_bridge_pair_visits={} \
                 deferred_pair_visits={}",
                rig_timing_started.elapsed().as_secs_f64(),
                rig_timing_outer_iterations,
                candidate_heap.len(),
                registration_order.len(),
                work.pnp_attempts,
                work.triangulation_attempts,
                work.direct_bridge_pair_visits,
                work.motion_bridge_pair_visits,
                work.deferred_pair_visits,
            );
            rig_timing_last_tick = Instant::now();
        }
        let mut rig_timing_heap_pops = 0u64;
        while let Some((support, Reverse(frame), version)) = candidate_heap.pop() {
            rig_timing_heap_pops += 1;
            if rig_timing && rig_timing_heap_pops.is_multiple_of(200_000) {
                eprintln!(
                    "rig-timing: heap-pop-progress elapsed={:.3}s heap_pops_this_outer_iter={} \
                     heap_len={} registered={} pnp_attempts={}",
                    rig_timing_started.elapsed().as_secs_f64(),
                    rig_timing_heap_pops,
                    candidate_heap.len(),
                    registration_order.len(),
                    work.pnp_attempts,
                );
            }
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
            if rig_timing && work.pnp_attempts.is_multiple_of(500) {
                eprintln!(
                    "rig-timing: pnp-progress elapsed={:.3}s pnp_attempts={} registered={} \
                     heap_len={}",
                    rig_timing_started.elapsed().as_secs_f64(),
                    work.pnp_attempts,
                    registration_order.len(),
                    candidate_heap.len(),
                );
            }
            let used_direct_bridge = frame_correspondences[frame]
                .iter()
                .any(|cached| cached.direct_point3d.is_some());
            let correspondences = frame_correspondences[frame]
                .iter()
                .filter_map(|cached| {
                    let point3d = cached
                        .direct_point3d
                        .or_else(|| cached.track_index.and_then(|track| tracks[track].position))?;
                    Some(GeneralizedCorrespondence2D3D {
                        sensor_index: cached.sensor_index,
                        point2d: cached.point2d,
                        point3d,
                        confidence: None,
                    })
                })
                .collect::<Vec<_>>();
            let distinct_sensors = correspondences
                .iter()
                .map(|correspondence| correspondence.sensor_index)
                .collect::<HashSet<_>>()
                .len();
            let required_sensors = if used_direct_bridge {
                config
                    .direct_stereo_min_pnp_sensors
                    .unwrap_or(config.min_pnp_sensors)
            } else {
                config.min_pnp_sensors
            };
            if distinct_sensors < required_sensors {
                work.pnp_insufficient_sensor_attempts += 1;
                if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
                    eprintln!(
                    "rig-sfm-debug: frame={frame} support={} sensors={distinct_sensors} pnp=insufficient-sensors required={}",
                    correspondences.len(),
                        required_sensors,
                );
                }
                continue;
            }
            if rig_timing {
                eprintln!(
                    "rig-timing: pnp-estimate-start frame={frame} correspondences={} \
                     pnp_attempts={} registered={}",
                    correspondences.len(),
                    work.pnp_attempts,
                    registration_order.len(),
                );
            }
            let rig_timing_pnp_estimate_started = if rig_timing {
                Some(Instant::now())
            } else {
                None
            };
            let pnp_report = pnp.estimate(rig, &correspondences);
            if let Some(started) = rig_timing_pnp_estimate_started {
                eprintln!(
                    "rig-timing: pnp-estimate-done frame={frame} elapsed={:.3}s \
                     correspondences={} succeeded={}",
                    started.elapsed().as_secs_f64(),
                    correspondences.len(),
                    pnp_report.is_some(),
                );
            }
            let Some(report) = pnp_report else {
                work.pnp_estimation_failures += 1;
                if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
                    eprintln!(
                    "rig-sfm-debug: frame={frame} support={} sensors={distinct_sensors} pnp=estimation-failed",
                    correspondences.len(),
                );
                }
                continue;
            };
            if report.inliers.len() < config.min_pnp_inliers.max(6) {
                work.pnp_inlier_rejections += 1;
                if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
                    eprintln!(
                    "rig-sfm-debug: frame={frame} support={} sensors={distinct_sensors} pnp=inlier-rejected inliers={} required={}",
                    correspondences.len(),
                    report.inliers.len(),
                    config.min_pnp_inliers.max(6),
                );
                }
                continue;
            }
            emit_registration_trace(
                trace_registration,
                RegistrationTrace {
                    mode: "legacy",
                    order: registration_order.len(),
                    frame,
                    support: correspondences.len(),
                    inliers: report.inliers.len(),
                    sensors: distinct_sensors,
                },
                &report.pose,
            );
            work.pnp_registrations += 1;
            work.direct_bridge_registrations += usize::from(used_direct_bridge);
            frame_poses[frame] = Some(report.pose);
            registration_order.push(frame);
            direct_source_queue.push(frame);
            motion_source_queue.push(frame);
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
            let rig_timing_frontier_len = frontier.len();
            if rig_timing {
                eprintln!(
                    "rig-timing: triangulate-frontier-start frame={frame} \
                     frontier_len={rig_timing_frontier_len} registered={}",
                    registration_order.len(),
                );
            }
            let rig_timing_triangulate_started = if rig_timing {
                Some(Instant::now())
            } else {
                None
            };
            let triangulation = triangulate_frontier(
                rig,
                features,
                &image_assignment,
                &image_poses,
                config,
                &mut tracks,
                frontier,
            );
            if let Some(started) = rig_timing_triangulate_started {
                let elapsed = started.elapsed().as_secs_f64();
                if elapsed >= 0.5 {
                    eprintln!(
                        "rig-timing: slow-triangulate-frontier elapsed={:.3}s frame={frame} \
                         frontier_len={rig_timing_frontier_len} attempts={} \
                         total_elapsed={:.3}s pnp_attempts={}",
                        elapsed,
                        triangulation.attempts,
                        rig_timing_started.elapsed().as_secs_f64(),
                        work.pnp_attempts,
                    );
                }
            }
            work.triangulation_attempts += triangulation.attempts;
            work.robust_triangulation_tracks += triangulation.robust_tracks;
            work.robust_triangulation_pruned_observations += triangulation.pruned_observations;
            work.robust_triangulation_majority_rejections += triangulation.majority_rejections;
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
                if rig_timing {
                    eprintln!(
                        "rig-timing: local-ba-start registration_count={} \
                         total_elapsed={:.3}s",
                        registration_order.len(),
                        rig_timing_started.elapsed().as_secs_f64(),
                    );
                }
                let local_ba_step_started = Instant::now();
                let (active_frames, anchor) = select_local_ba_frames(
                    rig,
                    frames,
                    features,
                    &image_assignment,
                    &image_tracks,
                    &image_poses,
                    &tracks,
                    &registration_order,
                    config,
                );
                if rig_timing {
                    eprintln!(
                        "rig-timing: select-local-ba-frames-done elapsed={:.3}s \
                         active_frames={}",
                        local_ba_step_started.elapsed().as_secs_f64(),
                        active_frames.len(),
                    );
                }
                let local_ba_config = BaConfig {
                    max_iterations: config.local_ba_iterations,
                    ..config.ba_config
                };
                let local_ba_solve_started = Instant::now();
                let local_ba_result = if active_frames.is_empty() {
                    None
                } else {
                    run_rig_bundle_adjustment(
                        rig,
                        features,
                        &image_assignment,
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
                };
                if rig_timing {
                    eprintln!(
                        "rig-timing: local-ba-solve-done elapsed={:.3}s active_frames={} \
                         total_elapsed={:.3}s",
                        local_ba_solve_started.elapsed().as_secs_f64(),
                        active_frames.len(),
                        rig_timing_started.elapsed().as_secs_f64(),
                    );
                }
                if local_ba_result.is_some() {
                    work.local_ba_runs += 1;
                    if config.ba_metric_tracks_only {
                        let affected_tracks = active_frames
                            .iter()
                            .flat_map(|frame| frames[*frame].images.iter())
                            .flat_map(|image| {
                                image_tracks[image.image_index]
                                    .iter()
                                    .map(|(_, track)| *track)
                            })
                            .collect::<HashSet<_>>();
                        let refreshed = retriangulate_unanchored_tracks(
                            rig,
                            features,
                            &image_assignment,
                            &image_poses,
                            config,
                            &mut tracks,
                            &affected_tracks,
                        );
                        work.ba_retriangulated_tracks += refreshed;
                        work.ba_requeued_frames += requeue_connected_frames(
                            &affected_tracks,
                            &tracks,
                            &image_assignment,
                            &frame_poses,
                            &frame_correspondences,
                            &mut frame_versions,
                            &mut candidate_heap,
                        );
                    }
                }
            }
        }
        if let Some(stereo_links) = stereo_links.as_ref() {
            let direct_pair_adjacency = direct_pair_adjacency
                .as_ref()
                .expect("direct pair adjacency accompanies stereo links");
            let direct = append_direct_stereo_pnp_correspondences(
                rig,
                frames,
                features,
                pairwise,
                &image_assignment,
                &image_poses,
                config,
                stereo_links,
                direct_pair_adjacency,
                &mut direct_source_queue,
                &mut direct_pairs_visited,
                &mut direct_target_observations,
                &mut frame_correspondences,
                &mut frame_versions,
                &mut candidate_heap,
            );
            work.direct_bridge_pair_visits += direct.pair_visits;
            work.direct_bridge_correspondence_insertions += direct.insertions;
            work.correspondence_cache_insertions += direct.insertions;
            if direct.insertions > 0 {
                continue;
            }
        }
        if let Some(adjacency) = motion_pair_adjacency.as_ref() {
            let update = collect_motion_bridge_candidates(
                rig,
                frames,
                features,
                mapping_pairwise,
                &image_assignment,
                &frame_poses,
                config,
                adjacency,
                &mut motion_source_queue,
                &mut motion_pairs_visited,
            );
            work.motion_bridge_pair_visits += update.pair_visits;
            work.motion_bridge_estimation_failures += update.estimation_failures;
            work.motion_bridge_rotation_rejections += update.rotation_rejections;
            motion_pending.extend(update.candidates);
            motion_pending.retain(|candidate| frame_poses[candidate.frame].is_none());
            motion_pending.sort_unstable_by(|left, right| {
                left.inliers
                    .cmp(&right.inliers)
                    .then_with(|| right.frame_gap.cmp(&left.frame_gap))
                    .then_with(|| right.frame.cmp(&left.frame))
            });
            if let Some(candidate) = motion_pending.pop() {
                let frame = candidate.frame;
                frame_poses[frame] = Some(candidate.pose);
                registration_order.push(frame);
                direct_source_queue.push(frame);
                motion_source_queue.push(frame);
                work.motion_bridge_registrations += 1;
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
                work.robust_triangulation_tracks += triangulation.robust_tracks;
                work.robust_triangulation_pruned_observations += triangulation.pruned_observations;
                work.robust_triangulation_majority_rejections += triangulation.majority_rejections;
                work.correspondence_cache_insertions += append_landmark_correspondences(
                    &triangulation.landmarks,
                    &tracks,
                    features,
                    &image_assignment,
                    &mut frame_correspondences,
                    &mut frame_versions,
                    &mut candidate_heap,
                );
                continue;
            }
        }
        break;
    }

    let required_pnp_support = config.min_pnp_inliers.max(6);
    for (frame, pose) in frame_poses.iter().enumerate() {
        if pose.is_some() {
            continue;
        }
        let support = frame_correspondences[frame].len();
        work.max_unregistered_support = work.max_unregistered_support.max(support);
        if support == 0 {
            work.unregistered_zero_support_frames += 1;
        } else if support < required_pnp_support {
            work.unregistered_below_pnp_support_frames += 1;
        } else {
            work.unregistered_eligible_pnp_frames += 1;
        }
        let sensors = frame_correspondences[frame]
            .iter()
            .map(|correspondence| correspondence.sensor_index)
            .collect::<HashSet<_>>()
            .len();
        if sensors < config.min_pnp_sensors {
            work.unregistered_below_sensor_frames += 1;
        }
    }
    if rig_timing {
        eprintln!(
            "rig-timing: growth-loop-done elapsed={:.3}s registered_frames={} pnp_attempts={} \
             triangulation_attempts={} local_ba_runs={}",
            rig_timing_started.elapsed().as_secs_f64(),
            registration_order.len(),
            work.pnp_attempts,
            work.triangulation_attempts,
            work.local_ba_runs,
        );
    }

    if config.recover_metric_conflict_tracks && !conflicting_components.is_empty() {
        let conflict_recovery_started = Instant::now();
        let recovered = recover_metric_conflict_tracks(
            rig,
            features,
            mapping_pairwise,
            &conflicting_components,
            &image_assignment,
            &image_poses,
            config,
        );
        work.geometry_recovered_tracks = recovered.len();
        work.geometry_recovered_observations =
            recovered.iter().map(|track| track.observations.len()).sum();
        tracks.extend(recovered);
        if rig_timing {
            eprintln!(
                "rig-timing: conflict-recovery-done elapsed={:.3}s components={} recovered_tracks={} \
                 recovered_observations={} max_hypotheses={}",
                conflict_recovery_started.elapsed().as_secs_f64(),
                conflicting_components.len(),
                work.geometry_recovered_tracks,
                work.geometry_recovered_observations,
                config.conflict_recovery_max_hypotheses,
            );
        }
    }

    // Correct a short, mutually-cancelling pose jump before final BA can
    // deliberately freeze the affected poses through
    // `final_ba_min_pose_observations`.  This must run before the isolated
    // one-frame detector: repairing the pair first turns a two-frame gauge
    // discontinuity into ordinary local motion, and keeps the two detectors'
    // decisions deterministic.
    if config.repair_paired_pose_jumps {
        let repair_started = Instant::now();
        let repair = repair_paired_pose_jumps(
            rig,
            frames,
            seed_frame_index,
            config.paired_pose_jump_absolute_step_m,
            config.paired_pose_jump_min_step_ratio,
            config.paired_pose_jump_max_frame_span,
            config.paired_pose_jump_max_closure_ratio,
            &mut frame_poses,
            &mut image_poses,
        );
        work.paired_pose_jump_repairs += repair.repairs;
        work.paired_pose_jump_repaired_frames += repair.repaired_frames;
        if rig_timing {
            eprintln!(
                "rig-timing: paired-pose-jump-repair-done elapsed={:.3}s repairs={} repaired_frames={}",
                repair_started.elapsed().as_secs_f64(),
                repair.repairs,
                repair.repaired_frames,
            );
        }
    }

    // Correct an unmistakable low-support PnP detour before final BA can
    // deliberately freeze that pose through `final_ba_min_pose_observations`.
    // A second pass after deferred registration handles poses added later.
    if config.repair_isolated_pose_outliers {
        let repair_started = Instant::now();
        let repair = repair_isolated_pose_outliers(
            rig,
            frames,
            seed_frame_index,
            config.isolated_pose_max_midpoint_error_m,
            config.isolated_pose_min_detour_ratio,
            config.isolated_pose_repair_max_passes,
            &mut frame_poses,
            &mut image_poses,
        );
        work.isolated_pose_repair_passes += repair.passes;
        work.isolated_pose_repairs += repair.repairs;
        if rig_timing {
            eprintln!(
                "rig-timing: isolated-pose-repair-done elapsed={:.3}s passes={} repairs={}",
                repair_started.elapsed().as_secs_f64(),
                repair.passes,
                repair.repairs,
            );
        }
    }

    let final_ba_started = Instant::now();
    let mut bundle_adjustment = if config.final_bundle_adjustment && config.final_ba_passes > 0 {
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
    if rig_timing {
        eprintln!(
            "rig-timing: final-ba-done elapsed={:.3}s passes={}",
            final_ba_started.elapsed().as_secs_f64(),
            config.final_ba_passes,
        );
    }

    let structure_refine_started = Instant::now();
    work.structure_refined_tracks = refine_rig_structure(
        rig,
        features,
        &image_assignment,
        &image_poses,
        config,
        &mut tracks,
    );
    if rig_timing {
        eprintln!(
            "rig-timing: structure-refine-done elapsed={:.3}s refined_tracks={} total_elapsed={:.3}s",
            structure_refine_started.elapsed().as_secs_f64(),
            work.structure_refined_tracks,
            rig_timing_started.elapsed().as_secs_f64(),
        );
    }

    if !deferred_pairwise.is_empty() {
        let deferred_stereo_links = build_relevant_verified_stereo_links(
            mapping_pairwise,
            deferred_pairwise,
            &image_assignment,
        );
        let deferred = register_deferred_frames(
            rig,
            frames,
            features,
            deferred_pairwise,
            &image_assignment,
            config,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
            &mut image_tracks,
            &deferred_stereo_links,
        );
        work.deferred_pair_visits = deferred.pair_visits;
        work.deferred_correspondence_insertions = deferred.correspondence_insertions;
        work.deferred_pnp_attempts = deferred.pnp_attempts;
        work.deferred_pnp_estimation_failures = deferred.pnp_estimation_failures;
        work.deferred_pnp_inlier_rejections = deferred.pnp_inlier_rejections;
        work.deferred_registrations = deferred.registrations;
        work.deferred_interpolation_registrations = deferred.interpolation_registrations;
        work.deferred_observations_attached = deferred.observations_attached;
    }

    if config.repair_isolated_pose_outliers {
        let repair = repair_isolated_pose_outliers(
            rig,
            frames,
            seed_frame_index,
            config.isolated_pose_max_midpoint_error_m,
            config.isolated_pose_min_detour_ratio,
            config.isolated_pose_repair_max_passes,
            &mut frame_poses,
            &mut image_poses,
        );
        work.isolated_pose_repair_passes += repair.passes;
        work.isolated_pose_repairs += repair.repairs;
        if repair.repairs > 0 {
            work.structure_refined_tracks += refine_rig_structure(
                rig,
                features,
                &image_assignment,
                &image_poses,
                config,
                &mut tracks,
            );
        }
    }

    if config.retriangulate_deferred_tracks_after_registration
        && !retriangulation_pairwise.is_empty()
    {
        let recovered = append_retriangulated_deferred_tracks(
            rig,
            features,
            retriangulation_pairwise,
            &image_assignment,
            &image_poses,
            config,
            &mut tracks,
        );
        work.deferred_retriangulated_tracks = recovered.tracks;
        work.deferred_retriangulated_observations = recovered.observations;
    }

    if config.complete_tracks_after_registration {
        let completion = complete_positioned_tracks(
            rig,
            features,
            pairwise,
            &image_assignment,
            &image_poses,
            config.track_completion_max_passes,
            config.track_completion_max_reprojection_error_px,
            &mut tracks,
        );
        work.track_completion_passes = completion.passes;
        work.track_completion_pair_visits = completion.pair_visits;
        work.track_completion_observations = completion.observations;
        work.track_completion_reprojection_rejections = completion.reprojection_rejections;
        if completion.observations > 0 {
            work.structure_refined_tracks += refine_rig_structure(
                rig,
                features,
                &image_assignment,
                &image_poses,
                config,
                &mut tracks,
            );
        }
    }
    for _ in 0..config.final_filter_refinement_passes {
        let pruned = filter_positioned_track_observations(
            rig,
            features,
            &image_assignment,
            &image_poses,
            config.max_reprojection_error_px,
            &mut tracks,
        );
        if pruned == 0 {
            break;
        }
        work.final_filter_refinement_passes += 1;
        work.final_filter_refinement_pruned_observations += pruned;
        if config.final_bundle_adjustment && config.final_ba_passes > 0 {
            let refinement = run_windowed_final_ba(
                rig,
                features,
                &image_assignment,
                config,
                &mut frame_poses,
                &mut image_poses,
                &mut tracks,
            )?;
            merge_rig_ba_stats(&mut bundle_adjustment, refinement);
        }
        work.structure_refined_tracks += refine_rig_structure(
            rig,
            features,
            &image_assignment,
            &image_poses,
            config,
            &mut tracks,
        );
    }
    if let Some(bin_frames) = temporal_support_bin_frames {
        log_temporal_track_support(
            "positioned",
            tracks
                .iter()
                .filter(|track| track.position.is_some())
                .map(|track| track.observations.iter().map(|&(image, _)| image).collect()),
            &image_assignment,
            bin_frames,
        );
    }
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
    if let Some(bin_frames) = temporal_support_bin_frames {
        log_temporal_track_support(
            "published",
            sfm_tracks.iter().map(|track| {
                track
                    .observations
                    .iter()
                    .map(|&(image, _, _)| image)
                    .collect()
            }),
            &image_assignment,
            bin_frames,
        );
    }
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
        track_build_stats,
        work,
        bundle_adjustment,
    })
}

#[derive(Debug, Default)]
pub(super) struct DirectBridgeUpdate {
    pub(super) pair_visits: usize,
    pub(super) insertions: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_direct_stereo_pnp_correspondences(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
    pair_adjacency: &[Vec<usize>],
    source_queue: &mut Vec<usize>,
    pair_visited: &mut [bool],
    target_observations: &mut [HashSet<(usize, usize)>],
    frame_correspondences: &mut [Vec<CachedRigCorrespondence>],
    frame_versions: &mut [usize],
    heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> DirectBridgeUpdate {
    let mut update = DirectBridgeUpdate::default();
    let mut changed_frames = HashSet::new();
    let queued = std::mem::take(source_queue);
    for source_frame in queued {
        if image_poses[frames[source_frame].images[0].image_index].is_none() {
            continue;
        }
        for &pair_index in &pair_adjacency[source_frame] {
            if pair_visited[pair_index] {
                continue;
            }
            let pair = &pairwise[pair_index];
            let frame_i = image_assignment[pair.image_i].0;
            let frame_j = image_assignment[pair.image_j].0;
            let target_frame = if frame_i == source_frame {
                frame_j
            } else {
                frame_i
            };
            if image_poses[frames[target_frame].images[0].image_index].is_some() {
                pair_visited[pair_index] = true;
                continue;
            }
            if source_frame.abs_diff(target_frame) > config.direct_stereo_pnp_max_frame_gap {
                pair_visited[pair_index] = true;
                continue;
            }
            pair_visited[pair_index] = true;
            update.pair_visits += 1;
            let source_is_i = frame_i == source_frame;
            for &(keypoint_i, keypoint_j) in &pair.matches {
                let (source_image, source_keypoint, target_image, target_keypoint) = if source_is_i
                {
                    (pair.image_i, keypoint_i, pair.image_j, keypoint_j)
                } else {
                    (pair.image_j, keypoint_j, pair.image_i, keypoint_i)
                };
                if target_observations[target_frame].contains(&(target_image, target_keypoint)) {
                    continue;
                }
                let Some(point3d) = triangulate_verified_stereo_observation(
                    rig,
                    features,
                    image_assignment,
                    image_poses,
                    config,
                    stereo_links,
                    source_image,
                    source_keypoint,
                ) else {
                    continue;
                };
                target_observations[target_frame].insert((target_image, target_keypoint));
                frame_correspondences[target_frame].push(CachedRigCorrespondence {
                    sensor_index: image_assignment[target_image].1,
                    point2d: features[target_image].keypoints[target_keypoint],
                    track_index: None,
                    direct_point3d: Some(point3d),
                });
                update.insertions += 1;
                changed_frames.insert(target_frame);
            }
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
    update
}

#[allow(clippy::too_many_arguments)]
pub(super) fn triangulate_verified_stereo_observation(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
    image: usize,
    keypoint: usize,
) -> Option<Point3<f64>> {
    let &(other_image, other_keypoint) = stereo_links.get(&(image, keypoint))?;
    let pose = image_poses[image].as_ref()?;
    let other_pose = image_poses[other_image].as_ref()?;
    let sensor = &rig.sensors()[image_assignment[image].1];
    let other_sensor = &rig.sensors()[image_assignment[other_image].1];
    let bearing = sensor
        .camera
        .normalize_pixel(&features[image].keypoints[keypoint])?;
    let other_bearing = other_sensor
        .camera
        .normalize_pixel(&features[other_image].keypoints[other_keypoint])?;
    let direction = pose
        .camera_to_world()
        .rotation
        .transform_vector(&Vector3::new(bearing.x, bearing.y, 1.0).normalize())
        .normalize();
    let other_direction = other_pose
        .camera_to_world()
        .rotation
        .transform_vector(&Vector3::new(other_bearing.x, other_bearing.y, 1.0).normalize())
        .normalize();
    let angle_deg = direction
        .dot(&other_direction)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    if angle_deg < config.direct_stereo_min_triangulation_angle_deg {
        return None;
    }
    let point = closest_ray_midpoint(
        &pose.camera_center_world(),
        &direction,
        &other_pose.camera_center_world(),
        &other_direction,
    )?;
    for (check_image, check_keypoint) in [(image, keypoint), (other_image, other_keypoint)] {
        let check_pose = image_poses[check_image].as_ref()?;
        let check_sensor = &rig.sensors()[image_assignment[check_image].1];
        let point_camera = check_pose.transform_world_point(&point);
        if point_camera.z <= 0.0 {
            return None;
        }
        let projected = check_sensor.camera.project(&point_camera)?;
        if (projected - features[check_image].keypoints[check_keypoint]).norm()
            > config.max_reprojection_error_px
        {
            return None;
        }
    }
    Some(point)
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
                track_index: Some(track_index),
                direct_point3d: None,
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

#[allow(clippy::too_many_arguments)]
fn requeue_connected_frames(
    track_indices: &HashSet<usize>,
    tracks: &[WorkingTrack],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    frame_correspondences: &[Vec<CachedRigCorrespondence>],
    frame_versions: &mut [usize],
    heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> usize {
    let mut frames = track_indices
        .iter()
        .flat_map(|track| tracks[*track].observations.iter())
        .map(|(image, _)| image_assignment[*image].0)
        .filter(|frame| frame_poses[*frame].is_none())
        .collect::<Vec<_>>();
    frames.sort_unstable();
    frames.dedup();
    for &frame in &frames {
        frame_versions[frame] += 1;
        heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    frames.len()
}
