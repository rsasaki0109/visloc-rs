//! Windowed joint rig bundle adjustment (`--joint-rig-ba`).

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct JointRigBaWindowUpdate {
    pub(super) pose_overrides: BTreeMap<u64, Pose>,
    pub(super) landmark_updates: BTreeMap<usize, LandmarkOutput>,
    pub(super) selected_landmarks: usize,
    pub(super) selected_observations: usize,
    pub(super) referenced_frames: usize,
    pub(super) free_frames: usize,
    pub(super) initial_cost: f64,
    pub(super) final_cost: f64,
    pub(super) solver_initial_cost: f64,
    pub(super) solver_final_cost: f64,
    pub(super) iterations: usize,
    pub(super) converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct JointRigBaFilteringCandidate {
    pub(super) pose_overrides: BTreeMap<u64, Pose>,
    pub(super) landmark_updates: BTreeMap<usize, LandmarkOutput>,
    pub(super) removed_observations: BTreeMap<ObservationKey, FilterObservationReason>,
    pub(super) removed_track_ids: BTreeSet<usize>,
    pub(super) selected_landmarks: usize,
    pub(super) selected_observations: usize,
    pub(super) referenced_frames: usize,
    pub(super) free_frames: usize,
    pub(super) full_pre_ba_cost: f64,
    pub(super) full_post_ba_cost: f64,
    pub(super) retained_pre_ba_cost: f64,
    pub(super) retained_post_filter_cost: f64,
    pub(super) retriangulated_tracks: usize,
    pub(super) raw_preserved_tracks: usize,
    pub(super) dlt_attempted_tracks: usize,
    pub(super) raw_fallback_reason_counts: BTreeMap<RawPointFallbackReason, usize>,
    pub(super) iterations: usize,
    pub(super) converged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum JointRigBaSkip {
    LandmarkCap { count: usize },
    ObservationCap { count: usize },
    ReferencedFrameCap { count: usize },
    NoFreeFrames,
    Invalid(String),
}

impl std::fmt::Display for JointRigBaSkip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LandmarkCap { count } => write!(f, "landmark cap exceeded ({count})"),
            Self::ObservationCap { count } => write!(f, "observation cap exceeded ({count})"),
            Self::ReferencedFrameCap { count } => {
                write!(f, "referenced-frame cap exceeded ({count})")
            }
            Self::NoFreeFrames => write!(f, "window has no free frames"),
            Self::Invalid(reason) => write!(f, "invalid BA candidate: {reason}"),
        }
    }
}

pub(super) fn derive_rig_pose_for_frame(
    manifest: &RigManifest,
    frame_id: u64,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<Pose, String> {
    let mut pose: Option<Pose> = None;
    for image in images
        .values()
        .filter(|image| image.atlas.frame_id == frame_id)
    {
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| format!("frame {frame_id} references unknown sensor"))?;
        let candidate = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&image.atlas.pose.world_to_camera),
        };
        if let Some(previous) = &pose {
            let center_error =
                (previous.camera_center_world() - candidate.camera_center_world()).norm();
            let rotation_error = previous
                .world_to_camera
                .rotation
                .rotation_to(&candidate.world_to_camera.rotation)
                .angle()
                .to_degrees();
            if !center_error.is_finite()
                || !rotation_error.is_finite()
                || center_error > MAX_RIG_CENTER_DISAGREEMENT_M
                || rotation_error > MAX_RIG_ROTATION_DISAGREEMENT_DEG
            {
                return Err(format!(
                    "frame {frame_id} rig pose disagreement: centre={center_error:.9}m rotation={rotation_error:.9}deg"
                ));
            }
        } else {
            pose = Some(candidate);
        }
    }
    pose.ok_or_else(|| format!("frame {frame_id} has no atlas images"))
}

fn derive_rig_poses_for_frames(
    manifest: &RigManifest,
    frame_ids: &BTreeSet<u64>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<BTreeMap<u64, Pose>, String> {
    frame_ids
        .iter()
        .map(|frame_id| {
            derive_rig_pose_for_frame(manifest, *frame_id, images).map(|pose| (*frame_id, pose))
        })
        .collect()
}

pub(super) fn selected_landmarks_for_frames(
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<Vec<usize>, String> {
    let mut selected = Vec::new();
    for (index, landmark) in landmarks.iter().enumerate() {
        let touches_active = landmark.observations.iter().any(|key| {
            images
                .get(&key.global_image_id)
                .is_some_and(|image| active_frames.contains(&image.atlas.frame_id))
        });
        if touches_active {
            selected.push(index);
        }
    }
    // This is intentionally a definition, not a heuristic: all tracks with
    // an active observation are selected, so no active pose is later changed
    // while silently leaving an unselected landmark unsupported.
    Ok(selected)
}

pub(super) fn validate_joint_ba_update(
    update: &JointRigBaWindowUpdate,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    selected: &[usize],
) -> Result<(), JointRigBaSkip> {
    let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
    for (index, landmark) in landmarks.iter().enumerate() {
        let touches_active = landmark.observations.iter().any(|key| {
            images.get(&key.global_image_id).is_some_and(|image| {
                update
                    .pose_overrides
                    .contains_key(&image.atlas.global_image_id)
            })
        });
        if touches_active != selected_set.contains(&index) {
            return Err(JointRigBaSkip::Invalid(format!(
                "selection invariant failed for landmark index {index}"
            )));
        }
    }
    if update.landmark_updates.len() != selected.len() {
        return Err(JointRigBaSkip::Invalid(format!(
            "candidate landmark count {} differs from selected count {}",
            update.landmark_updates.len(),
            selected.len()
        )));
    }
    let mut initial_cost = 0.0;
    let mut final_cost = 0.0;
    for index in selected {
        let old = landmarks.get(*index).ok_or_else(|| {
            JointRigBaSkip::Invalid("selected landmark index is invalid".to_owned())
        })?;
        let old_metrics =
            evaluate_landmark_metrics(old, &store.observations, images, cameras, &BTreeMap::new())
                .map_err(JointRigBaSkip::Invalid)?;
        initial_cost += old_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
        let candidate = update.landmark_updates.get(index).ok_or_else(|| {
            JointRigBaSkip::Invalid(format!("candidate omitted selected landmark {index}"))
        })?;
        if candidate.track_id != old.track_id || candidate.observations != old.observations {
            return Err(JointRigBaSkip::Invalid(format!(
                "candidate changed identity or observations for landmark {index}"
            )));
        }
        let candidate_metrics = evaluate_landmark_metrics(
            candidate,
            &store.observations,
            images,
            cameras,
            &update.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        final_cost += candidate_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
    }
    if !update.initial_cost.is_finite()
        || !update.final_cost.is_finite()
        || !initial_cost.is_finite()
        || !final_cost.is_finite()
        || final_cost > initial_cost + 1.0e-8 * initial_cost.abs().max(1.0)
    {
        return Err(JointRigBaSkip::Invalid(format!(
            "validated cost increased from {initial_cost:.12} to {final_cost:.12}"
        )));
    }
    if (update.initial_cost - initial_cost).abs() > 1.0e-8 * initial_cost.abs().max(1.0)
        || (update.final_cost - final_cost).abs() > 1.0e-8 * final_cost.abs().max(1.0)
    {
        return Err(JointRigBaSkip::Invalid(
            "candidate cost diagnostics disagree with independent reprojection".to_owned(),
        ));
    }
    for (index, candidate) in &update.landmark_updates {
        if !selected_set.contains(index) {
            return Err(JointRigBaSkip::Invalid(format!(
                "candidate updated unselected landmark {index}"
            )));
        }
        let metrics = evaluate_landmark_metrics(
            candidate,
            &store.observations,
            images,
            cameras,
            &update.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        if candidate.observations.len() < MIN_TRACK_OBSERVATIONS
            || metrics.mean_error > MAX_MEAN_REPROJECTION_PX
            || metrics.max_error > MAX_REPROJECTION_PX
        {
            return Err(JointRigBaSkip::Invalid(format!(
                "landmark {index} failed reprojection gate mean={:.6} max={:.6} observations={}",
                metrics.mean_error,
                metrics.max_error,
                candidate.observations.len()
            )));
        }
    }
    Ok(())
}

fn build_joint_rig_ba_window_raw(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
) -> Result<JointRigBaWindowUpdate, JointRigBaSkip> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    build_joint_rig_ba_window_raw_with_selected(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        &selected,
    )
}

pub(super) struct AtlasInnerTimer {
    started: Option<std::time::Instant>,
    frame: Option<u64>,
    phase: &'static str,
}

impl AtlasInnerTimer {
    pub(super) fn new(active_frames: &BTreeSet<u64>, phase: &'static str) -> Self {
        Self {
            started: std::env::var_os("VISLOC_ATLAS_TRACE_WINDOW_TIMING")
                .is_some()
                .then(std::time::Instant::now),
            frame: active_frames.first().copied(),
            phase,
        }
    }

    fn report(&self) {
        if let Some(started) = self.started {
            eprintln!(
                "atlas-inner-timing: start_frame={:?} phase={} seconds={:.9}",
                self.frame,
                self.phase,
                started.elapsed().as_secs_f64()
            );
        }
    }

    pub(super) fn next(&mut self, phase: &'static str) {
        self.report();
        self.phase = phase;
        if self.started.is_some() {
            self.started = Some(std::time::Instant::now());
        }
    }
}

impl Drop for AtlasInnerTimer {
    fn drop(&mut self) {
        self.report();
    }
}

pub(super) fn build_joint_rig_ba_window_raw_with_selected(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    selected: &[usize],
) -> Result<JointRigBaWindowUpdate, JointRigBaSkip> {
    let mut timer = AtlasInnerTimer::new(active_frames, "problem_build");
    if selected.len() > JOINT_BA_MAX_LANDMARKS {
        return Err(JointRigBaSkip::LandmarkCap {
            count: selected.len(),
        });
    }
    let selected_observations = selected.iter().try_fold(0usize, |count, index| {
        count.checked_add(landmarks[*index].observations.len())
    });
    let selected_observations =
        selected_observations.ok_or(JointRigBaSkip::ObservationCap { count: usize::MAX })?;
    if selected_observations > JOINT_BA_MAX_OBSERVATIONS {
        return Err(JointRigBaSkip::ObservationCap {
            count: selected_observations,
        });
    }
    let mut referenced_frames = BTreeSet::new();
    for index in selected {
        for key in &landmarks[*index].observations {
            let image = images.get(&key.global_image_id).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown image".to_owned())
            })?;
            referenced_frames.insert(image.atlas.frame_id);
        }
    }
    if referenced_frames.len() > JOINT_BA_MAX_REFERENCED_FRAMES {
        return Err(JointRigBaSkip::ReferencedFrameCap {
            count: referenced_frames.len(),
        });
    }
    let free_frames = active_frames
        .intersection(&referenced_frames)
        .copied()
        .collect::<BTreeSet<_>>();
    if free_frames.is_empty() {
        return Err(JointRigBaSkip::NoFreeFrames);
    }
    if free_frames.len() > JOINT_BA_MAX_FREE_FRAMES {
        return Err(JointRigBaSkip::Invalid(format!(
            "free-frame cap exceeded ({})",
            free_frames.len()
        )));
    }
    let rig_poses = derive_rig_poses_for_frames(manifest, &referenced_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    let camera = cameras
        .values()
        .next()
        .cloned()
        .ok_or_else(|| JointRigBaSkip::Invalid("no cameras available".to_owned()))?;
    let mut ba = BundleAdjustment::new(camera);
    for (frame_id, pose) in &rig_poses {
        ba.add_pose(*frame_id, pose.clone());
        if !free_frames.contains(frame_id) {
            ba.fix_pose(*frame_id);
        }
    }
    if !rig_poses
        .keys()
        .any(|frame_id| !free_frames.contains(frame_id))
    {
        let anchor = *referenced_frames
            .first()
            .ok_or(JointRigBaSkip::NoFreeFrames)?;
        ba.fix_pose(anchor);
    }
    for index in selected {
        let landmark = &landmarks[*index];
        let id = u64::try_from(landmark.track_id)
            .map_err(|_| JointRigBaSkip::Invalid("track id exceeds u64".to_owned()))?;
        if !landmark
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(JointRigBaSkip::Invalid(format!(
                "landmark {} has non-finite position",
                landmark.track_id
            )));
        }
        ba.add_landmark(id, landmark.position);
    }
    for index in selected {
        let landmark = &landmarks[*index];
        let landmark_id = u64::try_from(landmark.track_id)
            .map_err(|_| JointRigBaSkip::Invalid("track id exceeds u64".to_owned()))?;
        for key in &landmark.observations {
            let image = images.get(&key.global_image_id).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown image".to_owned())
            })?;
            let observation = store.observations.get(key).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown observation".to_owned())
            })?;
            let sensor = manifest
                .sensors
                .get(&image.atlas.sensor_index)
                .ok_or_else(|| {
                    JointRigBaSkip::Invalid("landmark references unknown sensor".to_owned())
                })?;
            let camera = cameras.get(&image.atlas.camera_id).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown camera".to_owned())
            })?;
            if !observation.xy.coords.iter().all(|value| value.is_finite()) {
                return Err(JointRigBaSkip::Invalid(
                    "non-finite BA observation".to_owned(),
                ));
            }
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: image.atlas.frame_id,
                landmark_id,
                xy: observation.xy,
                camera: camera.clone(),
                sensor_from_rig: sensor.sensor_from_rig.clone(),
            });
        }
    }
    let baseline_cost = ba.cost();
    if !baseline_cost.is_finite() {
        return Err(JointRigBaSkip::Invalid(
            "initial BA cost is non-finite".to_owned(),
        ));
    }
    timer.next("solver");
    let result = ba
        .optimize(&BaConfig {
            max_iterations: JOINT_BA_MAX_ITERATIONS,
            initial_lambda: Some(1.0e-4),
            linear_solver: LinearSolver::Sparse,
            robust_kernel: RobustKernel::None,
            refine_intrinsics: false,
            refine_distortion: false,
            parallel: false,
            ..BaConfig::default()
        })
        .map_err(|error| JointRigBaSkip::Invalid(format!("bundle adjustment failed: {error}")))?;
    timer.next("output_conversion_and_costs");
    let solver_initial_cost = result.initial_cost;
    let solver_final_cost = result.final_cost;
    if !solver_initial_cost.is_finite() || !solver_final_cost.is_finite() {
        return Err(JointRigBaSkip::Invalid(
            "BA costs are non-finite".to_owned(),
        ));
    }
    let mut pose_overrides = BTreeMap::new();
    for image in images.values() {
        if !free_frames.contains(&image.atlas.frame_id) {
            continue;
        }
        let rig_pose = ba
            .poses
            .get(&image.atlas.frame_id)
            .ok_or_else(|| JointRigBaSkip::Invalid("BA omitted referenced rig pose".to_owned()))?;
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| {
                JointRigBaSkip::Invalid("BA image references unknown sensor".to_owned())
            })?;
        let pose = Pose {
            world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
        };
        if !pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
            || !pose
                .world_to_camera
                .rotation
                .coords
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(JointRigBaSkip::Invalid(
                "BA produced non-finite pose".to_owned(),
            ));
        }
        pose_overrides.insert(image.atlas.global_image_id, pose);
    }
    let mut landmark_updates = BTreeMap::new();
    for index in selected {
        let old = &landmarks[*index];
        let id = u64::try_from(old.track_id)
            .map_err(|_| JointRigBaSkip::Invalid("track id exceeds u64".to_owned()))?;
        let position = *ba
            .landmarks
            .get(&id)
            .ok_or_else(|| JointRigBaSkip::Invalid("BA omitted selected landmark".to_owned()))?;
        let mut candidate = old.clone();
        candidate.position = position;
        let metrics = evaluate_landmark_metrics_allowing_nonpositive_depth(
            &candidate,
            &store.observations,
            images,
            cameras,
            &pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        candidate.errors = metrics.errors;
        candidate.mean_error = metrics.mean_error;
        candidate.rms_error = metrics.rms_error;
        candidate.max_error = metrics.max_error;
        landmark_updates.insert(*index, candidate);
    }
    let mut initial_cost = 0.0;
    let mut final_cost = 0.0;
    for index in selected {
        let old_metrics = evaluate_landmark_metrics(
            &landmarks[*index],
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )
        .map_err(JointRigBaSkip::Invalid)?;
        let new_metrics = evaluate_landmark_metrics_allowing_nonpositive_depth(
            landmark_updates
                .get(index)
                .ok_or_else(|| JointRigBaSkip::Invalid("BA update omitted landmark".to_owned()))?,
            &store.observations,
            images,
            cameras,
            &pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        initial_cost += old_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
        final_cost += new_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
    }
    let update = JointRigBaWindowUpdate {
        pose_overrides,
        landmark_updates,
        selected_landmarks: selected.len(),
        selected_observations,
        referenced_frames: referenced_frames.len(),
        free_frames: free_frames.len(),
        initial_cost,
        final_cost,
        solver_initial_cost,
        solver_final_cost,
        iterations: result.iterations.len(),
        converged: result.converged,
    };
    Ok(update)
}

pub(super) fn build_joint_rig_ba_window(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
) -> Result<JointRigBaWindowUpdate, JointRigBaSkip> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    let update =
        build_joint_rig_ba_window_raw(manifest, store, images, cameras, landmarks, active_frames)?;
    if let Err(reason) =
        validate_joint_ba_update(&update, store, images, cameras, landmarks, &selected)
    {
        println!(
            "joint_rig_ba_candidate status=rejected selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} solver_initial_cost={:.9} solver_final_cost={:.9} iterations={} converged={} reason={reason}",
            update.selected_landmarks,
            update.selected_observations,
            update.referenced_frames,
            update.free_frames,
            update.solver_initial_cost,
            update.solver_final_cost,
            update.iterations,
            update.converged,
        );
        return Err(reason);
    }
    Ok(update)
}

pub(super) fn run_joint_rig_ba(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut [LandmarkOutput],
) -> Result<JointRigBaSummary, String> {
    let mut frame_ids = images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<Vec<_>>();
    frame_ids.sort_unstable();
    frame_ids.dedup();
    let mut summary = JointRigBaSummary::default();
    for start in (0..frame_ids.len()).step_by(JOINT_BA_WINDOW_STRIDE) {
        let end = (start + JOINT_BA_WINDOW_LENGTH).min(frame_ids.len());
        let active_frames = frame_ids[start..end]
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert!(active_frames.len() <= JOINT_BA_MAX_FREE_FRAMES);
        summary.windows_considered += 1;
        let window_counts = joint_ba_window_counts(landmarks, images, &active_frames);
        if let Ok((selected_count, observation_count, referenced_count, free_count)) =
            window_counts.as_ref()
        {
            summary.selected_landmarks += *selected_count;
            summary.selected_observations += *observation_count;
            summary.max_referenced_frames = summary.max_referenced_frames.max(*referenced_count);
            summary.max_free_frames = summary.max_free_frames.max(*free_count);
        }
        match build_joint_rig_ba_window(manifest, store, images, cameras, landmarks, &active_frames)
        {
            Ok(update) => {
                apply_pose_overrides(images, &update.pose_overrides)?;
                for (index, landmark) in update.landmark_updates {
                    landmarks[index] = landmark;
                }
                summary.windows_accepted += 1;
                summary.max_iterations = summary.max_iterations.max(update.iterations);
                if update.converged {
                    summary.converged_windows += 1;
                }
                if summary.windows_accepted == 1 {
                    summary.max_solver_initial_cost = update.solver_initial_cost;
                    summary.min_solver_final_cost = update.solver_final_cost;
                } else {
                    summary.max_solver_initial_cost = summary
                        .max_solver_initial_cost
                        .max(update.solver_initial_cost);
                    summary.min_solver_final_cost =
                        summary.min_solver_final_cost.min(update.solver_final_cost);
                }
                summary.final_cost = update.final_cost;
                println!(
                    "joint_rig_ba_window start_frame={} end_frame={} active_frames={} status=accepted selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} solver_initial_cost={:.9} solver_final_cost={:.9} validated_initial_cost={:.9} validated_final_cost={:.9} iterations={} converged={}",
                    active_frames.first().copied().unwrap_or_default(),
                    active_frames.last().copied().unwrap_or_default(),
                    active_frames.len(),
                    update.selected_landmarks,
                    update.selected_observations,
                    update.referenced_frames,
                    update.free_frames,
                    update.solver_initial_cost,
                    update.solver_final_cost,
                    update.initial_cost,
                    update.final_cost,
                    update.iterations,
                    update.converged,
                );
            }
            Err(reason) => {
                summary.windows_skipped += 1;
                let (selected_count, observation_count, referenced_count, free_count) =
                    window_counts.unwrap_or_default();
                println!(
                    "joint_rig_ba_window start_frame={} end_frame={} active_frames={} status=skipped selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} reason={reason}",
                    active_frames.first().copied().unwrap_or_default(),
                    active_frames.last().copied().unwrap_or_default(),
                    active_frames.len(),
                    selected_count,
                    observation_count,
                    referenced_count,
                    free_count,
                );
            }
        }
    }
    Ok(summary)
}
