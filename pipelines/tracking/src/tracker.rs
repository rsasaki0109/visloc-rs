//! Frame-to-map tracking: [`Tracker`], [`ImageTracker`], the
//! [`FrameLocalizer`] abstraction, and covisibility local-map selection.

use super::*;

#[derive(Debug, Clone)]
pub struct Tracker<P, M = ConstantPoseMotionModel> {
    localization_pipeline: P,
    motion_model: M,
    config: TrackingConfig,
    state: TrackingState,
    successive_failures: usize,
    last_result: Option<TrackingResult>,
    last_successful_frame_id: Option<FrameId>,
    last_successful_pose: Option<Pose>,
    stats: TrackingStats,
}

#[derive(Debug, Clone)]
pub struct ImageTracker<X, T = Tracker<LocalizationPipeline>> {
    pub extractor: X,
    pub tracker: T,
}

impl<X> ImageTracker<X, Tracker<LocalizationPipeline, ConstantPoseMotionModel>>
where
    X: FeatureExtractor,
{
    pub fn new(extractor: X, config: TrackingConfig) -> Self {
        Self {
            extractor,
            tracker: Tracker::new(LocalizationPipeline::default(), config),
        }
    }
}

impl<X, T> ImageTracker<X, T>
where
    X: FeatureExtractor,
{
    pub fn with_tracker(extractor: X, tracker: T) -> Self {
        Self { extractor, tracker }
    }

    pub fn tracker(&self) -> &T {
        &self.tracker
    }

    pub fn tracker_mut(&mut self) -> &mut T {
        &mut self.tracker
    }

    pub fn into_parts(self) -> (X, T) {
        (self.extractor, self.tracker)
    }
}

impl<P> Tracker<P, ConstantPoseMotionModel>
where
    P: FrameLocalizer,
{
    pub fn new(localization_pipeline: P, config: TrackingConfig) -> Self {
        Self::with_motion_model(localization_pipeline, ConstantPoseMotionModel, config)
    }
}

impl<P, M> Tracker<P, M>
where
    P: FrameLocalizer,
    M: MotionModel,
{
    pub fn with_motion_model(
        localization_pipeline: P,
        motion_model: M,
        config: TrackingConfig,
    ) -> Self {
        Self {
            localization_pipeline,
            motion_model,
            config,
            state: TrackingState::Uninitialized,
            successive_failures: 0,
            last_result: None,
            last_successful_frame_id: None,
            last_successful_pose: None,
            stats: TrackingStats::default(),
        }
    }

    pub fn state(&self) -> TrackingState {
        self.state
    }

    pub fn successive_failures(&self) -> usize {
        self.successive_failures
    }

    pub fn last_result(&self) -> Option<&TrackingResult> {
        self.last_result.as_ref()
    }

    pub fn last_successful_frame_id(&self) -> Option<FrameId> {
        self.last_successful_frame_id
    }

    pub fn last_successful_pose(&self) -> Option<&Pose> {
        self.last_successful_pose.as_ref()
    }

    pub fn stats(&self) -> &TrackingStats {
        &self.stats
    }

    pub fn motion_model(&self) -> &M {
        &self.motion_model
    }

    /// Mutable access to the configured motion model. Use this to feed
    /// out-of-band inputs (e.g., raw IMU samples into
    /// [`ImuPredictiveMotionModel`]) that the per-frame `track_frame*`
    /// path does not surface.
    pub fn motion_model_mut(&mut self) -> &mut M {
        &mut self.motion_model
    }

    pub fn reset(&mut self) {
        self.state = TrackingState::Uninitialized;
        self.successive_failures = 0;
        self.last_result = None;
        self.last_successful_frame_id = None;
        self.last_successful_pose = None;
        self.stats = TrackingStats::default();
        self.motion_model.reset();
    }

    /// Override the tracker's per-frame history with a successful
    /// relocalization recovery result. Called by callers that detect
    /// a primary `track_frame` failure and recover via a separate
    /// `FrameLocalizer` (e.g. the relocalization-on-tracker-death stage
    /// in `OnlineSlamPipeline`).
    ///
    /// Reverts the failed-frame side-effects from the primary attempt
    /// (`successive_failures` counter, `last_result` / `last_successful_*`
    /// fields, `motion_model.observe(failed_result)`) and re-runs them
    /// as if the recovered result had been the primary outcome. The
    /// failed-frame audit counter (`stats.failed_frame_count`) is left
    /// unchanged so the caller can tell that primary tracking dropped
    /// the frame before relocalization rescued it.
    ///
    /// No-op when `result.localization.success == false` (callers must
    /// gate on success before invoking).
    pub fn accept_relocalization_result(&mut self, result: TrackingResult) {
        if !result.localization.success {
            return;
        }
        self.accept_external_tracking_success(result, true);
    }

    /// Override the tracker's per-frame history with a successful
    /// GT-seeded segment restart (e.g. stereo re-bootstrap after
    /// prolonged tracking loss). Same side-effect repair as
    /// [`Self::accept_relocalization_result`] but does not increment
    /// `relocalization_count`.
    pub fn accept_segment_restart_result(&mut self, result: TrackingResult) {
        if !result.localization.success {
            return;
        }
        self.accept_external_tracking_success(result, false);
    }

    fn accept_external_tracking_success(
        &mut self,
        result: TrackingResult,
        count_as_relocalization: bool,
    ) {
        self.state = TrackingState::Tracking;
        self.successive_failures = 0;
        self.last_successful_frame_id = Some(result.frame_id);
        self.last_successful_pose = result.localization.pose.clone();
        self.last_result = Some(result.clone());
        if count_as_relocalization {
            self.stats.relocalization_count += 1;
        }
        self.stats.successful_frame_count += 1;
        if result.localization.inlier_count > 0 {
            self.stats.total_inlier_count += result.localization.inlier_count;
            self.stats.total_correspondence_count += result.localization.correspondence_count;
        }
        self.motion_model.observe(&result);
    }

    /// Apply a rigid world-frame correction to the tracker's continuation
    /// state: `last_successful_pose` and the motion model's own cached
    /// world-frame state (e.g. [`ImuPredictiveMotionModel`]'s
    /// `velocity_world` and cached poses, or [`ConstantVelocityMotionModel`]'s
    /// cached poses), via [`MotionModel::apply_pose_correction`].
    ///
    /// `correction` follows the same convention used to move map
    /// landmarks after an external pose-graph optimisation: it maps OLD
    /// world-frame points/poses to NEW world-frame points/poses, i.e.
    /// `p_new = correction.transform_point(&p_old)`. A cached
    /// `world_to_camera` pose `T_wc_old` is corrected to
    /// `T_wc_old.compose(&correction.inverse())` — the pose whose
    /// projection of the corrected point matches the original
    /// projection of the old point (see the call site in
    /// `visloc-slam`'s online loop-closure refinement for the derivation).
    ///
    /// This is the tracker-side half of loop-closure correction
    /// propagation: without it, the very next `track_frame` call would
    /// predict a prior pose (and, for `pnp_pose_prior_warm_start` /
    /// covisibility-radius gating, a candidate radius) anchored to the
    /// pre-correction map even though the landmarks it is about to match
    /// against just moved.
    pub fn apply_pose_correction(&mut self, correction: &SE3) {
        if let Some(pose) = self.last_successful_pose.as_mut() {
            pose.world_to_camera = pose.world_to_camera.compose(&correction.inverse());
        }
        self.motion_model.apply_pose_correction(correction);
    }

    /// [`Self::apply_pose_correction`]'s counterpart for a `Sim(3)`
    /// world-frame correction — used after a `Sim(3)` pose-graph solve
    /// (`visloc-slam`'s online loop-closure refinement `Sim3` solver)
    /// instead of the rigid `SE(3)` solver, so scale-drift corrections
    /// keep the tracker's continuation state consistent too.
    ///
    /// `last_successful_pose` is a rigid [`Pose`], so only `correction`'s
    /// rotation+translation part is folded into it (mirroring the map's
    /// keyframe write-back convention, which likewise keeps
    /// `map.keyframes[*].frame.pose` rigid); `correction.scale` is left
    /// to [`MotionModel::apply_similarity_correction`] for models that
    /// cache a world-frame velocity, where it multiplies that vector
    /// (see [`ImuPredictiveMotionModel`]'s
    /// override).
    pub fn apply_similarity_pose_correction(&mut self, correction: &Sim3) {
        let se3_part = SE3::new(correction.rotation, correction.translation);
        if let Some(pose) = self.last_successful_pose.as_mut() {
            pose.world_to_camera = pose.world_to_camera.compose(&se3_part.inverse());
        }
        self.motion_model.apply_similarity_correction(correction);
    }

    pub fn pose_prior_for_frame(&self, frame: &Frame) -> Option<Pose> {
        self.motion_model.predict_pose(
            frame,
            self.last_result.as_ref(),
            self.last_successful_pose.as_ref(),
        )
    }

    pub fn localization_prior_for_frame(&self, frame: &Frame, radius: f64) -> LocalizationPrior {
        if let Some(pose_prior) = self.pose_prior_for_frame(frame) {
            LocalizationPrior::from_pose(pose_prior, radius)
        } else {
            LocalizationPrior::none()
        }
    }

    pub fn track_frame(&mut self, frame: &Frame, map: &VisualMap) -> TrackingResult {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frame_with_descriptor_store(frame, map, &descriptor_store)
    }

    pub fn track_frames(&mut self, frames: &[Frame], map: &VisualMap) -> Vec<TrackingResult> {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frames_with_descriptor_store(frames, map, &descriptor_store)
    }

    pub fn track_frame_with_provider<P2>(&mut self, frame: &Frame, provider: &P2) -> TrackingResult
    where
        P2: MapProvider + DescriptorProvider,
    {
        let map = provider.visual_map();
        let map_stats = map_provider_stats(provider);
        if let Some(descriptor_store) = provider.landmark_descriptor_store() {
            self.track_frame_with_descriptor_store_and_map_stats(
                frame,
                map,
                descriptor_store,
                map_stats,
            )
        } else {
            let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
            let map_stats = MapProviderStats {
                descriptor_count: descriptor_store.len(),
                ..map_stats
            };
            self.track_frame_with_descriptor_store_and_map_stats(
                frame,
                map,
                &descriptor_store,
                map_stats,
            )
        }
    }

    pub fn track_frames_with_provider<P2>(
        &mut self,
        frames: &[Frame],
        provider: &P2,
    ) -> Vec<TrackingResult>
    where
        P2: MapProvider + DescriptorProvider,
    {
        frames
            .iter()
            .map(|frame| self.track_frame_with_provider(frame, provider))
            .collect()
    }

    pub fn track_frame_with_prior_submap_provider<P2>(
        &mut self,
        frame: &Frame,
        provider: &P2,
        radius: f64,
    ) -> TrackingResult
    where
        P2: MapProvider + DescriptorProvider,
    {
        if let Some(pose_prior) = self.pose_prior_for_frame(frame) {
            let submap_provider = InMemoryMapProvider::from_provider_radius(
                provider,
                pose_prior.camera_center_world(),
                radius,
            );
            self.track_frame_with_provider(frame, &submap_provider)
        } else {
            self.track_frame_with_provider(frame, provider)
        }
    }

    pub fn track_frames_with_prior_submap_provider<P2>(
        &mut self,
        frames: &[Frame],
        provider: &P2,
        radius: f64,
    ) -> Vec<TrackingResult>
    where
        P2: MapProvider + DescriptorProvider,
    {
        frames
            .iter()
            .map(|frame| self.track_frame_with_prior_submap_provider(frame, provider, radius))
            .collect()
    }

    pub fn track_frame_with_localization_prior_submap_provider<P2>(
        &mut self,
        frame: &Frame,
        provider: &P2,
        prior: &LocalizationPrior,
    ) -> TrackingResult
    where
        P2: MapProvider + DescriptorProvider,
    {
        if let (Some(center_world), Some(radius)) = (prior.center_world(), prior.radius) {
            let submap_provider =
                InMemoryMapProvider::from_provider_radius(provider, center_world, radius);
            let mut result = self.track_frame_with_provider(frame, &submap_provider);
            result.used_external_localization_prior = true;
            result.external_localization_prior_radius = Some(radius);
            self.stats.external_localization_prior_used_count += 1;
            self.last_result = Some(result.clone());
            result
        } else {
            self.track_frame_with_provider(frame, provider)
        }
    }

    pub fn track_frames_with_localization_prior_submap_provider<'a, P2, I>(
        &mut self,
        frames_and_priors: I,
        provider: &P2,
    ) -> Vec<TrackingResult>
    where
        P2: MapProvider + DescriptorProvider,
        I: IntoIterator<Item = (&'a Frame, Option<&'a LocalizationPrior>)>,
    {
        frames_and_priors
            .into_iter()
            .map(|(frame, prior)| {
                if let Some(prior) = prior {
                    self.track_frame_with_localization_prior_submap_provider(frame, provider, prior)
                } else {
                    self.track_frame_with_provider(frame, provider)
                }
            })
            .collect()
    }

    pub fn track_frame_with_descriptor_store(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> TrackingResult {
        let map_stats = MapProviderStats {
            camera_count: map.cameras.len(),
            landmark_count: map.landmarks.len(),
            keyframe_count: map.keyframes.len(),
            descriptor_count: descriptor_store.len(),
        };
        self.track_frame_with_descriptor_store_and_map_stats(
            frame,
            map,
            descriptor_store,
            map_stats,
        )
    }

    pub fn track_frames_with_descriptor_store(
        &mut self,
        frames: &[Frame],
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Vec<TrackingResult> {
        frames
            .iter()
            .map(|frame| self.track_frame_with_descriptor_store(frame, map, descriptor_store))
            .collect()
    }

    fn track_frame_with_descriptor_store_and_map_stats(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        map_stats: MapProviderStats,
    ) -> TrackingResult {
        let pose_prior = self.motion_model.predict_pose(
            frame,
            self.last_result.as_ref(),
            self.last_successful_pose.as_ref(),
        );
        let used_pose_prior =
            pose_prior.is_some() && self.config.last_pose_candidate_radius.is_some();

        let covisibility_local_store =
            self.build_covisibility_local_map_store(map, descriptor_store);
        let covisibility_local_map_size =
            covisibility_local_store.as_ref().map(|store| store.len());
        let active_descriptor_store: &LandmarkDescriptorStore = covisibility_local_store
            .as_ref()
            .unwrap_or(descriptor_store);

        // Copied out (the config is `Copy`) rather than borrowed, so the
        // widen-retry ladder below can take `&mut self` for its stats
        // bookkeeping without fighting the borrow checker over `self.config`.
        let projection_guided_config = self.config.projection_guided_tracking;
        let mut localization = match (projection_guided_config, pose_prior.as_ref()) {
            (Some(projection_config), Some(prior)) => self.run_projection_guided_tracking(
                frame,
                map,
                active_descriptor_store,
                prior,
                &projection_config,
            ),
            _ => self.localize_appearance_global(
                frame,
                map,
                active_descriptor_store,
                pose_prior.as_ref(),
            ),
        };

        if localization.success {
            self.apply_local_map_refinement(
                frame,
                map,
                covisibility_local_store.as_ref(),
                &mut localization,
            );
        }

        let (tracking_failure_reason, continuation_pose) =
            self.apply_tracking_quality_gate(frame, map, pose_prior.as_ref(), &mut localization);

        let previous_state = self.state;
        let event = if localization.success {
            self.state = TrackingState::Tracking;
            self.successive_failures = 0;
            match previous_state {
                TrackingState::Uninitialized => TrackingEvent::Initialized,
                TrackingState::Tracking => TrackingEvent::Tracked,
                TrackingState::Lost => TrackingEvent::Relocalized,
            }
        } else {
            self.successive_failures += 1;
            if self.successive_failures >= self.config.min_successive_failures_to_lost {
                self.state = TrackingState::Lost;
                TrackingEvent::Lost
            } else if self.state == TrackingState::Uninitialized {
                self.state = TrackingState::Uninitialized;
                TrackingEvent::TrackingFailed
            } else {
                self.state = TrackingState::Tracking;
                TrackingEvent::TrackingFailed
            }
        };

        let result = TrackingResult {
            frame_id: frame.id,
            state: self.state,
            event,
            successive_failures: self.successive_failures,
            pose_prior,
            used_pose_prior,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason,
            map_landmark_count: map_stats.landmark_count,
            map_stats,
            localization,
            covisibility_local_map_size,
        };

        let mut continuation_result = result.clone();
        if let Some(pose) = continuation_pose {
            continuation_result.localization.pose = Some(pose);
        }
        self.update_history(&continuation_result);
        self.motion_model.observe(&continuation_result);
        result
    }

    /// Today's appearance-global localization path (descriptor search over
    /// the full radius-filtered candidate set, with the optional PnP
    /// warm-start). Factored out so it is shared, byte-for-byte, between the
    /// `projection_guided_tracking == None` default and the widen-retry
    /// ladder's fallback rung — enabling projection-guided tracking can
    /// therefore only ADD tracking chances, never remove this one.
    fn localize_appearance_global(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
    ) -> LocalizationResult {
        if self.config.pnp_pose_prior_warm_start
            && self.motion_model.allows_pnp_pose_prior_warm_start()
        {
            self.localization_pipeline
                .localize_frame_with_pose_prior_warm_start_and_descriptor_store(
                    frame,
                    map,
                    descriptor_store,
                    pose_prior,
                    self.config.last_pose_candidate_radius,
                )
        } else {
            self.localization_pipeline
                .localize_frame_with_pose_prior_and_descriptor_store(
                    frame,
                    map,
                    descriptor_store,
                    pose_prior,
                    self.config.last_pose_candidate_radius,
                )
        }
    }

    /// Stage 1 + widen-retry ladder: try projection-window matching at
    /// `config.search_radius_px`, multiplying the radius by
    /// `config.widen_factor` and retrying up to `config.max_widen_retries`
    /// times when the current radius's result fails the pose-estimation
    /// quality gate. Falls back to today's appearance-global path
    /// (`localize_appearance_global`) if every projection attempt fails, so
    /// enabling this feature can only add tracking chances versus today.
    fn run_projection_guided_tracking(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        config: &ProjectionGuidedTrackingConfig,
    ) -> LocalizationResult {
        let mut radius = config.search_radius_px;
        let mut projection_result =
            LocalizationResult::failure(LocalizationFailureReason::NoDescriptorMatches, 0, 0, 0);
        for attempt in 0..=config.max_widen_retries {
            self.stats.projection_guided_attempt_count += 1;
            projection_result = self
                .localization_pipeline
                .localize_frame_with_projection_window_descriptor_store_and_query_landmark_ratio(
                    frame,
                    map,
                    descriptor_store,
                    pose_prior,
                    self.config.last_pose_candidate_radius,
                    radius,
                    config.max_query_landmark_distance_ratio,
                );
            if projection_result.success {
                break;
            }
            if attempt < config.max_widen_retries {
                self.stats.projection_guided_widen_retry_count += 1;
                radius *= config.widen_factor;
            }
        }

        if projection_result.success {
            self.stats.projection_guided_success_count += 1;
            return projection_result;
        }

        let fallback =
            self.localize_appearance_global(frame, map, descriptor_store, Some(pose_prior));
        if fallback.success {
            self.stats.projection_guided_fallback_success_count += 1;
        }
        fallback
    }

    /// Stage 3: after ANY successful pose estimate (projection path or
    /// appearance-global fallback), project the covisibility local map (if
    /// one was built for this frame) with the ESTIMATED pose and re-optimize
    /// over the union of harvested and existing inlier correspondences.
    /// Accepts the refined pose only when its inlier count does not
    /// decrease relative to the pre-refinement result; otherwise
    /// `localization` is left unchanged. No-op unless
    /// `projection_guided_tracking` is configured with
    /// `local_map_refinement = true` AND a covisibility local map exists for
    /// this frame (there is otherwise no defined local-map landmark set to
    /// project).
    fn apply_local_map_refinement(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        covisibility_local_store: Option<&LandmarkDescriptorStore>,
        localization: &mut LocalizationResult,
    ) {
        let Some(projection_config) = self.config.projection_guided_tracking else {
            return;
        };
        if !projection_config.local_map_refinement {
            return;
        }
        let Some(local_store) = covisibility_local_store else {
            return;
        };

        let iterations = projection_config.refinement_iterations.max(1);
        let shrink = if projection_config
            .refinement_radius_shrink_factor
            .is_finite()
            && projection_config.refinement_radius_shrink_factor > 0.0
            && projection_config.refinement_radius_shrink_factor <= 1.0
        {
            projection_config.refinement_radius_shrink_factor
        } else {
            1.0
        };
        let strict_monotonic =
            iterations > 1 || projection_config.refinement_reassign_correspondences;
        let mut radius = projection_config.refinement_search_radius_px;
        let mut attempted_any = false;
        let mut accepted_any = false;

        for _ in 0..iterations {
            let refined = if projection_config.refinement_reassign_correspondences {
                self.localization_pipeline
                    .revise_pose_with_local_map_and_descriptor_store(
                        frame,
                        map,
                        local_store,
                        localization,
                        radius,
                    )
            } else {
                self.localization_pipeline
                    .refine_pose_with_local_map_and_descriptor_store(
                        frame,
                        map,
                        local_store,
                        localization,
                        radius,
                    )
            };
            radius *= shrink;

            let Some(refined) = refined else {
                continue;
            };
            attempted_any = true;
            self.stats.local_map_refinement_correspondence_gain_total += refined
                .correspondence_count
                .saturating_sub(localization.correspondence_count);

            let inliers_improved = refined.inlier_count > localization.inlier_count;
            let inliers_preserved = refined.inlier_count == localization.inlier_count;
            let error_not_worse =
                match (refined.reprojection_error, localization.reprojection_error) {
                    (Some(refined_error), Some(current_error)) => {
                        refined_error.is_finite() && refined_error <= current_error + 1.0e-9
                    }
                    (Some(refined_error), None) => refined_error.is_finite(),
                    (None, None) => true,
                    (None, Some(_)) => false,
                };
            let current_pairs = localization
                .inlier_query_indices
                .iter()
                .copied()
                .zip(localization.inlier_landmark_ids.iter().copied())
                .collect::<HashSet<_>>();
            let refined_pairs = refined
                .inlier_query_indices
                .iter()
                .copied()
                .zip(refined.inlier_landmark_ids.iter().copied())
                .collect::<HashSet<_>>();
            let retained_pair_ratio = if current_pairs.is_empty() {
                1.0
            } else {
                current_pairs.intersection(&refined_pairs).count() as f64
                    / current_pairs.len() as f64
            };
            let pair_retention_passes =
                retained_pair_ratio >= projection_config.refinement_min_inlier_pair_retention_ratio;
            let (translation_correction_m, rotation_correction_rad) =
                match (localization.pose.as_ref(), refined.pose.as_ref()) {
                    (Some(current_pose), Some(refined_pose)) => {
                        let translation = (refined_pose.camera_center_world()
                            - current_pose.camera_center_world())
                        .norm();
                        let rotation_delta = refined_pose.world_to_camera.rotation
                            * current_pose.world_to_camera.rotation.inverse();
                        (translation, rotation_delta.angle())
                    }
                    _ => (f64::INFINITY, f64::INFINITY),
                };
            let translation_correction_passes = projection_config
                .refinement_max_pose_translation_correction_m
                .is_none_or(|max| translation_correction_m <= max);
            let rotation_correction_passes = projection_config
                .refinement_max_pose_rotation_correction_rad
                .is_none_or(|max| rotation_correction_rad <= max);
            let accept = refined.success
                && (inliers_improved
                    || (inliers_preserved && (!strict_monotonic || error_not_worse)))
                && pair_retention_passes
                && translation_correction_passes
                && rotation_correction_passes;
            if accept {
                *localization = refined;
                accepted_any = true;
            }
        }

        if accepted_any {
            self.stats.local_map_refinement_accepted_count += 1;
        } else if attempted_any {
            self.stats.local_map_refinement_rejected_count += 1;
        }
    }

    /// Build a covisibility-graph-derived descriptor store, if the feature is
    /// enabled and the surrounding state allows it (tracker is in `Tracking`
    /// state with a known reference keyframe in `map`, the local-map landmark
    /// set is above `min_local_map_landmarks`). Returns `None` to signal that
    /// the caller should fall through to the original descriptor store.
    fn build_covisibility_local_map_store(
        &self,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Option<LandmarkDescriptorStore> {
        let config = self.config.covisibility_local_map.as_ref()?;
        if self.state != TrackingState::Tracking {
            return None;
        }
        let last_id = self.last_successful_frame_id?;
        let reference_kf = covisibility_pick_reference_keyframe(map, last_id)?;
        let local_landmarks = covisibility_local_map_landmarks(map, reference_kf, config);
        if local_landmarks.len() < config.min_local_map_landmarks {
            return None;
        }
        let mut filtered = LandmarkDescriptorStore::new();
        // Sort the local-landmark id set before iterating so downstream
        // consumers see a deterministic insertion order independent of
        // the per-process `HashSet` SipHash seed. Even though the
        // `LandmarkDescriptorStore` itself is a `HashMap`, this insulates
        // any caller that depends on observed insertion order.
        let mut local_landmark_ids: Vec<u64> = local_landmarks.iter().copied().collect();
        local_landmark_ids.sort();
        for landmark_id in &local_landmark_ids {
            if let Some(descriptor) = descriptor_store.get(*landmark_id) {
                filtered.insert(*landmark_id, descriptor.to_vec());
            }
        }
        if filtered.len() < config.min_local_map_landmarks {
            return None;
        }
        Some(filtered)
    }

    fn update_history(&mut self, result: &TrackingResult) {
        if self.stats.first_frame_id.is_none() {
            self.stats.first_frame_id = Some(result.frame_id);
        }
        self.stats.last_frame_id = Some(result.frame_id);
        self.stats.frame_count += 1;
        if result.localization.success {
            self.stats.successful_frame_count += 1;
            self.last_successful_frame_id = Some(result.frame_id);
            self.last_successful_pose = result.localization.pose.clone();
        } else {
            self.stats.failed_frame_count += 1;
        }

        if result.event == TrackingEvent::Lost {
            self.stats.lost_count += 1;
        }
        if result.event == TrackingEvent::Relocalized {
            self.stats.relocalization_count += 1;
        }
        if result.used_pose_prior {
            self.stats.pose_prior_used_count += 1;
        }
        if result.tracking_failure_reason.is_some() {
            self.stats.tracking_quality_gate_failure_count += 1;
        }
        if result.covisibility_local_map_size.is_some() {
            self.stats.covisibility_local_map_used_count += 1;
        }
        self.stats.total_inlier_count += result.localization.inlier_count;
        self.stats.total_correspondence_count += result.localization.correspondence_count;

        self.last_result = Some(result.clone());
    }

    fn apply_tracking_quality_gate(
        &mut self,
        frame: &Frame,
        map: &VisualMap,
        pose_prior: Option<&Pose>,
        localization: &mut LocalizationResult,
    ) -> (Option<TrackingFailureReason>, Option<Pose>) {
        if !localization.success {
            return (None, None);
        }

        if localization.inlier_count < self.config.min_inliers {
            *localization = localization.clone().rejected_by_quality_gate();
            return (
                Some(TrackingFailureReason::InsufficientInliers {
                    inlier_count: localization.inlier_count,
                    min_inliers: self.config.min_inliers,
                }),
                None,
            );
        }

        if localization.inlier_ratio < self.config.min_inlier_ratio {
            *localization = localization.clone().rejected_by_quality_gate();
            return (
                Some(TrackingFailureReason::InlierRatioTooLow {
                    inlier_ratio: localization.inlier_ratio,
                    min_inlier_ratio: self.config.min_inlier_ratio,
                }),
                None,
            );
        }

        if let (Some(reprojection_error), Some(max_reprojection_error)) = (
            localization.reprojection_error,
            self.config.max_mean_reprojection_error,
        ) {
            if reprojection_error > max_reprojection_error {
                *localization = localization.clone().rejected_by_quality_gate();
                return (
                    Some(TrackingFailureReason::MeanReprojectionErrorTooHigh {
                        reprojection_error,
                        max_reprojection_error,
                    }),
                    None,
                );
            }
        }

        let Some(max_translation_error) = self.config.max_pose_prior_translation_error else {
            return (None, None);
        };
        let Some(pose_prior) = pose_prior else {
            return (None, None);
        };
        let Some(estimated_pose) = localization.pose.as_ref() else {
            return (None, None);
        };

        let translation_error =
            (estimated_pose.camera_center_world() - pose_prior.camera_center_world()).norm();

        let mut max_translation_error = if self.config.pose_jump_gap_scaling {
            // The pose prior comes from the motion model, which (for the
            // constant-pose / constant-velocity models used here) is
            // anchored to `last_successful_pose`. If tracking has been
            // failing for several frames, that prior is stale — it hasn't
            // moved while the true camera pose has. Scale the gate by how
            // long the prior has been frozen so a good post-gap PnP
            // solution isn't rejected against a fixed radius measured from
            // a pose the camera left several frames ago.
            let frames_since_last_tracking_success = self
                .last_successful_frame_id
                .map(|last_id| frame.id.saturating_sub(last_id))
                .unwrap_or(1)
                .max(1);
            let multiplier = pose_jump_gap_scaling_multiplier(
                frames_since_last_tracking_success,
                self.config.pose_jump_gap_scaling_max_multiplier,
            );
            max_translation_error * multiplier as f64
        } else {
            max_translation_error
        };

        let mut continuation_pose = None;
        if let Some(override_config) = self.config.pose_prior_visual_override {
            let rotation_error = pose_prior
                .camera_to_world()
                .rotation
                .rotation_to(&estimated_pose.camera_to_world().rotation)
                .angle();
            if let Some(widened) = pose_prior_visual_override_threshold(
                override_config,
                localization,
                translation_error,
                rotation_error,
                max_translation_error,
            ) {
                max_translation_error = widened;
                self.stats.pose_prior_visual_override_count += 1;
                continuation_pose = covariance_weighted_continuation_pose(
                    frame,
                    map,
                    pose_prior,
                    estimated_pose,
                    localization,
                    override_config,
                )
                .or_else(|| Some(pose_prior.clone()));
            }
        }

        if translation_error <= max_translation_error {
            return (None, continuation_pose);
        }

        *localization = localization.clone().rejected_by_quality_gate();
        (
            Some(TrackingFailureReason::PosePriorTranslationErrorExceeded {
                translation_error,
                max_translation_error,
            }),
            None,
        )
    }
}

fn pose_prior_visual_override_satisfied(
    config: crate::PosePriorVisualOverrideConfig,
    localization: &LocalizationResult,
) -> bool {
    config.min_inliers > 0
        && config.min_inlier_ratio.is_finite()
        && (0.0..=1.0).contains(&config.min_inlier_ratio)
        && config.max_translation_error_multiplier.is_finite()
        && config.min_translation_error_multiplier.is_finite()
        && config.min_translation_error_multiplier >= 1.0
        && config.min_translation_error_multiplier <= config.max_translation_error_multiplier
        && config.max_translation_error_multiplier >= 1.0
        && localization.inlier_count >= config.min_inliers
        && localization.inlier_ratio.is_finite()
        && localization.inlier_ratio >= config.min_inlier_ratio
        && config.max_mean_reprojection_error.is_none_or(|maximum| {
            maximum.is_finite()
                && maximum >= 0.0
                && localization
                    .reprojection_error
                    .is_some_and(|value| value.is_finite() && value <= maximum)
        })
}

fn pose_prior_visual_override_threshold(
    config: crate::PosePriorVisualOverrideConfig,
    localization: &LocalizationResult,
    translation_error: f64,
    rotation_error: f64,
    base_threshold: f64,
) -> Option<f64> {
    let minimum = base_threshold * config.min_translation_error_multiplier;
    let maximum = base_threshold * config.max_translation_error_multiplier;
    (base_threshold.is_finite()
        && base_threshold > 0.0
        && translation_error.is_finite()
        && translation_error >= minimum
        && translation_error <= maximum
        && config.max_rotation_error_radians.is_none_or(|maximum| {
            maximum.is_finite()
                && maximum >= 0.0
                && rotation_error.is_finite()
                && rotation_error <= maximum
        })
        && pose_prior_visual_override_satisfied(config, localization))
    .then_some(maximum)
}

fn covariance_weighted_continuation_pose(
    frame: &Frame,
    map: &VisualMap,
    pose_prior: &Pose,
    estimated_pose: &Pose,
    localization: &LocalizationResult,
    config: crate::PosePriorVisualOverrideConfig,
) -> Option<Pose> {
    let camera = map.cameras.get(&frame.camera_id)?;
    let prior_std = config.continuation_prior_translation_std_m;
    let pixel_sigma = config.continuation_visual_pixel_sigma_px;
    let max_gain = config.continuation_max_gain;
    let max_condition = config.continuation_max_condition_number;
    if !prior_std.is_finite()
        || prior_std <= 0.0
        || !pixel_sigma.is_finite()
        || pixel_sigma <= 0.0
        || !max_gain.is_finite()
        || !(0.0..=1.0).contains(&max_gain)
        || !max_condition.is_finite()
        || max_condition < 1.0
    {
        return None;
    }

    let rotation = estimated_pose.world_to_camera.rotation;
    let center = estimated_pose.camera_center_world().coords;
    let eps = 1.0e-4;
    let mut hessian = Matrix3::<f64>::zeros();
    let mut used = 0usize;
    for landmark_id in &localization.inlier_landmark_ids {
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        let mut jacobian = nalgebra::SMatrix::<f64, 2, 3>::zeros();
        let mut valid = true;
        for axis in 0..3 {
            let mut plus_center = center;
            let mut minus_center = center;
            plus_center[axis] += eps;
            minus_center[axis] -= eps;
            let plus_translation = -(rotation.transform_vector(&plus_center));
            let minus_translation = -(rotation.transform_vector(&minus_center));
            let plus_pose = Pose::from_world_to_camera(rotation, plus_translation);
            let minus_pose = Pose::from_world_to_camera(rotation, minus_translation);
            let Some(plus) = camera.project(&plus_pose.transform_world_point(&landmark.position))
            else {
                valid = false;
                break;
            };
            let Some(minus) = camera.project(&minus_pose.transform_world_point(&landmark.position))
            else {
                valid = false;
                break;
            };
            jacobian.set_column(axis, &((plus - minus) / (2.0 * eps)));
        }
        if valid && jacobian.iter().all(|value| value.is_finite()) {
            hessian += jacobian.transpose() * jacobian / (pixel_sigma * pixel_sigma);
            used += 1;
        }
    }
    if used < 6 {
        return None;
    }
    let eigenvalues = hessian.symmetric_eigen().eigenvalues;
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    if !minimum.is_finite()
        || !maximum.is_finite()
        || minimum <= 0.0
        || maximum / minimum > max_condition
    {
        return None;
    }
    let visual_covariance = hessian.try_inverse()?;
    let prior_variance = prior_std * prior_std;
    let prior_covariance = Matrix3::<f64>::identity() * prior_variance;
    let innovation_covariance = prior_covariance + visual_covariance;
    let mut gain = prior_covariance * innovation_covariance.try_inverse()?;
    gain = (gain + gain.transpose()) * 0.5;
    let gain_maximum = gain
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    if !gain_maximum.is_finite() || gain_maximum <= 0.0 {
        return None;
    }
    if gain_maximum > max_gain {
        gain *= max_gain / gain_maximum;
    }

    let prior_center = pose_prior.camera_center_world().coords;
    let estimated_center = estimated_pose.camera_center_world().coords;
    let fused_center = prior_center + gain * (estimated_center - prior_center);
    let rotation_gain = (gain.trace() / 3.0).clamp(0.0, max_gain);
    let fused_rotation = pose_prior
        .world_to_camera
        .rotation
        .slerp(&estimated_pose.world_to_camera.rotation, rotation_gain);
    let fused_translation = -(fused_rotation.transform_vector(&fused_center));
    Some(Pose::from_world_to_camera(
        fused_rotation,
        fused_translation,
    ))
}

/// Multiplier applied to `max_pose_prior_translation_error` when
/// `pose_jump_gap_scaling` is enabled: the frame gap since the last
/// successful track, capped at `max_multiplier` and floored at 1. A gap of
/// 1 (the common case — the immediately preceding frame tracked
/// successfully) reproduces the unscaled gate.
fn pose_jump_gap_scaling_multiplier(
    frames_since_last_tracking_success: FrameId,
    max_multiplier: usize,
) -> FrameId {
    frames_since_last_tracking_success
        .min(max_multiplier as FrameId)
        .max(1)
}

/// Resolve the reference keyframe for covisibility-based local-map selection.
///
/// Prefers a keyframe whose `frame.id` matches `last_id` exactly (when the
/// last successful frame was itself promoted to a keyframe). Otherwise falls
/// back to the keyframe with the largest `frame.id <= last_id` so we still
/// anchor to a temporally-nearby past keyframe.
fn covisibility_pick_reference_keyframe(map: &VisualMap, last_id: FrameId) -> Option<u64> {
    if map.keyframes.contains_key(&last_id) {
        return Some(last_id);
    }
    let mut best: Option<u64> = None;
    for kf_id in map.keyframes.keys() {
        if *kf_id > last_id {
            continue;
        }
        match best {
            None => best = Some(*kf_id),
            Some(current) if *kf_id > current => best = Some(*kf_id),
            _ => {}
        }
    }
    best
}

/// Compute the covisibility-derived local-map landmark set: union of
/// landmarks observed by the reference keyframe and the keyframes that share
/// at least `min_shared_landmarks` landmarks with it (capped at
/// `max_keyframes` co-visible neighbours, ranked by descending shared count).
fn covisibility_local_map_landmarks(
    map: &VisualMap,
    reference_kf_id: u64,
    config: &CovisibilityLocalMapConfig,
) -> HashSet<u64> {
    let mut local_landmarks: HashSet<u64> = HashSet::new();

    let Some(reference_kf) = map.keyframes.get(&reference_kf_id) else {
        return local_landmarks;
    };

    let reference_landmarks: HashSet<u64> = reference_kf
        .observations
        .iter()
        .map(|obs| obs.landmark_id)
        .collect();
    if reference_landmarks.is_empty() {
        return local_landmarks;
    }
    local_landmarks.extend(reference_landmarks.iter().copied());

    let mut shared_counts: HashMap<u64, usize> = HashMap::new();
    for landmark_id in &reference_landmarks {
        let Some(landmark) = map.landmarks.get(landmark_id) else {
            continue;
        };
        for obs in &landmark.observations {
            let kf_id = obs.frame_id;
            if kf_id == reference_kf_id {
                continue;
            }
            if !map.keyframes.contains_key(&kf_id) {
                continue;
            }
            *shared_counts.entry(kf_id).or_insert(0) += 1;
        }
    }

    let mut ranked: Vec<(u64, usize)> = shared_counts
        .into_iter()
        .filter(|(_, count)| *count >= config.min_shared_landmarks)
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if let Some(cap) = config.max_keyframes {
        ranked.truncate(cap);
    }

    for (kf_id, _) in &ranked {
        if let Some(kf) = map.keyframes.get(kf_id) {
            for obs in &kf.observations {
                local_landmarks.insert(obs.landmark_id);
            }
        }
    }

    local_landmarks
}

impl<X, P, M> ImageTracker<X, Tracker<P, M>>
where
    X: FeatureExtractor,
    P: FrameLocalizer,
    M: MotionModel,
{
    pub fn reset(&mut self) {
        self.tracker.reset();
    }

    pub fn track_frame_image(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
    ) -> Result<TrackingResult, X::Error> {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frame_image_with_descriptor_store(
            frame_id,
            camera_id,
            image,
            map,
            &descriptor_store,
        )
    }

    pub fn track_frame_images<'a, I>(
        &mut self,
        frames: I,
        map: &VisualMap,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
    {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.track_frame_images_with_descriptor_store(frames, map, &descriptor_store)
    }

    pub fn track_frame_image_with_descriptor_store(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<TrackingResult, X::Error> {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self
            .tracker
            .track_frame_with_descriptor_store(&frame, map, descriptor_store))
    }

    pub fn track_frame_images_with_descriptor_store<'a, I>(
        &mut self,
        frames: I,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
    {
        let mut results = Vec::new();
        for (frame_id, camera_id, image) in frames {
            results.push(self.track_frame_image_with_descriptor_store(
                frame_id,
                camera_id,
                image,
                map,
                descriptor_store,
            )?);
        }
        Ok(results)
    }

    pub fn track_frame_image_with_provider<P2>(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
    ) -> Result<TrackingResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self.tracker.track_frame_with_provider(&frame, provider))
    }

    pub fn track_frame_images_with_provider<'a, I, P2>(
        &mut self,
        frames: I,
        provider: &P2,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
        P2: MapProvider + DescriptorProvider,
    {
        let mut results = Vec::new();
        for (frame_id, camera_id, image) in frames {
            results
                .push(self.track_frame_image_with_provider(frame_id, camera_id, image, provider)?);
        }
        Ok(results)
    }

    pub fn track_frame_image_with_prior_submap_provider<P2>(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
        radius: f64,
    ) -> Result<TrackingResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self
            .tracker
            .track_frame_with_prior_submap_provider(&frame, provider, radius))
    }

    pub fn track_frame_image_with_localization_prior_submap_provider<P2>(
        &mut self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
        prior: &LocalizationPrior,
    ) -> Result<TrackingResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        let features = self.extractor.extract(image)?;
        let frame = frame_from_features(frame_id, camera_id, features);
        Ok(self
            .tracker
            .track_frame_with_localization_prior_submap_provider(&frame, provider, prior))
    }

    pub fn track_frame_images_with_prior_submap_provider<'a, I, P2>(
        &mut self,
        frames: I,
        provider: &P2,
        radius: f64,
    ) -> Result<Vec<TrackingResult>, X::Error>
    where
        X::Image: 'a,
        I: IntoIterator<Item = (FrameId, CameraId, &'a X::Image)>,
        P2: MapProvider + DescriptorProvider,
    {
        let mut results = Vec::new();
        for (frame_id, camera_id, image) in frames {
            results.push(self.track_frame_image_with_prior_submap_provider(
                frame_id, camera_id, image, provider, radius,
            )?);
        }
        Ok(results)
    }
}

fn frame_from_features(frame_id: FrameId, camera_id: CameraId, features: FeatureSet) -> Frame {
    Frame {
        id: frame_id,
        camera_id,
        keypoints: features.keypoints,
        descriptors: features.descriptors,
        pose: None,
    }
}

pub trait FrameLocalizer {
    fn localize_frame_with_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult;

    fn localize_frame_with_pose_prior_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        _pose_prior: Option<&Pose>,
        _candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        self.localize_frame_with_descriptor_store(frame, map, descriptor_store)
    }

    /// Variant that, in addition to the radius candidate filter, ALSO threads
    /// the pose prior into the PnP RANSAC as a warm-start hypothesis. Default
    /// impl falls back to the non-warm-start variant so existing implementors
    /// don't need to change.
    fn localize_frame_with_pose_prior_warm_start_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
        candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        self.localize_frame_with_pose_prior_and_descriptor_store(
            frame,
            map,
            descriptor_store,
            pose_prior,
            candidate_radius,
        )
    }

    /// ORB-SLAM3-style projection-guided variant: for each candidate
    /// landmark, project it into the frame with `pose_prior` and match its
    /// descriptor only against query keypoints within `search_radius_px` of
    /// the projection (rather than an appearance-global search over the
    /// whole candidate set). Default implementation ignores the projection
    /// window and falls back to the appearance-global
    /// `_pose_prior_and_descriptor_store` variant, so a `FrameLocalizer`
    /// implementor that hasn't opted in still produces a (non-projection)
    /// result the tracker's widen-retry ladder can fall through to.
    fn localize_frame_with_projection_window_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        candidate_radius: Option<f64>,
        _search_radius_px: f64,
    ) -> LocalizationResult {
        self.localize_frame_with_pose_prior_and_descriptor_store(
            frame,
            map,
            descriptor_store,
            Some(pose_prior),
            candidate_radius,
        )
    }

    /// Projection-guided localization with an additional reverse
    /// query-keypoint -> landmark ambiguity ratio. Implementors without a
    /// specialized path retain the projection/default behavior above.
    #[allow(clippy::too_many_arguments)]
    fn localize_frame_with_projection_window_descriptor_store_and_query_landmark_ratio(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        candidate_radius: Option<f64>,
        search_radius_px: f64,
        _max_query_landmark_distance_ratio: Option<f32>,
    ) -> LocalizationResult {
        self.localize_frame_with_projection_window_and_descriptor_store(
            frame,
            map,
            descriptor_store,
            pose_prior,
            candidate_radius,
            search_radius_px,
        )
    }

    /// Stage-3 local-map refinement: given an already-successful `estimated`
    /// localization, project `local_map_descriptor_store`'s landmarks with
    /// the ESTIMATED pose and harvest additional correspondences within
    /// `refinement_search_radius_px`, then re-run pose estimation over the
    /// union with `estimated`'s inliers. Returns `None` when the implementor
    /// doesn't support refinement (default) or no usable correspondences
    /// could be harvested; callers (see `Tracker`) independently gate
    /// acceptance on inlier count, since this method does not compare
    /// against `estimated` itself.
    fn refine_pose_with_local_map_and_descriptor_store(
        &self,
        _frame: &Frame,
        _map: &VisualMap,
        _local_map_descriptor_store: &LandmarkDescriptorStore,
        _estimated: &LocalizationResult,
        _refinement_search_radius_px: f64,
    ) -> Option<LocalizationResult> {
        None
    }

    /// Rebuild projection-guided correspondences from scratch around the
    /// latest pose so 2D keypoints can change landmark assignment between
    /// rounds. Implementors without a specialized revision path retain the
    /// conservative refinement behavior.
    fn revise_pose_with_local_map_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        local_map_descriptor_store: &LandmarkDescriptorStore,
        estimated: &LocalizationResult,
        refinement_search_radius_px: f64,
    ) -> Option<LocalizationResult> {
        self.refine_pose_with_local_map_and_descriptor_store(
            frame,
            map,
            local_map_descriptor_store,
            estimated,
            refinement_search_radius_px,
        )
    }
}

impl<M, S, E> FrameLocalizer for LocalizationPipeline<M, S, E>
where
    M: Matcher + Clone,
    S: CandidateSelector + Clone,
    E: RobustPoseEstimator + Clone,
{
    fn localize_frame_with_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult {
        LocalizationPipeline::localize_frame_with_descriptor_store(
            self,
            frame,
            map,
            descriptor_store,
        )
    }

    fn localize_frame_with_pose_prior_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
        candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        let Some(radius) = candidate_radius else {
            return self.localize_frame_with_descriptor_store(frame, map, descriptor_store);
        };
        let Some(pose_prior) = pose_prior else {
            return self.localize_frame_with_descriptor_store(frame, map, descriptor_store);
        };

        let radius_selector = RadiusLandmarkSelector::new(pose_prior.camera_center_world(), radius);
        let candidate_selector =
            IntersectCandidateSelector::new(self.candidate_selector.clone(), radius_selector);

        LocalizationPipeline::localize_frame_with_candidate_selector_and_descriptor_store(
            self,
            frame,
            map,
            descriptor_store,
            candidate_selector,
        )
    }

    fn localize_frame_with_pose_prior_warm_start_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: Option<&Pose>,
        candidate_radius: Option<f64>,
    ) -> LocalizationResult {
        let Some(pose_prior) = pose_prior else {
            return self.localize_frame_with_descriptor_store(frame, map, descriptor_store);
        };
        let Some(camera) = map.cameras.get(&frame.camera_id).cloned() else {
            return LocalizationResult::failure(
                LocalizationFailureReason::MissingCamera {
                    camera_id: frame.camera_id,
                },
                0,
                0,
                0,
            );
        };
        let query = QueryImage::from_frame(frame, camera);

        if let Some(radius) = candidate_radius {
            let radius_selector =
                RadiusLandmarkSelector::new(pose_prior.camera_center_world(), radius);
            let candidate_selector =
                IntersectCandidateSelector::new(self.candidate_selector.clone(), radius_selector);
            self.localize_with_candidate_selector_and_descriptor_store_and_pose_prior(
                &query,
                map,
                descriptor_store,
                candidate_selector,
                Some(pose_prior),
            )
        } else {
            self.localize_with_candidate_selector_and_descriptor_store_and_pose_prior(
                &query,
                map,
                descriptor_store,
                self.candidate_selector.clone(),
                Some(pose_prior),
            )
        }
    }

    fn localize_frame_with_projection_window_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        candidate_radius: Option<f64>,
        search_radius_px: f64,
    ) -> LocalizationResult {
        LocalizationPipeline::localize_frame_with_projection_window_and_descriptor_store(
            self,
            frame,
            map,
            descriptor_store,
            pose_prior,
            candidate_radius,
            search_radius_px,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn localize_frame_with_projection_window_descriptor_store_and_query_landmark_ratio(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        candidate_radius: Option<f64>,
        search_radius_px: f64,
        max_query_landmark_distance_ratio: Option<f32>,
    ) -> LocalizationResult {
        LocalizationPipeline::localize_frame_with_projection_window_descriptor_store_and_query_landmark_ratio(
            self,
            frame,
            map,
            descriptor_store,
            pose_prior,
            candidate_radius,
            search_radius_px,
            max_query_landmark_distance_ratio,
        )
    }

    fn refine_pose_with_local_map_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        local_map_descriptor_store: &LandmarkDescriptorStore,
        estimated: &LocalizationResult,
        refinement_search_radius_px: f64,
    ) -> Option<LocalizationResult> {
        LocalizationPipeline::refine_frame_pose_with_local_map_and_descriptor_store(
            self,
            frame,
            map,
            local_map_descriptor_store,
            estimated,
            refinement_search_radius_px,
        )
    }

    fn revise_pose_with_local_map_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        local_map_descriptor_store: &LandmarkDescriptorStore,
        estimated: &LocalizationResult,
        refinement_search_radius_px: f64,
    ) -> Option<LocalizationResult> {
        LocalizationPipeline::revise_frame_pose_with_local_map_and_descriptor_store(
            self,
            frame,
            map,
            local_map_descriptor_store,
            estimated,
            refinement_search_radius_px,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackingStats {
    pub first_frame_id: Option<FrameId>,
    pub last_frame_id: Option<FrameId>,
    pub frame_count: usize,
    pub successful_frame_count: usize,
    pub failed_frame_count: usize,
    pub lost_count: usize,
    pub relocalization_count: usize,
    pub pose_prior_used_count: usize,
    pub external_localization_prior_used_count: usize,
    pub tracking_quality_gate_failure_count: usize,
    /// Number of frames accepted only because a strong visual solution
    /// activated the bounded pose-prior translation-gate widening.
    pub pose_prior_visual_override_count: usize,
    pub total_inlier_count: usize,
    pub total_correspondence_count: usize,
    pub covisibility_local_map_used_count: usize,
    /// Number of stage-1 projection-guided pose-estimation attempts made
    /// across all frames (one per widen-retry rung tried, including the
    /// first). Zero when `projection_guided_tracking` is disabled or no pose
    /// prior was available for a frame. Populated live by `Tracker`; unlike
    /// most other counters here, NOT reconstructed by
    /// [`TrackingStats::from_results`] (the per-attempt detail is not carried
    /// on `TrackingResult`).
    pub projection_guided_attempt_count: usize,
    /// Number of widen-retry rungs actually taken (i.e. attempts beyond the
    /// first that used a widened radius after the previous rung failed the
    /// quality gate).
    pub projection_guided_widen_retry_count: usize,
    /// Number of frames where a projection-guided attempt (at any widen
    /// rung) produced the accepted pose estimate.
    pub projection_guided_success_count: usize,
    /// Number of frames where every projection-guided attempt failed and
    /// the appearance-global fallback path produced the accepted pose
    /// estimate.
    pub projection_guided_fallback_success_count: usize,
    /// Sum, across frames where local-map refinement ran, of additional
    /// correspondences harvested beyond the pre-refinement correspondence
    /// count (0 when refinement harvested nothing new).
    pub local_map_refinement_correspondence_gain_total: usize,
    /// Number of frames where local-map refinement ran and its result was
    /// accepted (inlier count did not decrease).
    pub local_map_refinement_accepted_count: usize,
    /// Number of frames where local-map refinement ran but was rejected
    /// (inlier count decreased, or the refined estimate failed its own
    /// quality gate).
    pub local_map_refinement_rejected_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackingEvaluationConfig {
    pub min_success_rate: Option<f64>,
    pub max_failure_rate: Option<f64>,
    pub max_lost_count: Option<usize>,
    pub max_tracking_quality_gate_failure_count: Option<usize>,
    pub min_external_localization_prior_usage_rate: Option<f64>,
    pub min_overall_inlier_ratio: Option<f64>,
    pub min_mean_inliers_per_successful_frame: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingEvaluationResult {
    pub passed: bool,
    pub stats: TrackingStats,
    pub config: TrackingEvaluationConfig,
    pub failures: Vec<TrackingEvaluationFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackingEvaluationFailure {
    SuccessRateTooLow { actual: f64, minimum: f64 },
    FailureRateTooHigh { actual: f64, maximum: f64 },
    LostCountTooHigh { actual: usize, maximum: usize },
    QualityGateFailureCountTooHigh { actual: usize, maximum: usize },
    ExternalLocalizationPriorUsageRateTooLow { actual: f64, minimum: f64 },
    OverallInlierRatioTooLow { actual: f64, minimum: f64 },
    MeanInliersPerSuccessfulFrameTooLow { actual: f64, minimum: f64 },
}

impl TrackingStats {
    pub fn from_results(results: &[TrackingResult]) -> Self {
        let mut stats = Self::default();
        for result in results {
            if stats.first_frame_id.is_none() {
                stats.first_frame_id = Some(result.frame_id);
            }
            stats.last_frame_id = Some(result.frame_id);
            stats.frame_count += 1;
            if result.localization.success {
                stats.successful_frame_count += 1;
            } else {
                stats.failed_frame_count += 1;
            }
            if result.event == TrackingEvent::Lost {
                stats.lost_count += 1;
            }
            if result.event == TrackingEvent::Relocalized {
                stats.relocalization_count += 1;
            }
            if result.used_pose_prior {
                stats.pose_prior_used_count += 1;
            }
            if result.used_external_localization_prior {
                stats.external_localization_prior_used_count += 1;
            }
            if result.tracking_failure_reason.is_some() {
                stats.tracking_quality_gate_failure_count += 1;
            }
            stats.total_inlier_count += result.localization.inlier_count;
            stats.total_correspondence_count += result.localization.correspondence_count;
            if result.covisibility_local_map_size.is_some() {
                stats.covisibility_local_map_used_count += 1;
            }
        }
        stats
    }

    pub fn success_rate(&self) -> f64 {
        ratio(self.successful_frame_count, self.frame_count)
    }

    pub fn failure_rate(&self) -> f64 {
        ratio(self.failed_frame_count, self.frame_count)
    }

    pub fn pose_prior_usage_rate(&self) -> f64 {
        ratio(self.pose_prior_used_count, self.frame_count)
    }

    pub fn external_localization_prior_usage_rate(&self) -> f64 {
        ratio(
            self.external_localization_prior_used_count,
            self.frame_count,
        )
    }

    pub fn overall_inlier_ratio(&self) -> f64 {
        ratio(self.total_inlier_count, self.total_correspondence_count)
    }

    pub fn mean_inliers_per_successful_frame(&self) -> f64 {
        ratio(self.total_inlier_count, self.successful_frame_count)
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"first_frame_id\": {},\n",
                "  \"last_frame_id\": {},\n",
                "  \"frame_count\": {},\n",
                "  \"successful_frame_count\": {},\n",
                "  \"failed_frame_count\": {},\n",
                "  \"lost_count\": {},\n",
                "  \"relocalization_count\": {},\n",
                "  \"pose_prior_used_count\": {},\n",
                "  \"external_localization_prior_used_count\": {},\n",
                "  \"tracking_quality_gate_failure_count\": {},\n",
                "  \"total_inlier_count\": {},\n",
                "  \"total_correspondence_count\": {},\n",
                "  \"success_rate\": {},\n",
                "  \"failure_rate\": {},\n",
                "  \"pose_prior_usage_rate\": {},\n",
                "  \"external_localization_prior_usage_rate\": {},\n",
                "  \"overall_inlier_ratio\": {},\n",
                "  \"mean_inliers_per_successful_frame\": {},\n",
                "  \"projection_guided_attempt_count\": {},\n",
                "  \"projection_guided_widen_retry_count\": {},\n",
                "  \"projection_guided_success_count\": {},\n",
                "  \"projection_guided_fallback_success_count\": {},\n",
                "  \"local_map_refinement_correspondence_gain_total\": {},\n",
                "  \"local_map_refinement_accepted_count\": {},\n",
                "  \"local_map_refinement_rejected_count\": {}\n",
                "}}\n"
            ),
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.frame_count,
            self.successful_frame_count,
            self.failed_frame_count,
            self.lost_count,
            self.relocalization_count,
            self.pose_prior_used_count,
            self.external_localization_prior_used_count,
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.external_localization_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
            self.projection_guided_attempt_count,
            self.projection_guided_widen_retry_count,
            self.projection_guided_success_count,
            self.projection_guided_fallback_success_count,
            self.local_map_refinement_correspondence_gain_total,
            self.local_map_refinement_accepted_count,
            self.local_map_refinement_rejected_count,
        )
    }

    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"first_frame_id\": {}, ",
                "\"last_frame_id\": {}, ",
                "\"frame_count\": {}, ",
                "\"successful_frame_count\": {}, ",
                "\"failed_frame_count\": {}, ",
                "\"lost_count\": {}, ",
                "\"relocalization_count\": {}, ",
                "\"pose_prior_used_count\": {}, ",
                "\"external_localization_prior_used_count\": {}, ",
                "\"tracking_quality_gate_failure_count\": {}, ",
                "\"total_inlier_count\": {}, ",
                "\"total_correspondence_count\": {}, ",
                "\"success_rate\": {}, ",
                "\"failure_rate\": {}, ",
                "\"pose_prior_usage_rate\": {}, ",
                "\"external_localization_prior_usage_rate\": {}, ",
                "\"overall_inlier_ratio\": {}, ",
                "\"mean_inliers_per_successful_frame\": {}, ",
                "\"projection_guided_attempt_count\": {}, ",
                "\"projection_guided_widen_retry_count\": {}, ",
                "\"projection_guided_success_count\": {}, ",
                "\"projection_guided_fallback_success_count\": {}, ",
                "\"local_map_refinement_correspondence_gain_total\": {}, ",
                "\"local_map_refinement_accepted_count\": {}, ",
                "\"local_map_refinement_rejected_count\": {}",
                "}}"
            ),
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.frame_count,
            self.successful_frame_count,
            self.failed_frame_count,
            self.lost_count,
            self.relocalization_count,
            self.pose_prior_used_count,
            self.external_localization_prior_used_count,
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.external_localization_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
            self.projection_guided_attempt_count,
            self.projection_guided_widen_retry_count,
            self.projection_guided_success_count,
            self.projection_guided_fallback_success_count,
            self.local_map_refinement_correspondence_gain_total,
            self.local_map_refinement_accepted_count,
            self.local_map_refinement_rejected_count,
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn evaluate(&self, config: TrackingEvaluationConfig) -> TrackingEvaluationResult {
        TrackingEvaluationResult::from_stats(self.clone(), config)
    }
}

impl TrackingEvaluationResult {
    pub fn from_stats(stats: TrackingStats, config: TrackingEvaluationConfig) -> Self {
        let mut failures = Vec::new();

        if let Some(minimum) = config.min_success_rate {
            let actual = stats.success_rate();
            if actual < minimum {
                failures.push(TrackingEvaluationFailure::SuccessRateTooLow { actual, minimum });
            }
        }

        if let Some(maximum) = config.max_failure_rate {
            let actual = stats.failure_rate();
            if actual > maximum {
                failures.push(TrackingEvaluationFailure::FailureRateTooHigh { actual, maximum });
            }
        }

        if let Some(maximum) = config.max_lost_count {
            if stats.lost_count > maximum {
                failures.push(TrackingEvaluationFailure::LostCountTooHigh {
                    actual: stats.lost_count,
                    maximum,
                });
            }
        }

        if let Some(maximum) = config.max_tracking_quality_gate_failure_count {
            if stats.tracking_quality_gate_failure_count > maximum {
                failures.push(TrackingEvaluationFailure::QualityGateFailureCountTooHigh {
                    actual: stats.tracking_quality_gate_failure_count,
                    maximum,
                });
            }
        }

        if let Some(minimum) = config.min_external_localization_prior_usage_rate {
            let actual = stats.external_localization_prior_usage_rate();
            if actual < minimum {
                failures.push(
                    TrackingEvaluationFailure::ExternalLocalizationPriorUsageRateTooLow {
                        actual,
                        minimum,
                    },
                );
            }
        }

        if let Some(minimum) = config.min_overall_inlier_ratio {
            let actual = stats.overall_inlier_ratio();
            if actual < minimum {
                failures
                    .push(TrackingEvaluationFailure::OverallInlierRatioTooLow { actual, minimum });
            }
        }

        if let Some(minimum) = config.min_mean_inliers_per_successful_frame {
            let actual = stats.mean_inliers_per_successful_frame();
            if actual < minimum {
                failures.push(
                    TrackingEvaluationFailure::MeanInliersPerSuccessfulFrameTooLow {
                        actual,
                        minimum,
                    },
                );
            }
        }

        let passed = failures.is_empty();
        Self {
            passed,
            stats,
            config,
            failures,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"config\": {},\n",
                "  \"stats\": {},\n",
                "  \"failures\": {}\n",
                "}}\n"
            ),
            self.passed,
            self.config.to_json_inline(),
            self.stats.to_json_inline(),
            tracking_evaluation_failures_json(&self.failures)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

impl TrackingEvaluationConfig {
    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"min_success_rate\": {}, ",
                "\"max_failure_rate\": {}, ",
                "\"max_lost_count\": {}, ",
                "\"max_tracking_quality_gate_failure_count\": {}, ",
                "\"min_external_localization_prior_usage_rate\": {}, ",
                "\"min_overall_inlier_ratio\": {}, ",
                "\"min_mean_inliers_per_successful_frame\": {}",
                "}}"
            ),
            optional_f64_json(self.min_success_rate),
            optional_f64_json(self.max_failure_rate),
            optional_usize_json(self.max_lost_count),
            optional_usize_json(self.max_tracking_quality_gate_failure_count),
            optional_f64_json(self.min_external_localization_prior_usage_rate),
            optional_f64_json(self.min_overall_inlier_ratio),
            optional_f64_json(self.min_mean_inliers_per_successful_frame)
        )
    }
}

impl TrackingEvaluationFailure {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::SuccessRateTooLow { .. } => "success_rate_too_low",
            Self::FailureRateTooHigh { .. } => "failure_rate_too_high",
            Self::LostCountTooHigh { .. } => "lost_count_too_high",
            Self::QualityGateFailureCountTooHigh { .. } => "quality_gate_failure_count_too_high",
            Self::ExternalLocalizationPriorUsageRateTooLow { .. } => {
                "external_localization_prior_usage_rate_too_low"
            }
            Self::OverallInlierRatioTooLow { .. } => "overall_inlier_ratio_too_low",
            Self::MeanInliersPerSuccessfulFrameTooLow { .. } => {
                "mean_inliers_per_successful_frame_too_low"
            }
        }
    }

    fn to_json_inline(&self) -> String {
        match self {
            Self::SuccessRateTooLow { actual, minimum }
            | Self::ExternalLocalizationPriorUsageRateTooLow { actual, minimum }
            | Self::OverallInlierRatioTooLow { actual, minimum }
            | Self::MeanInliersPerSuccessfulFrameTooLow { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::FailureRateTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
            Self::LostCountTooHigh { actual, maximum }
            | Self::QualityGateFailureCountTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
        }
    }
}

fn tracking_evaluation_failures_json(failures: &[TrackingEvaluationFailure]) -> String {
    let mut output = String::from("[");
    for (index, failure) in failures.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&failure.to_json_inline());
    }
    output.push(']');
    output
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod covisibility_local_map_tests {
    use super::*;
    use nalgebra::Point2;
    use visloc_core::types::{Camera, Keyframe, Landmark, Observation};

    fn make_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn make_keyframe(id: u64) -> Keyframe {
        Keyframe {
            frame: Frame::new(id, 1),
            observations: Vec::new(),
        }
    }

    fn make_landmark(id: u64) -> Landmark {
        Landmark::new(id, Point3::new(0.0, 0.0, 5.0))
    }

    fn link_observation(map: &mut VisualMap, kf_id: u64, landmark_id: u64) {
        let obs = Observation {
            frame_id: kf_id,
            landmark_id,
            keypoint_index: 0,
            xy: Point2::new(0.0, 0.0),
        };
        if let Some(kf) = map.keyframes.get_mut(&kf_id) {
            kf.observations.push(obs.clone());
        }
        if let Some(lm) = map.landmarks.get_mut(&landmark_id) {
            lm.observations.push(obs);
        }
    }

    fn make_three_kf_map() -> VisualMap {
        let mut map = VisualMap::new();
        let camera = make_camera();
        map.cameras.insert(camera.id, camera);
        for kf_id in [1, 2, 3] {
            map.keyframes.insert(kf_id, make_keyframe(kf_id));
        }
        for lm_id in 1..=20 {
            map.landmarks.insert(lm_id, make_landmark(lm_id));
        }
        for lm_id in 1..=10 {
            link_observation(&mut map, 1, lm_id);
            link_observation(&mut map, 2, lm_id);
        }
        for lm_id in 11..=15 {
            link_observation(&mut map, 2, lm_id);
            link_observation(&mut map, 3, lm_id);
        }
        for lm_id in 16..=20 {
            link_observation(&mut map, 3, lm_id);
        }
        map
    }

    #[test]
    fn picks_exact_keyframe_when_last_frame_id_matches() {
        let map = make_three_kf_map();
        assert_eq!(covisibility_pick_reference_keyframe(&map, 2), Some(2));
    }

    #[test]
    fn picks_nearest_prior_keyframe_when_last_frame_id_misses() {
        let map = make_three_kf_map();
        assert_eq!(covisibility_pick_reference_keyframe(&map, 5), Some(3));
    }

    #[test]
    fn returns_none_when_no_keyframe_is_in_past() {
        let map = make_three_kf_map();
        assert!(covisibility_pick_reference_keyframe(&map, 0).is_none());
    }

    #[test]
    fn covisibility_local_map_includes_reference_landmarks_and_neighbours() {
        let map = make_three_kf_map();
        let config = CovisibilityLocalMapConfig {
            max_keyframes: Some(5),
            min_shared_landmarks: 1,
            min_local_map_landmarks: 1,
        };
        let local = covisibility_local_map_landmarks(&map, 2, &config);
        // KF=2 sees landmarks {1..=15}. Co-visible with KF=1 (shares 1..=10) and
        // KF=3 (shares 11..=15). Union ∪ KF1 ∪ KF3 covers 1..=20.
        for lm in 1..=20 {
            assert!(local.contains(&lm), "missing landmark {}", lm);
        }
    }

    #[test]
    fn covisibility_local_map_drops_low_shared_neighbours() {
        let map = make_three_kf_map();
        let config = CovisibilityLocalMapConfig {
            max_keyframes: Some(5),
            min_shared_landmarks: 6, // KF3 shares only 5 with KF2, so it's dropped
            min_local_map_landmarks: 1,
        };
        let local = covisibility_local_map_landmarks(&map, 2, &config);
        // KF1 shares 10 with KF2 → kept (landmarks 1..=10 contributed via KF1).
        // KF3 shares 5 with KF2 → dropped, so landmarks 16..=20 (only in KF3)
        // should NOT appear.
        for lm in 1..=15 {
            assert!(local.contains(&lm), "expected landmark {}", lm);
        }
        for lm in 16..=20 {
            assert!(!local.contains(&lm), "unexpected landmark {}", lm);
        }
    }

    #[test]
    fn covisibility_local_map_respects_max_keyframes_cap() {
        let map = make_three_kf_map();
        let config = CovisibilityLocalMapConfig {
            max_keyframes: Some(1), // only the strongest neighbour
            min_shared_landmarks: 1,
            min_local_map_landmarks: 1,
        };
        let local = covisibility_local_map_landmarks(&map, 2, &config);
        // Strongest neighbour of KF2 is KF1 (10 shared) > KF3 (5 shared).
        // So KF3-only landmarks (16..=20) must NOT appear.
        for lm in 16..=20 {
            assert!(!local.contains(&lm), "unexpected landmark {} from KF3", lm);
        }
        // But reference-only landmarks 11..=15 (only in KF2 + KF3) should still
        // appear because they are in the reference KF.
        for lm in 11..=15 {
            assert!(local.contains(&lm), "expected reference landmark {}", lm);
        }
    }
}

#[cfg(test)]
mod pose_jump_gap_scaling_tests {
    use super::*;

    #[test]
    fn gap_of_one_reproduces_unscaled_gate() {
        // The common case: the immediately preceding frame tracked
        // successfully, so the gap is 1 and the multiplier is a no-op.
        assert_eq!(pose_jump_gap_scaling_multiplier(1, 10), 1);
    }

    #[test]
    fn gap_scales_up_to_the_configured_cap() {
        assert_eq!(pose_jump_gap_scaling_multiplier(3, 10), 3);
        assert_eq!(pose_jump_gap_scaling_multiplier(10, 10), 10);
        assert_eq!(pose_jump_gap_scaling_multiplier(50, 10), 10);
    }

    #[test]
    fn multiplier_never_drops_below_one_even_with_a_zero_cap() {
        assert_eq!(pose_jump_gap_scaling_multiplier(0, 10), 1);
        assert_eq!(pose_jump_gap_scaling_multiplier(5, 0), 1);
    }
}

#[cfg(test)]
mod pose_prior_visual_override_tests {
    use super::*;
    use crate::PosePriorVisualOverrideConfig;
    use visloc_core::types::{
        Camera, Frame, Landmark, LocalizationFailureReason, LocalizationResult,
    };

    fn localization(
        inliers: usize,
        correspondences: usize,
        reprojection: f64,
    ) -> LocalizationResult {
        let mut result = LocalizationResult::failure(
            LocalizationFailureReason::QualityGateFailed,
            correspondences,
            correspondences,
            correspondences,
        );
        result.inlier_count = inliers;
        result.inlier_ratio = inliers as f64 / correspondences as f64;
        result.reprojection_error = Some(reprojection);
        result
    }

    #[test]
    fn strong_visual_solution_enables_bounded_override() {
        let result = localization(150, 200, 1.5);
        assert!(pose_prior_visual_override_satisfied(
            PosePriorVisualOverrideConfig::default(),
            &result
        ));
    }

    #[test]
    fn weak_or_high_error_visual_solution_cannot_override_prior() {
        let config = PosePriorVisualOverrideConfig::default();
        assert!(!pose_prior_visual_override_satisfied(
            config,
            &localization(99, 120, 1.0)
        ));
        assert!(!pose_prior_visual_override_satisfied(
            config,
            &localization(150, 300, 1.0)
        ));
        assert!(!pose_prior_visual_override_satisfied(
            config,
            &localization(150, 200, 3.1)
        ));
    }

    #[test]
    fn invalid_override_config_is_fail_closed() {
        let result = localization(150, 200, 1.0);
        assert!(!pose_prior_visual_override_satisfied(
            PosePriorVisualOverrideConfig {
                max_translation_error_multiplier: 0.5,
                ..PosePriorVisualOverrideConfig::default()
            },
            &result
        ));
    }

    #[test]
    fn hysteresis_rejects_marginal_crossing_but_accepts_cliff_innovation() {
        let result = localization(306, 467, 2.0);
        let config = PosePriorVisualOverrideConfig::default();
        assert_eq!(
            pose_prior_visual_override_threshold(config, &result, 0.216, 0.01, 0.2),
            None
        );
        assert_eq!(
            pose_prior_visual_override_threshold(config, &result, 0.311, 0.005, 0.2),
            Some(0.5)
        );
        assert_eq!(
            pose_prior_visual_override_threshold(config, &result, 0.6, 0.005, 0.2),
            None,
            "bounded override must still reject a teleport"
        );
        assert_eq!(
            pose_prior_visual_override_threshold(config, &result, 0.311, 0.03, 0.2),
            None,
            "large rotation innovation must reject the visual override"
        );
    }

    #[test]
    fn continuation_pose_uses_capped_covariance_gain() {
        let camera = Camera::pinhole(1, 640, 480, 400.0, 400.0, 320.0, 240.0);
        let frame = Frame::new(10, 1);
        let mut map = VisualMap::new();
        map.cameras.insert(1, camera);
        let mut result = localization(12, 12, 1.0);
        for (index, (x, y, z)) in [
            (-1.0, -0.5, 3.0),
            (0.0, -0.5, 3.5),
            (1.0, -0.5, 4.0),
            (-1.0, 0.5, 4.5),
            (0.0, 0.5, 5.0),
            (1.0, 0.5, 5.5),
            (-0.7, -0.2, 6.0),
            (0.7, -0.2, 6.5),
            (-0.7, 0.2, 7.0),
            (0.7, 0.2, 7.5),
            (-0.2, 0.0, 8.0),
            (0.2, 0.0, 8.5),
        ]
        .into_iter()
        .enumerate()
        {
            let id = index as u64 + 1;
            map.landmarks
                .insert(id, Landmark::new(id, Point3::new(x, y, z)));
            result.inlier_landmark_ids.push(id);
        }
        let prior = Pose::identity();
        let estimated =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let fused = covariance_weighted_continuation_pose(
            &frame,
            &map,
            &prior,
            &estimated,
            &result,
            PosePriorVisualOverrideConfig::default(),
        )
        .expect("well-conditioned inlier geometry should yield covariance");
        let x = fused.camera_center_world().x;
        assert!(x > 0.0 && x <= 0.15 + 1.0e-12, "fused x={x}");
    }
}
