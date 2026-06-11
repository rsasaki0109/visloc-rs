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
        self.state = TrackingState::Tracking;
        self.successive_failures = 0;
        self.last_successful_frame_id = Some(result.frame_id);
        self.last_successful_pose = result.localization.pose.clone();
        self.last_result = Some(result.clone());
        self.stats.relocalization_count += 1;
        self.stats.successful_frame_count += 1;
        if result.localization.inlier_count > 0 {
            self.stats.total_inlier_count += result.localization.inlier_count;
            self.stats.total_correspondence_count += result.localization.correspondence_count;
        }
        self.motion_model.observe(&result);
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

        let mut localization = if self.config.pnp_pose_prior_warm_start {
            self.localization_pipeline
                .localize_frame_with_pose_prior_warm_start_and_descriptor_store(
                    frame,
                    map,
                    active_descriptor_store,
                    pose_prior.as_ref(),
                    self.config.last_pose_candidate_radius,
                )
        } else {
            self.localization_pipeline
                .localize_frame_with_pose_prior_and_descriptor_store(
                    frame,
                    map,
                    active_descriptor_store,
                    pose_prior.as_ref(),
                    self.config.last_pose_candidate_radius,
                )
        };
        let tracking_failure_reason =
            self.apply_tracking_quality_gate(pose_prior.as_ref(), &mut localization);

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

        self.update_history(&result);
        self.motion_model.observe(&result);
        result
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
        &self,
        pose_prior: Option<&Pose>,
        localization: &mut LocalizationResult,
    ) -> Option<TrackingFailureReason> {
        if !localization.success {
            return None;
        }

        if localization.inlier_count < self.config.min_inliers {
            *localization = localization.clone().rejected_by_quality_gate();
            return Some(TrackingFailureReason::InsufficientInliers {
                inlier_count: localization.inlier_count,
                min_inliers: self.config.min_inliers,
            });
        }

        if localization.inlier_ratio < self.config.min_inlier_ratio {
            *localization = localization.clone().rejected_by_quality_gate();
            return Some(TrackingFailureReason::InlierRatioTooLow {
                inlier_ratio: localization.inlier_ratio,
                min_inlier_ratio: self.config.min_inlier_ratio,
            });
        }

        if let (Some(reprojection_error), Some(max_reprojection_error)) = (
            localization.reprojection_error,
            self.config.max_mean_reprojection_error,
        ) {
            if reprojection_error > max_reprojection_error {
                *localization = localization.clone().rejected_by_quality_gate();
                return Some(TrackingFailureReason::MeanReprojectionErrorTooHigh {
                    reprojection_error,
                    max_reprojection_error,
                });
            }
        }

        let max_translation_error = self.config.max_pose_prior_translation_error?;
        let pose_prior = pose_prior?;
        let estimated_pose = localization.pose.as_ref()?;

        let translation_error =
            (estimated_pose.camera_center_world() - pose_prior.camera_center_world()).norm();
        if translation_error <= max_translation_error {
            return None;
        }

        *localization = localization.clone().rejected_by_quality_gate();
        Some(TrackingFailureReason::PosePriorTranslationErrorExceeded {
            translation_error,
            max_translation_error,
        })
    }
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
    pub total_inlier_count: usize,
    pub total_correspondence_count: usize,
    pub covisibility_local_map_used_count: usize,
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
                "  \"mean_inliers_per_successful_frame\": {}\n",
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
                "\"mean_inliers_per_successful_frame\": {}",
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
