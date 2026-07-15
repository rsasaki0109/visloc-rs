#![forbid(unsafe_code)]
//! Stateless visual localization pipeline composition.
//!
//! The pipeline builds 2D-3D correspondences from query features and a visual
//! map, runs robust pose estimation, applies optional quality gates, and returns
//! diagnostics-rich localization results. Map providers, descriptor providers,
//! candidate selectors, matchers, and pose estimators are all replaceable.

mod correspondence;

use std::collections::HashSet;

use nalgebra::Point3;
use visloc_core::geometry::Pose;
use visloc_core::types::{
    Camera, CameraId, Frame, FrameId, LandmarkDescriptorStore, LandmarkId,
    LocalizationFailureReason, LocalizationResult, LocalizationSuccess, QueryImage, VisualMap,
};
use visloc_vision::features::{FeatureExtractor, FeatureSet};
use visloc_vision::matching::{BruteForceMatcher, Matcher};
use visloc_vision::pnp::Correspondence2D3D;
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};

pub use correspondence::{
    AllLandmarksSelector, CandidateSelector, CorrespondenceBuildError, CorrespondenceBuilder,
    CorrespondenceSet, FixedLandmarkSelector, IntersectCandidateSelector,
    ProjectionCorrespondenceBuilder, RadiusLandmarkSelector,
};

#[derive(Debug, Clone, PartialEq)]
pub struct LocalizationConfig {
    pub ratio: Option<f32>,
    pub ransac_iterations: usize,
    pub reprojection_threshold: f64,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_mean_reprojection_error: Option<f64>,
    pub max_median_reprojection_error: Option<f64>,
    pub max_reprojection_error: Option<f64>,
}

impl Default for LocalizationConfig {
    fn default() -> Self {
        Self {
            ratio: Some(0.8),
            ransac_iterations: 128,
            reprojection_threshold: 4.0,
            min_inliers: 0,
            min_inlier_ratio: 0.0,
            max_mean_reprojection_error: None,
            max_median_reprojection_error: None,
            max_reprojection_error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameLocalizationResult {
    pub frame_id: FrameId,
    pub result: LocalizationResult,
}

pub trait MapProvider {
    fn visual_map(&self) -> &VisualMap;
}

pub trait DescriptorProvider {
    fn landmark_descriptor_store(&self) -> Option<&LandmarkDescriptorStore>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MapProviderStats {
    pub camera_count: usize,
    pub landmark_count: usize,
    pub keyframe_count: usize,
    pub descriptor_count: usize,
}

pub fn map_provider_stats<P>(provider: &P) -> MapProviderStats
where
    P: MapProvider + DescriptorProvider,
{
    let map = provider.visual_map();
    let descriptor_count = provider
        .landmark_descriptor_store()
        .map(LandmarkDescriptorStore::len)
        .unwrap_or_else(|| {
            map.landmarks
                .values()
                .filter(|landmark| landmark.descriptor.is_some())
                .count()
        });

    MapProviderStats {
        camera_count: map.cameras.len(),
        landmark_count: map.landmarks.len(),
        keyframe_count: map.keyframes.len(),
        descriptor_count,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalizationPrior {
    pub pose: Option<Pose>,
    pub position_world: Option<Point3<f64>>,
    pub radius: Option<f64>,
}

impl LocalizationPrior {
    pub fn none() -> Self {
        Self {
            pose: None,
            position_world: None,
            radius: None,
        }
    }

    pub fn from_pose(pose: Pose, radius: f64) -> Self {
        Self {
            pose: Some(pose),
            position_world: None,
            radius: Some(radius),
        }
    }

    pub fn from_position(position_world: Point3<f64>, radius: f64) -> Self {
        Self {
            pose: None,
            position_world: Some(position_world),
            radius: Some(radius),
        }
    }

    pub fn center_world(&self) -> Option<Point3<f64>> {
        if let Some(position_world) = self.position_world {
            Some(position_world)
        } else {
            self.pose.as_ref().map(Pose::camera_center_world)
        }
    }

    pub fn to_radius_submap_selector(&self) -> Option<RadiusSubmapSelector> {
        Some(RadiusSubmapSelector::new(
            self.center_world()?,
            self.radius?,
        ))
    }
}

impl Default for LocalizationPrior {
    fn default() -> Self {
        Self::none()
    }
}

pub trait SubmapSelector<P>
where
    P: MapProvider + DescriptorProvider,
{
    fn select_submap(&self, provider: &P) -> InMemoryMapProvider;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllMapSelector;

impl<P> SubmapSelector<P> for AllMapSelector
where
    P: MapProvider + DescriptorProvider,
{
    fn select_submap(&self, provider: &P) -> InMemoryMapProvider {
        if let Some(descriptor_store) = provider.landmark_descriptor_store() {
            InMemoryMapProvider::with_descriptor_store(
                provider.visual_map().clone(),
                descriptor_store.clone(),
            )
        } else {
            InMemoryMapProvider::new(provider.visual_map().clone())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedLandmarkSubmapSelector {
    pub landmark_ids: Vec<LandmarkId>,
}

impl FixedLandmarkSubmapSelector {
    pub fn new(mut landmark_ids: Vec<LandmarkId>) -> Self {
        landmark_ids.sort_unstable();
        landmark_ids.dedup();
        Self { landmark_ids }
    }
}

impl<P> SubmapSelector<P> for FixedLandmarkSubmapSelector
where
    P: MapProvider + DescriptorProvider,
{
    fn select_submap(&self, provider: &P) -> InMemoryMapProvider {
        InMemoryMapProvider::from_provider_landmarks(provider, self.landmark_ids.iter().copied())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RadiusSubmapSelector {
    pub center_world: Point3<f64>,
    pub radius: f64,
}

impl RadiusSubmapSelector {
    pub fn new(center_world: Point3<f64>, radius: f64) -> Self {
        Self {
            center_world,
            radius,
        }
    }
}

impl<P> SubmapSelector<P> for RadiusSubmapSelector
where
    P: MapProvider + DescriptorProvider,
{
    fn select_submap(&self, provider: &P) -> InMemoryMapProvider {
        InMemoryMapProvider::from_provider_radius(provider, self.center_world, self.radius)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriorSubmapSelector {
    pub prior: LocalizationPrior,
}

impl PriorSubmapSelector {
    pub fn new(prior: LocalizationPrior) -> Self {
        Self { prior }
    }
}

impl<P> SubmapSelector<P> for PriorSubmapSelector
where
    P: MapProvider + DescriptorProvider,
{
    fn select_submap(&self, provider: &P) -> InMemoryMapProvider {
        if let Some(selector) = self.prior.to_radius_submap_selector() {
            selector.select_submap(provider)
        } else {
            AllMapSelector.select_submap(provider)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectableMapProvider<P, S>
where
    P: MapProvider + DescriptorProvider,
    S: SubmapSelector<P>,
{
    pub base_provider: P,
    pub selector: S,
    selected_provider: InMemoryMapProvider,
}

impl<P, S> SelectableMapProvider<P, S>
where
    P: MapProvider + DescriptorProvider,
    S: SubmapSelector<P>,
{
    pub fn new(base_provider: P, selector: S) -> Self {
        let selected_provider = selector.select_submap(&base_provider);
        Self {
            base_provider,
            selector,
            selected_provider,
        }
    }

    pub fn refresh(&mut self) {
        self.selected_provider = self.selector.select_submap(&self.base_provider);
    }

    pub fn selected_provider(&self) -> &InMemoryMapProvider {
        &self.selected_provider
    }
}

impl<P, S> MapProvider for SelectableMapProvider<P, S>
where
    P: MapProvider + DescriptorProvider,
    S: SubmapSelector<P>,
{
    fn visual_map(&self) -> &VisualMap {
        self.selected_provider.visual_map()
    }
}

impl<P, S> DescriptorProvider for SelectableMapProvider<P, S>
where
    P: MapProvider + DescriptorProvider,
    S: SubmapSelector<P>,
{
    fn landmark_descriptor_store(&self) -> Option<&LandmarkDescriptorStore> {
        self.selected_provider.landmark_descriptor_store()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InMemoryMapProvider {
    pub map: VisualMap,
    pub descriptor_store: Option<LandmarkDescriptorStore>,
}

impl InMemoryMapProvider {
    pub fn new(map: VisualMap) -> Self {
        Self {
            map,
            descriptor_store: None,
        }
    }

    pub fn with_descriptor_store(
        map: VisualMap,
        descriptor_store: LandmarkDescriptorStore,
    ) -> Self {
        Self {
            map,
            descriptor_store: Some(descriptor_store),
        }
    }

    pub fn from_provider_landmarks<P, I>(provider: &P, landmark_ids: I) -> Self
    where
        P: MapProvider + DescriptorProvider,
        I: IntoIterator<Item = LandmarkId>,
    {
        let source_map = provider.visual_map();
        let landmark_ids = landmark_ids.into_iter().collect::<HashSet<_>>();
        let mut map = VisualMap::new();
        map.cameras = source_map.cameras.clone();
        map.landmarks = source_map
            .landmarks
            .iter()
            .filter(|(landmark_id, _)| landmark_ids.contains(landmark_id))
            .map(|(landmark_id, landmark)| (*landmark_id, landmark.clone()))
            .collect();
        map.landmark_position_covariances = source_map
            .landmark_position_covariances
            .iter()
            .filter(|(landmark_id, _)| landmark_ids.contains(landmark_id))
            .map(|(landmark_id, covariance)| (*landmark_id, *covariance))
            .collect();
        map.keyframes = source_map
            .keyframes
            .iter()
            .filter_map(|(keyframe_id, keyframe)| {
                let mut keyframe = keyframe.clone();
                keyframe
                    .observations
                    .retain(|observation| landmark_ids.contains(&observation.landmark_id));
                if keyframe.observations.is_empty() {
                    None
                } else {
                    Some((*keyframe_id, keyframe))
                }
            })
            .collect();

        let descriptor_store = descriptor_store_for_submap(provider, &map);
        Self {
            map,
            descriptor_store,
        }
    }

    pub fn from_provider_radius<P>(provider: &P, center_world: Point3<f64>, radius: f64) -> Self
    where
        P: MapProvider + DescriptorProvider,
    {
        let landmark_ids = if radius < 0.0 {
            Vec::new()
        } else {
            let radius_squared = radius * radius;
            provider
                .visual_map()
                .landmarks
                .values()
                .filter(|landmark| {
                    (landmark.position - center_world).norm_squared() <= radius_squared
                })
                .map(|landmark| landmark.id)
                .collect::<Vec<_>>()
        };
        Self::from_provider_landmarks(provider, landmark_ids)
    }
}

impl MapProvider for InMemoryMapProvider {
    fn visual_map(&self) -> &VisualMap {
        &self.map
    }
}

impl DescriptorProvider for InMemoryMapProvider {
    fn landmark_descriptor_store(&self) -> Option<&LandmarkDescriptorStore> {
        self.descriptor_store.as_ref()
    }
}

fn descriptor_store_for_submap<P>(
    provider: &P,
    submap: &VisualMap,
) -> Option<LandmarkDescriptorStore>
where
    P: MapProvider + DescriptorProvider,
{
    let mut descriptor_store = LandmarkDescriptorStore::new();
    if let Some(source_store) = provider.landmark_descriptor_store() {
        // Sort the submap landmark ids before iterating so the
        // insertion order into the new `LandmarkDescriptorStore` is
        // independent of the per-process `HashMap` SipHash seed. The
        // store itself is a `HashMap` but downstream callers should
        // not observe ordering differences across binary builds.
        let mut landmark_ids: Vec<u64> = submap.landmarks.keys().copied().collect();
        landmark_ids.sort();
        for landmark_id in landmark_ids {
            if let Some(descriptor) = source_store.get(landmark_id) {
                descriptor_store.insert(landmark_id, descriptor.to_vec());
            }
        }
    } else {
        descriptor_store = LandmarkDescriptorStore::from_visual_map(submap);
    }

    if descriptor_store.is_empty() {
        None
    } else {
        Some(descriptor_store)
    }
}

#[derive(Debug, Clone)]
pub struct ImageLocalizer<X, P = LocalizationPipeline> {
    pub extractor: X,
    pub pipeline: P,
}

impl<X> ImageLocalizer<X, LocalizationPipeline>
where
    X: FeatureExtractor,
{
    pub fn new(extractor: X) -> Self {
        Self {
            extractor,
            pipeline: LocalizationPipeline::default(),
        }
    }
}

impl<X, P> ImageLocalizer<X, P>
where
    X: FeatureExtractor,
{
    pub fn with_pipeline(extractor: X, pipeline: P) -> Self {
        Self {
            extractor,
            pipeline,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalizationPipeline<M = BruteForceMatcher, S = AllLandmarksSelector, E = PnPRansac> {
    pub matcher: M,
    pub candidate_selector: S,
    pub pose_estimator: E,
    pub config: LocalizationConfig,
}

impl Default for LocalizationPipeline<BruteForceMatcher, AllLandmarksSelector, PnPRansac> {
    fn default() -> Self {
        let config = LocalizationConfig::default();
        Self {
            matcher: BruteForceMatcher {
                ratio: config.ratio,
            },
            candidate_selector: AllLandmarksSelector,
            pose_estimator: PnPRansac {
                pose_estimator: Default::default(),
                iterations: config.ransac_iterations,
                reprojection_threshold: config.reprojection_threshold,
                ..PnPRansac::default()
            },
            config,
        }
    }
}

impl<M> LocalizationPipeline<M, AllLandmarksSelector, PnPRansac>
where
    M: Matcher + Clone,
{
    pub fn new(matcher: M, config: LocalizationConfig) -> Self {
        Self {
            matcher,
            candidate_selector: AllLandmarksSelector,
            pose_estimator: PnPRansac {
                pose_estimator: Default::default(),
                iterations: config.ransac_iterations,
                reprojection_threshold: config.reprojection_threshold,
                ..PnPRansac::default()
            },
            config,
        }
    }
}

impl<M, S> LocalizationPipeline<M, S, PnPRansac>
where
    M: Matcher + Clone,
    S: CandidateSelector + Clone,
{
    pub fn with_candidate_selector(
        matcher: M,
        candidate_selector: S,
        config: LocalizationConfig,
    ) -> Self {
        Self {
            matcher,
            candidate_selector,
            pose_estimator: PnPRansac {
                pose_estimator: Default::default(),
                iterations: config.ransac_iterations,
                reprojection_threshold: config.reprojection_threshold,
                ..PnPRansac::default()
            },
            config,
        }
    }
}

impl<M, S, E> LocalizationPipeline<M, S, E>
where
    M: Matcher + Clone,
    S: CandidateSelector + Clone,
    E: RobustPoseEstimator + Clone,
{
    pub fn with_pose_estimator(
        matcher: M,
        candidate_selector: S,
        pose_estimator: E,
        config: LocalizationConfig,
    ) -> Self {
        Self {
            matcher,
            candidate_selector,
            pose_estimator,
            config,
        }
    }

    pub fn localize(&self, query: &QueryImage, map: &VisualMap) -> LocalizationResult {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.localize_with_descriptor_store(query, map, &descriptor_store)
    }

    pub fn localize_with_provider<P>(&self, query: &QueryImage, provider: &P) -> LocalizationResult
    where
        P: MapProvider + DescriptorProvider,
    {
        let map = provider.visual_map();
        if let Some(descriptor_store) = provider.landmark_descriptor_store() {
            self.localize_with_descriptor_store(query, map, descriptor_store)
        } else {
            let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
            self.localize_with_descriptor_store(query, map, &descriptor_store)
        }
    }

    pub fn localize_image_with_extractor<X>(
        &self,
        image: &X::Image,
        camera: Camera,
        map: &VisualMap,
        extractor: &X,
    ) -> Result<LocalizationResult, X::Error>
    where
        X: FeatureExtractor,
    {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.localize_image_with_extractor_and_descriptor_store(
            image,
            camera,
            map,
            &descriptor_store,
            extractor,
        )
    }

    pub fn localize_image_with_extractor_and_provider<X, P>(
        &self,
        image: &X::Image,
        camera: Camera,
        provider: &P,
        extractor: &X,
    ) -> Result<LocalizationResult, X::Error>
    where
        X: FeatureExtractor,
        P: MapProvider + DescriptorProvider,
    {
        let features = extractor.extract(image)?;
        let query = query_from_features(camera, features);
        Ok(self.localize_with_provider(&query, provider))
    }

    pub fn localize_image_with_extractor_and_descriptor_store<X>(
        &self,
        image: &X::Image,
        camera: Camera,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        extractor: &X,
    ) -> Result<LocalizationResult, X::Error>
    where
        X: FeatureExtractor,
    {
        let features = extractor.extract(image)?;
        let query = query_from_features(camera, features);
        Ok(self.localize_with_descriptor_store(&query, map, descriptor_store))
    }

    pub fn localize_frame(&self, frame: &Frame, map: &VisualMap) -> LocalizationResult {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.localize_frame_with_descriptor_store(frame, map, &descriptor_store)
    }

    pub fn localize_frame_with_provider<P>(&self, frame: &Frame, provider: &P) -> LocalizationResult
    where
        P: MapProvider + DescriptorProvider,
    {
        let map = provider.visual_map();
        if let Some(descriptor_store) = provider.landmark_descriptor_store() {
            self.localize_frame_with_descriptor_store(frame, map, descriptor_store)
        } else {
            let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
            self.localize_frame_with_descriptor_store(frame, map, &descriptor_store)
        }
    }

    pub fn localize_frame_image_with_extractor<X>(
        &self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
        extractor: &X,
    ) -> Result<FrameLocalizationResult, X::Error>
    where
        X: FeatureExtractor,
    {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.localize_frame_image_with_extractor_and_descriptor_store(
            frame_id,
            camera_id,
            image,
            map,
            &descriptor_store,
            extractor,
        )
    }

    pub fn localize_frame_image_with_extractor_and_provider<X, P>(
        &self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P,
        extractor: &X,
    ) -> Result<FrameLocalizationResult, X::Error>
    where
        X: FeatureExtractor,
        P: MapProvider + DescriptorProvider,
    {
        let map = provider.visual_map();
        if let Some(descriptor_store) = provider.landmark_descriptor_store() {
            self.localize_frame_image_with_extractor_and_descriptor_store(
                frame_id,
                camera_id,
                image,
                map,
                descriptor_store,
                extractor,
            )
        } else {
            let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
            self.localize_frame_image_with_extractor_and_descriptor_store(
                frame_id,
                camera_id,
                image,
                map,
                &descriptor_store,
                extractor,
            )
        }
    }

    pub fn localize_frame_image_with_extractor_and_descriptor_store<X>(
        &self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        extractor: &X,
    ) -> Result<FrameLocalizationResult, X::Error>
    where
        X: FeatureExtractor,
    {
        let Some(camera) = map.cameras.get(&camera_id).cloned() else {
            return Ok(FrameLocalizationResult {
                frame_id,
                result: LocalizationResult::failure(
                    LocalizationFailureReason::MissingCamera { camera_id },
                    0,
                    0,
                    0,
                ),
            });
        };
        let result = self.localize_image_with_extractor_and_descriptor_store(
            image,
            camera,
            map,
            descriptor_store,
            extractor,
        )?;
        Ok(FrameLocalizationResult { frame_id, result })
    }

    pub fn localize_frames(
        &self,
        frames: &[Frame],
        map: &VisualMap,
    ) -> Vec<FrameLocalizationResult> {
        let descriptor_store = LandmarkDescriptorStore::from_visual_map(map);
        self.localize_frames_with_descriptor_store(frames, map, &descriptor_store)
    }

    pub fn localize_frame_with_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult {
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
        self.localize_with_descriptor_store(&query, map, descriptor_store)
    }

    pub fn localize_frame_with_candidate_selector_and_descriptor_store<S2>(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        candidate_selector: S2,
    ) -> LocalizationResult
    where
        S2: CandidateSelector + Clone,
    {
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
        self.localize_with_candidate_selector_and_descriptor_store(
            &query,
            map,
            descriptor_store,
            candidate_selector,
        )
    }

    pub fn localize_frames_with_descriptor_store(
        &self,
        frames: &[Frame],
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Vec<FrameLocalizationResult> {
        frames
            .iter()
            .map(|frame| FrameLocalizationResult {
                frame_id: frame.id,
                result: self.localize_frame_with_descriptor_store(frame, map, descriptor_store),
            })
            .collect()
    }

    pub fn localize_with_candidate_selector_and_descriptor_store<S2>(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        candidate_selector: S2,
    ) -> LocalizationResult
    where
        S2: CandidateSelector + Clone,
    {
        self.run_localization(query, map, descriptor_store, candidate_selector, None)
    }

    pub fn localize_with_candidate_selector_and_descriptor_store_and_pose_prior<S2>(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        candidate_selector: S2,
        pose_prior: Option<&Pose>,
    ) -> LocalizationResult
    where
        S2: CandidateSelector + Clone,
    {
        self.run_localization(query, map, descriptor_store, candidate_selector, pose_prior)
    }

    pub fn localize_with_descriptor_store(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult {
        self.run_localization(
            query,
            map,
            descriptor_store,
            self.candidate_selector.clone(),
            None,
        )
    }

    /// ORB-SLAM3-style projection-guided variant of
    /// [`Self::localize_frame_with_pose_prior_and_descriptor_store`]: instead
    /// of an appearance-global descriptor search restricted to a radius
    /// candidate set, each candidate landmark is projected into the frame
    /// with `pose_prior` and matched only against query keypoints within
    /// `search_radius_px` pixels of the projection (see
    /// [`ProjectionCorrespondenceBuilder`]). Falls through to the existing
    /// pose-estimation-and-quality-gate path unchanged.
    pub fn localize_frame_with_projection_window_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        candidate_radius: Option<f64>,
        search_radius_px: f64,
    ) -> LocalizationResult {
        self.localize_frame_with_projection_window_descriptor_store_and_query_landmark_ratio(
            frame,
            map,
            descriptor_store,
            pose_prior,
            candidate_radius,
            search_radius_px,
            None,
        )
    }

    // This compatibility entry point exposes the existing projection-window
    // arguments plus the optional reverse-ratio gate without replacing the
    // shorter public wrapper above.
    #[allow(clippy::too_many_arguments)]
    pub fn localize_frame_with_projection_window_descriptor_store_and_query_landmark_ratio(
        &self,
        frame: &Frame,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        candidate_radius: Option<f64>,
        search_radius_px: f64,
        max_query_landmark_distance_ratio: Option<f32>,
    ) -> LocalizationResult {
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
            self.localize_with_projection_window_descriptor_store_and_query_landmark_ratio(
                &query,
                map,
                descriptor_store,
                candidate_selector,
                pose_prior,
                search_radius_px,
                max_query_landmark_distance_ratio,
            )
        } else {
            self.localize_with_projection_window_descriptor_store_and_query_landmark_ratio(
                &query,
                map,
                descriptor_store,
                self.candidate_selector.clone(),
                pose_prior,
                search_radius_px,
                max_query_landmark_distance_ratio,
            )
        }
    }

    pub fn localize_with_projection_window_and_descriptor_store<S2>(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        candidate_selector: S2,
        pose_prior: &Pose,
        search_radius_px: f64,
    ) -> LocalizationResult
    where
        S2: CandidateSelector + Clone,
    {
        self.localize_with_projection_window_descriptor_store_and_query_landmark_ratio(
            query,
            map,
            descriptor_store,
            candidate_selector,
            pose_prior,
            search_radius_px,
            None,
        )
    }

    // Keep the selector-generic compatibility entry point parallel to the
    // frame wrapper; callers that do not need the reverse ratio use the
    // shorter method above.
    #[allow(clippy::too_many_arguments)]
    pub fn localize_with_projection_window_descriptor_store_and_query_landmark_ratio<S2>(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        candidate_selector: S2,
        pose_prior: &Pose,
        search_radius_px: f64,
        max_query_landmark_distance_ratio: Option<f32>,
    ) -> LocalizationResult
    where
        S2: CandidateSelector + Clone,
    {
        let correspondence_set = match ProjectionCorrespondenceBuilder::with_candidate_selector(
            self.matcher.clone(),
            candidate_selector,
        )
        .build_with_pose_prior_and_query_landmark_ratio(
            query,
            map,
            descriptor_store,
            pose_prior,
            search_radius_px,
            max_query_landmark_distance_ratio,
        ) {
            Ok(correspondence_set) => correspondence_set,
            Err(error) => return result_from_correspondence_error(error),
        };
        self.estimate_and_gate(query, correspondence_set, Some(pose_prior))
    }

    /// Stage-3 local-map refinement: given an already-successful `estimated`
    /// localization for `frame`, project `local_map_descriptor_store`'s
    /// landmarks with the ESTIMATED pose, harvest additional correspondences
    /// within `refinement_search_radius_px`, and re-run pose estimation over
    /// the union with `estimated`'s own inlier correspondences (duplicates
    /// by landmark id are dropped, preferring the already-known-good inlier
    /// entry). Returns `None` when `estimated` carries no pose or the
    /// combined correspondence set is too small for the estimator to run;
    /// callers are responsible for accept/reject decisions (this method does
    /// not compare inlier counts against `estimated`).
    pub fn refine_frame_pose_with_local_map_and_descriptor_store(
        &self,
        frame: &Frame,
        map: &VisualMap,
        local_map_descriptor_store: &LandmarkDescriptorStore,
        estimated: &LocalizationResult,
        refinement_search_radius_px: f64,
    ) -> Option<LocalizationResult> {
        let estimated_pose = estimated.pose.as_ref()?;
        let camera = map.cameras.get(&frame.camera_id)?.clone();
        let query = QueryImage::from_frame(frame, camera.clone());

        let harvested = ProjectionCorrespondenceBuilder::new(self.matcher.clone())
            .build_with_pose_prior(
                &query,
                map,
                local_map_descriptor_store,
                estimated_pose,
                refinement_search_radius_px,
            )
            .ok();

        let mut seen_landmarks: HashSet<LandmarkId> = HashSet::new();
        let mut correspondences = Vec::new();
        let mut query_indices = Vec::new();
        let mut landmark_ids = Vec::new();

        for (&query_index, &landmark_id) in estimated
            .inlier_query_indices
            .iter()
            .zip(estimated.inlier_landmark_ids.iter())
        {
            let Some(point2d) = frame.keypoints.get(query_index).copied() else {
                continue;
            };
            let Some(landmark) = map.landmarks.get(&landmark_id) else {
                continue;
            };
            if !seen_landmarks.insert(landmark_id) {
                continue;
            }
            correspondences.push(Correspondence2D3D {
                point2d,
                point3d: landmark.position,
                confidence: None,
            });
            query_indices.push(query_index);
            landmark_ids.push(landmark_id);
        }

        if let Some(harvested) = &harvested {
            for ((correspondence, &query_index), &landmark_id) in harvested
                .correspondences
                .iter()
                .zip(harvested.query_indices.iter())
                .zip(harvested.landmark_ids.iter())
            {
                if !seen_landmarks.insert(landmark_id) {
                    continue;
                }
                correspondences.push(correspondence.clone());
                query_indices.push(query_index);
                landmark_ids.push(landmark_id);
            }
        }

        let correspondence_count = correspondences.len();
        let confidence_weights = correspondence_confidence_weights(&correspondences);
        let report = self.pose_estimator.estimate_with_pose_prior_and_weights(
            &correspondences,
            &camera,
            Some(estimated_pose),
            confidence_weights.as_deref(),
        )?;

        let inlier_query_indices = report
            .inliers
            .iter()
            .filter_map(|inlier| query_indices.get(*inlier).copied())
            .collect::<Vec<_>>();
        let inlier_landmark_ids = report
            .inliers
            .iter()
            .filter_map(|inlier| landmark_ids.get(*inlier).copied())
            .collect::<Vec<_>>();

        let result = LocalizationResult::success(LocalizationSuccess {
            pose: report.pose,
            candidate_landmark_count: correspondence_count,
            match_count: correspondence_count,
            correspondence_count,
            inliers: report.inliers,
            inlier_query_indices,
            inlier_landmark_ids,
            inlier_reprojection_errors: report.inlier_reprojection_errors,
            mean_reprojection_error: report.mean_reprojection_error,
            median_reprojection_error: report.median_reprojection_error,
            max_reprojection_error: report.max_reprojection_error,
        })
        .with_estimator_diagnostics(report.diagnostics);

        Some(if passes_quality_gate(&result, &self.config) {
            result
        } else {
            result.rejected_by_quality_gate()
        })
    }

    fn run_localization<S2>(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        candidate_selector: S2,
        pose_prior: Option<&Pose>,
    ) -> LocalizationResult
    where
        S2: CandidateSelector + Clone,
    {
        let correspondence_set = match CorrespondenceBuilder::with_candidate_selector(
            self.matcher.clone(),
            candidate_selector,
        )
        .build(query, map, descriptor_store)
        {
            Ok(correspondence_set) => correspondence_set,
            Err(error) => {
                return result_from_correspondence_error(error);
            }
        };
        self.estimate_and_gate(query, correspondence_set, pose_prior)
    }

    /// Shared tail of both `run_localization` (appearance-global) and
    /// `localize_with_projection_window_and_descriptor_store`
    /// (projection-guided): run the configured `RobustPoseEstimator` over an
    /// already-built `CorrespondenceSet` and apply the quality gate. Kept as
    /// one function so PnP/gate behaviour cannot drift between the two
    /// correspondence-building strategies.
    fn estimate_and_gate(
        &self,
        query: &QueryImage,
        correspondence_set: CorrespondenceSet,
        pose_prior: Option<&Pose>,
    ) -> LocalizationResult {
        let match_count = correspondence_set.match_count;
        let candidate_landmark_count = correspondence_set.candidate_landmark_count;
        let correspondence_count = correspondence_set.correspondences.len();

        let confidence_weights =
            correspondence_confidence_weights(&correspondence_set.correspondences);
        let pose_report = if pose_prior.is_some() {
            self.pose_estimator.estimate_with_pose_prior_and_weights(
                &correspondence_set.correspondences,
                &query.camera,
                pose_prior,
                confidence_weights.as_deref(),
            )
        } else if let Some(weights) = confidence_weights.as_deref() {
            self.pose_estimator.estimate_with_weights(
                &correspondence_set.correspondences,
                &query.camera,
                weights,
            )
        } else {
            self.pose_estimator
                .estimate(&correspondence_set.correspondences, &query.camera)
        };

        let Some(report) = pose_report else {
            let result = LocalizationResult::failure(
                LocalizationFailureReason::PoseEstimationFailed {
                    correspondence_count,
                },
                candidate_landmark_count,
                match_count,
                correspondence_count,
            );
            if let Some(diagnostics) = self
                .pose_estimator
                .failure_diagnostics(&correspondence_set.correspondences, &query.camera)
            {
                return result.with_pose_failure_diagnostics(diagnostics);
            }
            return result;
        };

        let inlier_query_indices = report
            .inliers
            .iter()
            .filter_map(|inlier| correspondence_set.query_indices.get(*inlier).copied())
            .collect::<Vec<_>>();
        let inlier_landmark_ids = report
            .inliers
            .iter()
            .filter_map(|inlier| correspondence_set.landmark_ids.get(*inlier).copied())
            .collect::<Vec<_>>();

        let result = LocalizationResult::success(LocalizationSuccess {
            pose: report.pose,
            candidate_landmark_count,
            match_count,
            correspondence_count,
            inliers: report.inliers,
            inlier_query_indices,
            inlier_landmark_ids,
            inlier_reprojection_errors: report.inlier_reprojection_errors,
            mean_reprojection_error: report.mean_reprojection_error,
            median_reprojection_error: report.median_reprojection_error,
            max_reprojection_error: report.max_reprojection_error,
        })
        .with_estimator_diagnostics(report.diagnostics);

        if passes_quality_gate(&result, &self.config) {
            result
        } else {
            result.rejected_by_quality_gate()
        }
    }
}

fn correspondence_confidence_weights(
    correspondences: &[visloc_vision::pnp::Correspondence2D3D],
) -> Option<Vec<f32>> {
    let weights = correspondences
        .iter()
        .map(|correspondence| {
            correspondence
                .confidence
                .filter(|confidence| confidence.is_finite() && *confidence > 0.0)
                .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();
    weights
        .iter()
        .any(|weight| *weight > 0.0)
        .then_some(weights)
}

impl<X, M, S, E> ImageLocalizer<X, LocalizationPipeline<M, S, E>>
where
    X: FeatureExtractor,
    M: Matcher + Clone,
    S: CandidateSelector + Clone,
    E: RobustPoseEstimator + Clone,
{
    pub fn localize_image(
        &self,
        image: &X::Image,
        camera: Camera,
        map: &VisualMap,
    ) -> Result<LocalizationResult, X::Error> {
        self.pipeline
            .localize_image_with_extractor(image, camera, map, &self.extractor)
    }

    pub fn localize_image_with_descriptor_store(
        &self,
        image: &X::Image,
        camera: Camera,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<LocalizationResult, X::Error> {
        self.pipeline
            .localize_image_with_extractor_and_descriptor_store(
                image,
                camera,
                map,
                descriptor_store,
                &self.extractor,
            )
    }

    pub fn localize_image_with_provider<P2>(
        &self,
        image: &X::Image,
        camera: Camera,
        provider: &P2,
    ) -> Result<LocalizationResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        self.pipeline.localize_image_with_extractor_and_provider(
            image,
            camera,
            provider,
            &self.extractor,
        )
    }

    pub fn localize_frame_image(
        &self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
    ) -> Result<FrameLocalizationResult, X::Error> {
        self.pipeline.localize_frame_image_with_extractor(
            frame_id,
            camera_id,
            image,
            map,
            &self.extractor,
        )
    }

    pub fn localize_frame_image_with_descriptor_store(
        &self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<FrameLocalizationResult, X::Error> {
        self.pipeline
            .localize_frame_image_with_extractor_and_descriptor_store(
                frame_id,
                camera_id,
                image,
                map,
                descriptor_store,
                &self.extractor,
            )
    }

    pub fn localize_frame_image_with_provider<P2>(
        &self,
        frame_id: FrameId,
        camera_id: CameraId,
        image: &X::Image,
        provider: &P2,
    ) -> Result<FrameLocalizationResult, X::Error>
    where
        P2: MapProvider + DescriptorProvider,
    {
        self.pipeline
            .localize_frame_image_with_extractor_and_provider(
                frame_id,
                camera_id,
                image,
                provider,
                &self.extractor,
            )
    }
}

pub fn localize(query: QueryImage, map: VisualMap) -> LocalizationResult {
    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac>::default()
        .localize(&query, &map)
}

pub fn localize_frame(frame: Frame, map: VisualMap) -> LocalizationResult {
    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac>::default()
        .localize_frame(&frame, &map)
}

pub fn localize_frames(frames: Vec<Frame>, map: VisualMap) -> Vec<FrameLocalizationResult> {
    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac>::default()
        .localize_frames(&frames, &map)
}

pub fn localize_with_descriptor_store(
    query: QueryImage,
    map: VisualMap,
    descriptor_store: LandmarkDescriptorStore,
) -> LocalizationResult {
    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac>::default()
        .localize_with_descriptor_store(&query, &map, &descriptor_store)
}

pub fn localize_frame_with_descriptor_store(
    frame: Frame,
    map: VisualMap,
    descriptor_store: LandmarkDescriptorStore,
) -> LocalizationResult {
    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac>::default()
        .localize_frame_with_descriptor_store(&frame, &map, &descriptor_store)
}

pub fn localize_frames_with_descriptor_store(
    frames: Vec<Frame>,
    map: VisualMap,
    descriptor_store: LandmarkDescriptorStore,
) -> Vec<FrameLocalizationResult> {
    LocalizationPipeline::<BruteForceMatcher, AllLandmarksSelector, PnPRansac>::default()
        .localize_frames_with_descriptor_store(&frames, &map, &descriptor_store)
}

fn query_from_features(camera: Camera, features: FeatureSet) -> QueryImage {
    QueryImage {
        camera,
        keypoints: features.keypoints,
        descriptors: features.descriptors,
    }
}

fn result_from_correspondence_error(error: CorrespondenceBuildError) -> LocalizationResult {
    match error {
        CorrespondenceBuildError::QueryFeatureShapeMismatch {
            keypoint_count,
            descriptor_count,
        } => LocalizationResult::failure(
            LocalizationFailureReason::QueryFeatureShapeMismatch {
                keypoint_count,
                descriptor_count,
            },
            0,
            0,
            0,
        ),
        CorrespondenceBuildError::NoCandidateLandmarks => {
            LocalizationResult::failure(LocalizationFailureReason::NoCandidateLandmarks, 0, 0, 0)
        }
        CorrespondenceBuildError::NoMapDescriptors {
            candidate_landmark_count,
        } => LocalizationResult::failure(
            LocalizationFailureReason::NoMapDescriptors,
            candidate_landmark_count,
            0,
            0,
        ),
        CorrespondenceBuildError::NoDescriptorMatches {
            candidate_landmark_count,
        } => LocalizationResult::failure(
            LocalizationFailureReason::NoDescriptorMatches,
            candidate_landmark_count,
            0,
            0,
        ),
        CorrespondenceBuildError::InvalidQueryLandmarkRatio => LocalizationResult::failure(
            LocalizationFailureReason::InvalidProjectionQueryLandmarkRatio,
            0,
            0,
            0,
        ),
    }
}

fn passes_quality_gate(result: &LocalizationResult, config: &LocalizationConfig) -> bool {
    if result.inlier_count < config.min_inliers {
        return false;
    }
    if result.inlier_ratio < config.min_inlier_ratio {
        return false;
    }
    if let Some(max_error) = config.max_mean_reprojection_error {
        if result.reprojection_error.unwrap_or(f64::INFINITY) > max_error {
            return false;
        }
    }
    if let Some(max_error) = config.max_median_reprojection_error {
        if result.median_reprojection_error.unwrap_or(f64::INFINITY) > max_error {
            return false;
        }
    }
    if let Some(max_error) = config.max_reprojection_error {
        if result.max_reprojection_error.unwrap_or(f64::INFINITY) > max_error {
            return false;
        }
    }
    true
}
