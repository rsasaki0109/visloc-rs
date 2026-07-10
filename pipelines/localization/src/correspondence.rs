use std::collections::HashSet;

use nalgebra::Point3;
use visloc_core::geometry::Pose;
use visloc_core::types::{LandmarkDescriptorStore, LandmarkId, QueryImage, VisualMap};
use visloc_vision::matching::{DescriptorMatch, Matcher};
use visloc_vision::pnp::Correspondence2D3D;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorrespondenceBuildError {
    QueryFeatureShapeMismatch {
        keypoint_count: usize,
        descriptor_count: usize,
    },
    NoCandidateLandmarks,
    NoMapDescriptors {
        candidate_landmark_count: usize,
    },
    NoDescriptorMatches {
        candidate_landmark_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorrespondenceSet {
    pub correspondences: Vec<Correspondence2D3D>,
    pub query_indices: Vec<usize>,
    pub landmark_ids: Vec<LandmarkId>,
    pub candidate_landmark_count: usize,
    pub match_count: usize,
    pub descriptor_matches: Vec<DescriptorMatch>,
}

pub trait CandidateSelector {
    fn select_landmark_ids(&self, query: &QueryImage, map: &VisualMap) -> Vec<LandmarkId>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllLandmarksSelector;

impl CandidateSelector for AllLandmarksSelector {
    fn select_landmark_ids(&self, _query: &QueryImage, map: &VisualMap) -> Vec<LandmarkId> {
        let mut landmark_ids = map.landmarks.keys().copied().collect::<Vec<_>>();
        landmark_ids.sort_unstable();
        landmark_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedLandmarkSelector {
    landmark_ids: Vec<LandmarkId>,
}

impl FixedLandmarkSelector {
    pub fn new(mut landmark_ids: Vec<LandmarkId>) -> Self {
        landmark_ids.sort_unstable();
        landmark_ids.dedup();
        Self { landmark_ids }
    }
}

impl CandidateSelector for FixedLandmarkSelector {
    fn select_landmark_ids(&self, _query: &QueryImage, _map: &VisualMap) -> Vec<LandmarkId> {
        self.landmark_ids.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntersectCandidateSelector<A, B> {
    pub first: A,
    pub second: B,
}

impl<A, B> IntersectCandidateSelector<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> CandidateSelector for IntersectCandidateSelector<A, B>
where
    A: CandidateSelector,
    B: CandidateSelector,
{
    fn select_landmark_ids(&self, query: &QueryImage, map: &VisualMap) -> Vec<LandmarkId> {
        let first_ids = self
            .first
            .select_landmark_ids(query, map)
            .into_iter()
            .collect::<HashSet<_>>();
        let mut landmark_ids = self
            .second
            .select_landmark_ids(query, map)
            .into_iter()
            .filter(|landmark_id| first_ids.contains(landmark_id))
            .collect::<Vec<_>>();
        landmark_ids.sort_unstable();
        landmark_ids.dedup();
        landmark_ids
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RadiusLandmarkSelector {
    pub center_world: Point3<f64>,
    pub radius: f64,
}

impl RadiusLandmarkSelector {
    pub fn new(center_world: Point3<f64>, radius: f64) -> Self {
        Self {
            center_world,
            radius,
        }
    }
}

impl CandidateSelector for RadiusLandmarkSelector {
    fn select_landmark_ids(&self, _query: &QueryImage, map: &VisualMap) -> Vec<LandmarkId> {
        if self.radius < 0.0 {
            return Vec::new();
        }

        let radius_squared = self.radius * self.radius;
        let mut landmark_ids = map
            .landmarks
            .values()
            .filter(|landmark| {
                (landmark.position - self.center_world).norm_squared() <= radius_squared
            })
            .map(|landmark| landmark.id)
            .collect::<Vec<_>>();
        landmark_ids.sort_unstable();
        landmark_ids
    }
}

#[derive(Debug, Clone)]
pub struct CorrespondenceBuilder<M, S = AllLandmarksSelector> {
    matcher: M,
    candidate_selector: S,
}

impl<M> CorrespondenceBuilder<M, AllLandmarksSelector>
where
    M: Matcher,
{
    pub fn new(matcher: M) -> Self {
        Self {
            matcher,
            candidate_selector: AllLandmarksSelector,
        }
    }
}

impl<M, S> CorrespondenceBuilder<M, S>
where
    M: Matcher,
    S: CandidateSelector,
{
    pub fn with_candidate_selector(matcher: M, candidate_selector: S) -> Self {
        Self {
            matcher,
            candidate_selector,
        }
    }

    pub fn build(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
    ) -> Result<CorrespondenceSet, CorrespondenceBuildError> {
        if query.keypoints.len() != query.descriptors.len() {
            return Err(CorrespondenceBuildError::QueryFeatureShapeMismatch {
                keypoint_count: query.keypoints.len(),
                descriptor_count: query.descriptors.len(),
            });
        }

        let candidate_landmark_ids = self.candidate_selector.select_landmark_ids(query, map);
        if candidate_landmark_ids.is_empty() {
            return Err(CorrespondenceBuildError::NoCandidateLandmarks);
        }

        let candidate_landmark_count = candidate_landmark_ids.len();
        let candidate_landmark_ids = candidate_landmark_ids.into_iter().collect::<HashSet<_>>();
        let paired = descriptor_store
            .ordered_landmark_descriptors(map)
            .into_iter()
            .filter(|(landmark, _)| candidate_landmark_ids.contains(&landmark.id))
            .collect::<Vec<_>>();
        if paired.is_empty() {
            return Err(CorrespondenceBuildError::NoMapDescriptors {
                candidate_landmark_count,
            });
        }

        let (landmarks, descriptors): (Vec<_>, Vec<_>) = paired
            .into_iter()
            .map(|(landmark, descriptor)| (landmark, descriptor.to_vec()))
            .unzip();

        let matches = self
            .matcher
            .match_descriptors(&query.descriptors, &descriptors);
        if matches.is_empty() {
            return Err(CorrespondenceBuildError::NoDescriptorMatches {
                candidate_landmark_count,
            });
        }

        let mut correspondences = Vec::new();
        let mut query_indices = Vec::new();
        let mut landmark_ids = Vec::new();
        for descriptor_match in &matches {
            let Some(point2d) = query.keypoints.get(descriptor_match.query_index).copied() else {
                continue;
            };
            let Some(landmark) = landmarks.get(descriptor_match.train_index) else {
                continue;
            };
            correspondences.push(Correspondence2D3D {
                point2d,
                point3d: landmark.position,
                confidence: descriptor_match.confidence,
            });
            query_indices.push(descriptor_match.query_index);
            landmark_ids.push(landmark.id);
        }

        Ok(CorrespondenceSet {
            correspondences,
            query_indices,
            landmark_ids,
            candidate_landmark_count,
            match_count: matches.len(),
            descriptor_matches: matches,
        })
    }
}

/// ORB-SLAM3-style projection-guided correspondence building: instead of
/// matching every candidate landmark descriptor against every query
/// descriptor (`CorrespondenceBuilder`'s all-vs-all search), each candidate
/// landmark is projected into the query image with a pose prior and matched
/// only against the query keypoints that fall within `search_radius_px` of
/// the projection. This trades the appearance-only search for a much
/// smaller, geometrically-gated candidate window per landmark, which both
/// disambiguates repeated/near-duplicate descriptors (only one is ever in
/// the window) and is materially cheaper when the map is large.
#[derive(Debug, Clone)]
pub struct ProjectionCorrespondenceBuilder<M, S = AllLandmarksSelector> {
    matcher: M,
    candidate_selector: S,
}

impl<M> ProjectionCorrespondenceBuilder<M, AllLandmarksSelector>
where
    M: Matcher,
{
    pub fn new(matcher: M) -> Self {
        Self {
            matcher,
            candidate_selector: AllLandmarksSelector,
        }
    }
}

impl<M, S> ProjectionCorrespondenceBuilder<M, S>
where
    M: Matcher,
    S: CandidateSelector,
{
    pub fn with_candidate_selector(matcher: M, candidate_selector: S) -> Self {
        Self {
            matcher,
            candidate_selector,
        }
    }

    /// Build correspondences by projecting each candidate landmark with
    /// `pose_prior` and matching its descriptor only against query keypoints
    /// within `search_radius_px` (Euclidean, in pixels) of the projection.
    /// Landmarks that project behind the camera, outside the image bounds,
    /// or land in a keypoint-free window contribute no correspondence and
    /// are silently skipped (mirroring how `CorrespondenceBuilder` silently
    /// drops descriptor pairs the matcher rejects). Reuses the caller's
    /// `Matcher` for the actual per-window nearest-neighbour decision, so
    /// ratio-test and cross-check semantics are unchanged from the
    /// appearance-global path.
    pub fn build_with_pose_prior(
        &self,
        query: &QueryImage,
        map: &VisualMap,
        descriptor_store: &LandmarkDescriptorStore,
        pose_prior: &Pose,
        search_radius_px: f64,
    ) -> Result<CorrespondenceSet, CorrespondenceBuildError> {
        if query.keypoints.len() != query.descriptors.len() {
            return Err(CorrespondenceBuildError::QueryFeatureShapeMismatch {
                keypoint_count: query.keypoints.len(),
                descriptor_count: query.descriptors.len(),
            });
        }

        let candidate_landmark_ids = self.candidate_selector.select_landmark_ids(query, map);
        if candidate_landmark_ids.is_empty() {
            return Err(CorrespondenceBuildError::NoCandidateLandmarks);
        }

        let candidate_landmark_count = candidate_landmark_ids.len();
        let candidate_landmark_ids = candidate_landmark_ids.into_iter().collect::<HashSet<_>>();
        let paired = descriptor_store
            .ordered_landmark_descriptors(map)
            .into_iter()
            .filter(|(landmark, _)| candidate_landmark_ids.contains(&landmark.id))
            .collect::<Vec<_>>();
        if paired.is_empty() {
            return Err(CorrespondenceBuildError::NoMapDescriptors {
                candidate_landmark_count,
            });
        }

        let radius_squared = search_radius_px * search_radius_px;
        let mut correspondences = Vec::new();
        let mut query_indices = Vec::new();
        let mut landmark_ids = Vec::new();
        let mut descriptor_matches = Vec::new();

        for (landmark_index, (landmark, descriptor)) in paired.iter().enumerate() {
            let point_camera = pose_prior.transform_world_point(&landmark.position);
            let Some(projected) = query.camera.project(&point_camera) else {
                continue;
            };
            if projected.x < 0.0
                || projected.y < 0.0
                || projected.x >= query.camera.width as f64
                || projected.y >= query.camera.height as f64
            {
                continue;
            }

            let mut window_indices = Vec::new();
            let mut window_descriptors = Vec::new();
            for (keypoint_index, keypoint) in query.keypoints.iter().enumerate() {
                let dx = keypoint.x - projected.x;
                let dy = keypoint.y - projected.y;
                if dx * dx + dy * dy <= radius_squared {
                    window_indices.push(keypoint_index);
                    window_descriptors.push(query.descriptors[keypoint_index].clone());
                }
            }
            if window_descriptors.is_empty() {
                continue;
            }

            let train = [descriptor.to_vec()];
            let window_matches = self.matcher.match_descriptors(&window_descriptors, &train);
            for window_match in window_matches {
                let Some(&global_query_index) = window_indices.get(window_match.query_index) else {
                    continue;
                };
                let Some(point2d) = query.keypoints.get(global_query_index).copied() else {
                    continue;
                };
                correspondences.push(Correspondence2D3D {
                    point2d,
                    point3d: landmark.position,
                    confidence: window_match.confidence,
                });
                query_indices.push(global_query_index);
                landmark_ids.push(landmark.id);
                descriptor_matches.push(DescriptorMatch {
                    query_index: global_query_index,
                    train_index: landmark_index,
                    ..window_match
                });
            }
        }

        if correspondences.is_empty() {
            return Err(CorrespondenceBuildError::NoDescriptorMatches {
                candidate_landmark_count,
            });
        }

        Ok(CorrespondenceSet {
            correspondences,
            query_indices,
            landmark_ids,
            candidate_landmark_count,
            match_count: descriptor_matches.len(),
            descriptor_matches,
        })
    }
}
