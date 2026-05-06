use std::collections::HashSet;

use nalgebra::Point3;
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
