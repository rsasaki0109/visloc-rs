//! DROID-style lifecycle management for a sparse keyframe factor graph.
//!
//! This module deliberately separates factor proposal/lifecycle from the
//! optimizer.  The current proposer derives temporal, proximity, and stereo
//! support from a [`VisualMap`].  A future learned frontend can update the
//! target correction, anisotropic information, and damping on the same factor
//! records without changing the metric BA/PnP acceptance layer.

use std::collections::{BTreeMap, BTreeSet};

use visloc_core::types::{FrameId, LandmarkId, Observation, VisualMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SparseFactorKind {
    Temporal,
    Proximity,
    Stereo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SparseFactorKey {
    pub from_keyframe_id: FrameId,
    pub to_keyframe_id: FrameId,
    pub kind: SparseFactorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFactorState {
    Active,
    Inactive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFactorInactiveReason {
    LowConfidence,
    WindowAge,
    ActiveBudget,
}

/// Optimizer-facing measurement carried by every edge.  Geometric factors
/// start with zero correction and isotropic information; learned update
/// operators may replace these values in-place later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseFactorMeasurement {
    pub target_correction_px: [f32; 2],
    pub information: [f32; 2],
    pub damping: f32,
}

impl SparseFactorMeasurement {
    fn geometric(confidence: f32, damping: f32) -> Self {
        let confidence = confidence.clamp(0.0, 1.0);
        Self {
            target_correction_px: [0.0, 0.0],
            information: [confidence, confidence],
            damping,
        }
    }

    fn is_valid(self) -> bool {
        self.target_correction_px.iter().all(|v| v.is_finite())
            && self.information.iter().all(|v| v.is_finite() && *v >= 0.0)
            && self.damping.is_finite()
            && self.damping >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseKeyframeFactor {
    pub key: SparseFactorKey,
    pub state: SparseFactorState,
    pub inactive_reason: Option<SparseFactorInactiveReason>,
    pub support_count: usize,
    pub confidence: f32,
    pub measurement: SparseFactorMeasurement,
    pub created_generation: u64,
    pub last_scored_generation: u64,
    pub inactive_since_generation: Option<u64>,
    pub low_confidence_streak: usize,
    pub update_count: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseFactorGraphConfig {
    /// Add bidirectional temporal edges to this many previous keyframes.
    pub temporal_radius: usize,
    /// Number of newest keyframes whose incident edges remain frontend-active.
    pub active_window_keyframes: usize,
    /// Maximum camera-centre separation for a non-temporal proximity edge.
    pub proximity_radius_meters: f64,
    pub proximity_min_shared_landmarks: usize,
    pub max_proximity_neighbors_per_keyframe: usize,
    /// Shared/stereo support at which geometric confidence saturates.
    pub support_saturation_count: usize,
    pub minimum_active_confidence: f32,
    pub reactivation_confidence: f32,
    /// Consecutive low-confidence rescoring passes before an edge is retired.
    pub low_confidence_patience: usize,
    pub max_active_factors: usize,
    /// Inactive factors older than this many keyframe updates are forgotten.
    pub max_inactive_age_generations: u64,
    /// Temporal continuity edges are retired by window age, not confidence.
    pub protect_temporal_from_confidence_pruning: bool,
    pub geometric_damping: f32,
}

impl Default for SparseFactorGraphConfig {
    fn default() -> Self {
        Self {
            temporal_radius: 3,
            active_window_keyframes: 12,
            proximity_radius_meters: 2.5,
            proximity_min_shared_landmarks: 15,
            max_proximity_neighbors_per_keyframe: 8,
            support_saturation_count: 50,
            minimum_active_confidence: 0.15,
            reactivation_confidence: 0.25,
            low_confidence_patience: 2,
            max_active_factors: 256,
            max_inactive_age_generations: 80,
            protect_temporal_from_confidence_pruning: true,
            geometric_damping: 1.0e-4,
        }
    }
}

impl SparseFactorGraphConfig {
    pub fn is_valid(&self) -> bool {
        self.temporal_radius > 0
            && self.active_window_keyframes > 1
            && self.proximity_radius_meters.is_finite()
            && self.proximity_radius_meters > 0.0
            && self.proximity_min_shared_landmarks > 0
            && self.max_proximity_neighbors_per_keyframe > 0
            && self.support_saturation_count > 0
            && self.minimum_active_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_active_confidence)
            && self.reactivation_confidence.is_finite()
            && (self.minimum_active_confidence..=1.0).contains(&self.reactivation_confidence)
            && self.low_confidence_patience > 0
            && self.max_active_factors > 0
            && self.max_inactive_age_generations > 0
            && self.geometric_damping.is_finite()
            && self.geometric_damping >= 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SparseFactorGraphUpdateStats {
    pub keyframe_id: FrameId,
    pub generation: u64,
    pub added: usize,
    pub reactivated: usize,
    pub inactivated_low_confidence: usize,
    pub inactivated_window_age: usize,
    pub inactivated_budget: usize,
    pub pruned: usize,
    pub active_temporal: usize,
    pub active_proximity: usize,
    pub active_stereo: usize,
    pub inactive: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseFactorGraph {
    config: SparseFactorGraphConfig,
    generation: u64,
    keyframe_order: Vec<FrameId>,
    factors: BTreeMap<SparseFactorKey, SparseKeyframeFactor>,
}

impl SparseFactorGraph {
    pub fn new(config: SparseFactorGraphConfig) -> Self {
        assert!(
            config.is_valid(),
            "invalid sparse factor graph configuration"
        );
        Self {
            config,
            generation: 0,
            keyframe_order: Vec::new(),
            factors: BTreeMap::new(),
        }
    }

    pub fn config(&self) -> &SparseFactorGraphConfig {
        &self.config
    }

    pub fn factors(&self) -> impl Iterator<Item = &SparseKeyframeFactor> {
        self.factors.values()
    }

    pub fn active_factors(&self) -> impl Iterator<Item = &SparseKeyframeFactor> {
        self.factors
            .values()
            .filter(|factor| factor.state == SparseFactorState::Active)
    }

    pub fn inactive_factors(&self) -> impl Iterator<Item = &SparseKeyframeFactor> {
        self.factors
            .values()
            .filter(|factor| factor.state == SparseFactorState::Inactive)
    }

    pub fn factor(&self, key: SparseFactorKey) -> Option<&SparseKeyframeFactor> {
        self.factors.get(&key)
    }

    /// Replace an edge target/information estimate, e.g. from a learned
    /// recurrent update operator. Invalid numerical values are rejected.
    pub fn update_measurement(
        &mut self,
        key: SparseFactorKey,
        measurement: SparseFactorMeasurement,
    ) -> bool {
        if !measurement.is_valid() {
            return false;
        }
        let Some(factor) = self.factors.get_mut(&key) else {
            return false;
        };
        factor.measurement = measurement;
        factor.update_count = factor.update_count.saturating_add(1);
        true
    }

    /// Explicitly reactivate an inactive factor selected by a broader
    /// geometric/learned recovery pass. Ordinary rescoring does not revive
    /// budget- or age-retired edges, avoiding active-set churn.
    pub fn reactivate_factor(&mut self, key: SparseFactorKey) -> bool {
        let Some(factor) = self.factors.get_mut(&key) else {
            return false;
        };
        if factor.state == SparseFactorState::Active
            || factor.confidence < self.config.reactivation_confidence
        {
            return false;
        }
        factor.state = SparseFactorState::Active;
        factor.inactive_reason = None;
        factor.inactive_since_generation = None;
        factor.low_confidence_streak = 0;
        true
    }

    /// Update proposals and lifecycle after a keyframe has been committed to
    /// the map. Existing inactive factors are rescored so a broader recovery
    /// pass can reactivate geometry that becomes supported again.
    pub fn update_from_map(
        &mut self,
        map: &VisualMap,
        new_keyframe_id: FrameId,
    ) -> SparseFactorGraphUpdateStats {
        self.generation = self.generation.saturating_add(1);
        if !self.keyframe_order.contains(&new_keyframe_id) {
            self.keyframe_order.push(new_keyframe_id);
        }
        self.keyframe_order
            .retain(|id| map.keyframes.contains_key(id));

        let mut stats = SparseFactorGraphUpdateStats {
            keyframe_id: new_keyframe_id,
            generation: self.generation,
            ..SparseFactorGraphUpdateStats::default()
        };
        let proposals = self.propose_new_factors(map, new_keyframe_id);
        for (key, support_count, confidence) in proposals {
            match self.factors.get_mut(&key) {
                Some(factor) => {
                    factor.support_count = support_count;
                    factor.confidence = confidence;
                    factor.last_scored_generation = self.generation;
                    factor.low_confidence_streak = 0;
                    factor.update_count = factor.update_count.saturating_add(1);
                    if factor.state == SparseFactorState::Inactive
                        && confidence >= self.config.reactivation_confidence
                    {
                        factor.state = SparseFactorState::Active;
                        factor.inactive_reason = None;
                        factor.inactive_since_generation = None;
                        stats.reactivated += 1;
                    }
                }
                None => {
                    self.factors.insert(
                        key,
                        SparseKeyframeFactor {
                            key,
                            state: SparseFactorState::Active,
                            inactive_reason: None,
                            support_count,
                            confidence,
                            measurement: SparseFactorMeasurement::geometric(
                                confidence,
                                self.config.geometric_damping,
                            ),
                            created_generation: self.generation,
                            last_scored_generation: self.generation,
                            inactive_since_generation: None,
                            low_confidence_streak: 0,
                            update_count: 1,
                        },
                    );
                    stats.added += 1;
                }
            }
        }

        self.rescore_existing(map, &mut stats);
        self.retire_outside_active_window(&mut stats);
        self.enforce_active_budget(&mut stats);
        self.prune_stale_inactive(&mut stats);
        self.fill_counts(&mut stats);
        stats
    }

    /// Highest-confidence inactive edges incident to `query_keyframe_id`.
    /// These are the first edges a periodic broader/global recovery pass
    /// should ask its geometric or learned updater to reconsider.
    pub fn broader_recovery_candidates(
        &self,
        query_keyframe_id: FrameId,
        maximum: usize,
    ) -> Vec<&SparseKeyframeFactor> {
        let mut candidates = self
            .inactive_factors()
            .filter(|factor| {
                factor.key.from_keyframe_id == query_keyframe_id
                    || factor.key.to_keyframe_id == query_keyframe_id
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| right.support_count.cmp(&left.support_count))
                .then_with(|| left.key.cmp(&right.key))
        });
        candidates.truncate(maximum);
        candidates
    }

    /// Active non-self neighbors connected to one keyframe. Both directed
    /// orientations collapse to one deterministic id set.
    pub fn active_neighbor_keyframe_ids(&self, keyframe_id: FrameId) -> BTreeSet<FrameId> {
        let mut neighbors = BTreeSet::new();
        for factor in self.active_factors() {
            if factor.key.kind == SparseFactorKind::Stereo {
                continue;
            }
            if factor.key.from_keyframe_id == keyframe_id
                && factor.key.to_keyframe_id != keyframe_id
            {
                neighbors.insert(factor.key.to_keyframe_id);
            } else if factor.key.to_keyframe_id == keyframe_id
                && factor.key.from_keyframe_id != keyframe_id
            {
                neighbors.insert(factor.key.from_keyframe_id);
            }
        }
        neighbors
    }

    fn propose_new_factors(
        &self,
        map: &VisualMap,
        new_id: FrameId,
    ) -> Vec<(SparseFactorKey, usize, f32)> {
        let Some(new_keyframe) = map.keyframes.get(&new_id) else {
            return Vec::new();
        };
        let Some(new_position) = new_keyframe
            .frame
            .pose
            .as_ref()
            .map(|pose| pose.camera_center_world())
        else {
            return Vec::new();
        };
        let previous = self
            .keyframe_order
            .iter()
            .copied()
            .filter(|id| *id != new_id)
            .collect::<Vec<_>>();
        let temporal_ids = previous
            .iter()
            .rev()
            .take(self.config.temporal_radius)
            .copied()
            .collect::<BTreeSet<_>>();
        let mut proposals = Vec::new();
        for other_id in temporal_ids.iter().copied() {
            let (support, confidence) = pair_support(map, other_id, new_id, &self.config);
            for (from, to) in [(other_id, new_id), (new_id, other_id)] {
                proposals.push((
                    SparseFactorKey {
                        from_keyframe_id: from,
                        to_keyframe_id: to,
                        kind: SparseFactorKind::Temporal,
                    },
                    support,
                    confidence,
                ));
            }
        }

        let mut proximity = previous
            .iter()
            .copied()
            .filter(|id| !temporal_ids.contains(id))
            .filter_map(|other_id| {
                let other = map.keyframes.get(&other_id)?;
                let distance =
                    (other.frame.pose.as_ref()?.camera_center_world() - new_position).norm();
                if distance > self.config.proximity_radius_meters {
                    return None;
                }
                let (support, confidence) = pair_support(map, other_id, new_id, &self.config);
                (support >= self.config.proximity_min_shared_landmarks)
                    .then_some((other_id, support, confidence, distance))
            })
            .collect::<Vec<_>>();
        proximity.sort_by(|left, right| {
            right
                .2
                .total_cmp(&left.2)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| left.3.total_cmp(&right.3))
                .then_with(|| left.0.cmp(&right.0))
        });
        proximity.truncate(self.config.max_proximity_neighbors_per_keyframe);
        for (other_id, support, confidence, _) in proximity {
            for (from, to) in [(other_id, new_id), (new_id, other_id)] {
                proposals.push((
                    SparseFactorKey {
                        from_keyframe_id: from,
                        to_keyframe_id: to,
                        kind: SparseFactorKind::Proximity,
                    },
                    support,
                    confidence,
                ));
            }
        }

        let stereo = map
            .stereo_observations
            .iter()
            .filter(|observation| observation.frame_id == new_id)
            .count();
        if stereo > 0 {
            let confidence = support_confidence(stereo, 1.0, &self.config);
            proposals.push((
                SparseFactorKey {
                    from_keyframe_id: new_id,
                    to_keyframe_id: new_id,
                    kind: SparseFactorKind::Stereo,
                },
                stereo,
                confidence,
            ));
        }
        proposals
    }

    fn rescore_existing(&mut self, map: &VisualMap, stats: &mut SparseFactorGraphUpdateStats) {
        let active_window = self.active_keyframe_ids();
        for factor in self.factors.values_mut() {
            let (support, confidence) = match factor.key.kind {
                SparseFactorKind::Stereo => {
                    let count = map
                        .stereo_observations
                        .iter()
                        .filter(|observation| observation.frame_id == factor.key.from_keyframe_id)
                        .count();
                    (count, support_confidence(count, 1.0, &self.config))
                }
                SparseFactorKind::Temporal | SparseFactorKind::Proximity => pair_support(
                    map,
                    factor.key.from_keyframe_id,
                    factor.key.to_keyframe_id,
                    &self.config,
                ),
            };
            factor.support_count = support;
            factor.confidence = confidence;
            factor.last_scored_generation = self.generation;
            if confidence < self.config.minimum_active_confidence {
                factor.low_confidence_streak = factor.low_confidence_streak.saturating_add(1);
            } else {
                factor.low_confidence_streak = 0;
            }
            let confidence_prunable = factor.key.kind != SparseFactorKind::Temporal
                || !self.config.protect_temporal_from_confidence_pruning;
            if factor.state == SparseFactorState::Active
                && confidence_prunable
                && factor.low_confidence_streak >= self.config.low_confidence_patience
            {
                factor.state = SparseFactorState::Inactive;
                factor.inactive_reason = Some(SparseFactorInactiveReason::LowConfidence);
                factor.inactive_since_generation = Some(self.generation);
                stats.inactivated_low_confidence += 1;
            } else if factor.state == SparseFactorState::Inactive
                && factor.inactive_reason == Some(SparseFactorInactiveReason::LowConfidence)
                && confidence >= self.config.reactivation_confidence
                && (active_window.contains(&factor.key.from_keyframe_id)
                    || active_window.contains(&factor.key.to_keyframe_id))
            {
                factor.state = SparseFactorState::Active;
                factor.inactive_reason = None;
                factor.inactive_since_generation = None;
                factor.low_confidence_streak = 0;
                stats.reactivated += 1;
            }
        }
    }

    fn active_keyframe_ids(&self) -> BTreeSet<FrameId> {
        self.keyframe_order
            .iter()
            .rev()
            .take(self.config.active_window_keyframes)
            .copied()
            .collect()
    }

    fn retire_outside_active_window(&mut self, stats: &mut SparseFactorGraphUpdateStats) {
        let active = self.active_keyframe_ids();
        for factor in self.factors.values_mut() {
            if factor.state == SparseFactorState::Active
                && !active.contains(&factor.key.from_keyframe_id)
                && !active.contains(&factor.key.to_keyframe_id)
            {
                factor.state = SparseFactorState::Inactive;
                factor.inactive_reason = Some(SparseFactorInactiveReason::WindowAge);
                factor.inactive_since_generation = Some(self.generation);
                stats.inactivated_window_age += 1;
            }
        }
    }

    fn enforce_active_budget(&mut self, stats: &mut SparseFactorGraphUpdateStats) {
        let active_count = self.active_factors().count();
        if active_count <= self.config.max_active_factors {
            return;
        }
        let mut candidates = self
            .factors
            .values()
            .filter(|factor| factor.state == SparseFactorState::Active)
            .map(|factor| {
                let protected = matches!(
                    factor.key.kind,
                    SparseFactorKind::Temporal | SparseFactorKind::Stereo
                );
                (
                    factor.key,
                    protected,
                    factor.confidence,
                    factor.created_generation,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| left.2.total_cmp(&right.2))
                .then_with(|| left.3.cmp(&right.3))
                .then_with(|| left.0.cmp(&right.0))
        });
        let retire_count = active_count - self.config.max_active_factors;
        for (key, _, _, _) in candidates.into_iter().take(retire_count) {
            if let Some(factor) = self.factors.get_mut(&key) {
                factor.state = SparseFactorState::Inactive;
                factor.inactive_reason = Some(SparseFactorInactiveReason::ActiveBudget);
                factor.inactive_since_generation = Some(self.generation);
                stats.inactivated_budget += 1;
            }
        }
    }

    fn prune_stale_inactive(&mut self, stats: &mut SparseFactorGraphUpdateStats) {
        let before = self.factors.len();
        self.factors.retain(|_, factor| {
            factor.state == SparseFactorState::Active
                || factor.inactive_since_generation.is_some_and(|since| {
                    self.generation.saturating_sub(since)
                        <= self.config.max_inactive_age_generations
                })
        });
        stats.pruned = before - self.factors.len();
    }

    fn fill_counts(&self, stats: &mut SparseFactorGraphUpdateStats) {
        for factor in self.factors.values() {
            match (factor.state, factor.key.kind) {
                (SparseFactorState::Inactive, _) => stats.inactive += 1,
                (SparseFactorState::Active, SparseFactorKind::Temporal) => {
                    stats.active_temporal += 1
                }
                (SparseFactorState::Active, SparseFactorKind::Proximity) => {
                    stats.active_proximity += 1
                }
                (SparseFactorState::Active, SparseFactorKind::Stereo) => stats.active_stereo += 1,
            }
        }
    }
}

fn pair_support(
    map: &VisualMap,
    left_id: FrameId,
    right_id: FrameId,
    config: &SparseFactorGraphConfig,
) -> (usize, f32) {
    let (Some(left), Some(right)) = (map.keyframes.get(&left_id), map.keyframes.get(&right_id))
    else {
        return (0, 0.0);
    };
    let left_by_landmark = left
        .observations
        .iter()
        .map(|observation| (observation.landmark_id, observation))
        .collect::<BTreeMap<LandmarkId, &Observation>>();
    let mut confidence_sum = 0.0f32;
    let mut support = 0usize;
    for right_observation in &right.observations {
        let Some(left_observation) = left_by_landmark.get(&right_observation.landmark_id) else {
            continue;
        };
        let left_confidence = map.observation_confidence(left_observation).unwrap_or(1.0);
        let right_confidence = map.observation_confidence(right_observation).unwrap_or(1.0);
        confidence_sum += (left_confidence * right_confidence).max(0.0).sqrt();
        support += 1;
    }
    let mean_confidence = if support == 0 {
        0.0
    } else {
        confidence_sum / support as f32
    };
    (
        support,
        support_confidence(support, mean_confidence, config),
    )
}

fn support_confidence(
    support: usize,
    mean_observation_confidence: f32,
    config: &SparseFactorGraphConfig,
) -> f32 {
    let support_score = (support as f32 / config.support_saturation_count as f32).clamp(0.0, 1.0);
    (support_score * mean_observation_confidence).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::{
        geometry::{Pose, SE3},
        types::{Frame, Keyframe, Landmark, Observation, StereoObservation, VisualMap},
    };

    use super::*;

    fn add_keyframe(map: &mut VisualMap, id: u64, x: f64, landmarks: &[u64]) {
        let mut frame = Frame::new(id, 0);
        frame.pose = Some(Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(-x, 0.0, 0.0)),
        });
        let mut keyframe = Keyframe {
            frame,
            observations: Vec::new(),
        };
        for (keypoint_index, landmark_id) in landmarks.iter().copied().enumerate() {
            keyframe.observations.push(Observation {
                frame_id: id,
                landmark_id,
                keypoint_index,
                xy: Point2::new(keypoint_index as f64, 0.0),
            });
            let landmark = map
                .landmarks
                .entry(landmark_id)
                .or_insert_with(|| Landmark::new(landmark_id, Point3::new(0.0, 0.0, 4.0)));
            landmark
                .observations
                .push(keyframe.observations.last().unwrap().clone());
        }
        map.keyframes.insert(id, keyframe);
    }

    fn config() -> SparseFactorGraphConfig {
        SparseFactorGraphConfig {
            temporal_radius: 1,
            active_window_keyframes: 3,
            proximity_radius_meters: 5.0,
            proximity_min_shared_landmarks: 2,
            max_proximity_neighbors_per_keyframe: 3,
            support_saturation_count: 2,
            minimum_active_confidence: 0.25,
            reactivation_confidence: 0.5,
            low_confidence_patience: 2,
            max_active_factors: 32,
            max_inactive_age_generations: 10,
            protect_temporal_from_confidence_pruning: true,
            geometric_damping: 1.0e-4,
        }
    }

    #[test]
    fn proposes_bidirectional_temporal_proximity_and_stereo_factors() {
        let mut map = VisualMap::new();
        add_keyframe(&mut map, 0, 0.0, &[1, 2]);
        add_keyframe(&mut map, 1, 0.1, &[1, 2]);
        add_keyframe(&mut map, 2, 0.2, &[1, 2]);
        map.stereo_observations.push(StereoObservation {
            frame_id: 2,
            landmark_id: 1,
            right_camera_id: 1,
            xy_right: Point2::new(1.0, 1.0),
            left_to_right: SE3::identity(),
        });
        let mut graph = SparseFactorGraph::new(config());
        graph.update_from_map(&map, 0);
        graph.update_from_map(&map, 1);
        let stats = graph.update_from_map(&map, 2);
        assert_eq!(stats.active_temporal, 4);
        assert_eq!(stats.active_proximity, 2);
        assert_eq!(stats.active_stereo, 1);
        assert!(graph
            .factor(SparseFactorKey {
                from_keyframe_id: 0,
                to_keyframe_id: 2,
                kind: SparseFactorKind::Proximity,
            })
            .is_some());
    }

    #[test]
    fn retires_old_edges_but_keeps_them_for_broader_recovery() {
        let mut map = VisualMap::new();
        let mut graph = SparseFactorGraph::new(SparseFactorGraphConfig {
            active_window_keyframes: 2,
            ..config()
        });
        for id in 0..4 {
            add_keyframe(&mut map, id, id as f64 * 10.0, &[1, 2]);
            graph.update_from_map(&map, id);
        }
        let key = SparseFactorKey {
            from_keyframe_id: 0,
            to_keyframe_id: 1,
            kind: SparseFactorKind::Temporal,
        };
        assert_eq!(
            graph.factor(key).unwrap().state,
            SparseFactorState::Inactive
        );
        assert!(graph
            .broader_recovery_candidates(0, 4)
            .iter()
            .any(|f| f.key == key));
    }

    #[test]
    fn low_confidence_proximity_factor_is_inactivated_after_patience() {
        let mut map = VisualMap::new();
        add_keyframe(&mut map, 0, 0.0, &[1, 2]);
        add_keyframe(&mut map, 1, 0.1, &[1, 2]);
        add_keyframe(&mut map, 2, 0.2, &[1, 2]);
        let mut graph = SparseFactorGraph::new(config());
        graph.update_from_map(&map, 0);
        graph.update_from_map(&map, 1);
        graph.update_from_map(&map, 2);
        let key = SparseFactorKey {
            from_keyframe_id: 0,
            to_keyframe_id: 2,
            kind: SparseFactorKind::Proximity,
        };
        map.keyframes.get_mut(&0).unwrap().observations.clear();
        graph.update_from_map(&map, 2);
        graph.update_from_map(&map, 2);
        assert_eq!(
            graph.factor(key).unwrap().state,
            SparseFactorState::Inactive
        );
    }

    #[test]
    fn measurement_update_is_numeric_and_optimizer_facing() {
        let mut map = VisualMap::new();
        add_keyframe(&mut map, 0, 0.0, &[1, 2]);
        add_keyframe(&mut map, 1, 0.1, &[1, 2]);
        let mut graph = SparseFactorGraph::new(config());
        graph.update_from_map(&map, 0);
        graph.update_from_map(&map, 1);
        let key = SparseFactorKey {
            from_keyframe_id: 0,
            to_keyframe_id: 1,
            kind: SparseFactorKind::Temporal,
        };
        assert!(graph.update_measurement(
            key,
            SparseFactorMeasurement {
                target_correction_px: [0.4, -0.2],
                information: [0.8, 0.3],
                damping: 0.01,
            }
        ));
        assert_eq!(
            graph.factor(key).unwrap().measurement.information,
            [0.8, 0.3]
        );
        assert!(!graph.update_measurement(
            key,
            SparseFactorMeasurement {
                target_correction_px: [f32::NAN, 0.0],
                information: [1.0, 1.0],
                damping: 0.0,
            }
        ));
    }

    #[test]
    fn confidence_uses_shared_observation_scores() {
        let mut map = VisualMap::new();
        add_keyframe(&mut map, 0, 0.0, &[1, 2]);
        add_keyframe(&mut map, 1, 0.1, &[1, 2]);
        let observations = map.keyframes[&1].observations.clone();
        for observation in observations {
            assert!(map.set_observation_confidence(&observation, 0.25));
        }
        let mut graph = SparseFactorGraph::new(config());
        graph.update_from_map(&map, 0);
        graph.update_from_map(&map, 1);
        let factor = graph
            .factor(SparseFactorKey {
                from_keyframe_id: 0,
                to_keyframe_id: 1,
                kind: SparseFactorKind::Temporal,
            })
            .unwrap();
        assert!((factor.confidence - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn budget_retired_factor_does_not_thrash_back_on_ordinary_rescore() {
        let mut map = VisualMap::new();
        add_keyframe(&mut map, 0, 0.0, &[1, 2]);
        add_keyframe(&mut map, 1, 0.1, &[1, 2]);
        add_keyframe(&mut map, 2, 0.2, &[1, 2]);
        let mut graph = SparseFactorGraph::new(SparseFactorGraphConfig {
            max_active_factors: 1,
            ..config()
        });
        graph.update_from_map(&map, 0);
        graph.update_from_map(&map, 1);
        let retired_key = graph
            .inactive_factors()
            .find(|factor| factor.inactive_reason == Some(SparseFactorInactiveReason::ActiveBudget))
            .unwrap()
            .key;

        let stats = graph.update_from_map(&map, 2);

        assert_eq!(stats.reactivated, 0);
        let retired = graph.factor(retired_key).unwrap();
        assert_eq!(retired.state, SparseFactorState::Inactive);
        assert_eq!(
            retired.inactive_reason,
            Some(SparseFactorInactiveReason::ActiveBudget)
        );
    }
}
