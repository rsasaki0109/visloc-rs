use std::collections::HashMap;
use std::convert::Infallible;

use nalgebra::{UnitQuaternion, Vector2, Vector3};

use crate::{Frame, FrameId, TwoViewMatchSet, VisualOdometryEstimate, VisualOdometryFrontend, SE3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoViewMatchVisualOdometryConfig {
    pub min_matches: usize,
    pub min_inliers: usize,
    pub max_residual_pixels: f64,
    pub pixel_translation_scale: f64,
    pub forward_translation: f64,
}

impl Default for TwoViewMatchVisualOdometryConfig {
    fn default() -> Self {
        Self {
            min_matches: 8,
            min_inliers: 6,
            max_residual_pixels: 3.0,
            pixel_translation_scale: 0.01,
            forward_translation: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TwoViewMatchVisualOdometryFrontend {
    config: TwoViewMatchVisualOdometryConfig,
    matches_by_frame_pair: HashMap<(FrameId, FrameId), TwoViewMatchSet>,
}

impl TwoViewMatchVisualOdometryFrontend {
    pub fn new(config: TwoViewMatchVisualOdometryConfig) -> Self {
        Self {
            config,
            matches_by_frame_pair: HashMap::new(),
        }
    }

    pub fn config(&self) -> TwoViewMatchVisualOdometryConfig {
        self.config
    }

    pub fn insert_matches(
        &mut self,
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
        matches: TwoViewMatchSet,
    ) {
        self.matches_by_frame_pair
            .insert((previous_frame_id, current_frame_id), matches);
    }

    pub fn with_matches(
        mut self,
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
        matches: TwoViewMatchSet,
    ) -> Self {
        self.insert_matches(previous_frame_id, current_frame_id, matches);
        self
    }

    pub fn matches_for(
        &self,
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
    ) -> Option<&TwoViewMatchSet> {
        self.matches_by_frame_pair
            .get(&(previous_frame_id, current_frame_id))
    }
}

impl VisualOdometryFrontend for TwoViewMatchVisualOdometryFrontend {
    type Error = Infallible;

    fn estimate_relative_pose(
        &self,
        previous_frame: &Frame,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error> {
        let Some(matches) = self.matches_for(previous_frame.id, current_frame.id) else {
            return Ok(None);
        };

        let Some((flow, inlier_count, mean_residual)) = estimate_translation_flow(
            matches,
            self.config.min_matches,
            self.config.min_inliers,
            self.config.max_residual_pixels,
        ) else {
            return Ok(None);
        };

        let previous_to_current = SE3::new(
            UnitQuaternion::identity(),
            Vector3::new(
                flow.x * self.config.pixel_translation_scale,
                flow.y * self.config.pixel_translation_scale,
                self.config.forward_translation,
            ),
        );
        let mut estimate =
            VisualOdometryEstimate::new(previous_frame.id, current_frame.id, previous_to_current);
        estimate.match_count = matches.len();
        estimate.inlier_count = inlier_count;
        estimate.mean_reprojection_error = Some(mean_residual);

        Ok(Some(estimate))
    }
}

fn estimate_translation_flow(
    matches: &TwoViewMatchSet,
    min_matches: usize,
    min_inliers: usize,
    max_residual_pixels: f64,
) -> Option<(Vector2<f64>, usize, f64)> {
    if matches.len() < min_matches {
        return None;
    }

    let flows = matches
        .matches()
        .iter()
        .map(|feature_match| feature_match.current_xy - feature_match.previous_xy)
        .collect::<Vec<_>>();
    let center = Vector2::new(
        median(flows.iter().map(|flow| flow.x).collect()),
        median(flows.iter().map(|flow| flow.y).collect()),
    );

    let inlier_flows = flows
        .iter()
        .copied()
        .filter(|flow| (*flow - center).norm() <= max_residual_pixels)
        .collect::<Vec<_>>();
    if inlier_flows.len() < min_inliers {
        return None;
    }

    let flow_sum = inlier_flows
        .iter()
        .fold(Vector2::zeros(), |sum, flow| sum + flow);
    let flow = flow_sum / inlier_flows.len() as f64;
    let mean_residual = inlier_flows
        .iter()
        .map(|inlier_flow| (*inlier_flow - flow).norm())
        .sum::<f64>()
        / inlier_flows.len() as f64;

    Some((flow, inlier_flows.len(), mean_residual))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}
