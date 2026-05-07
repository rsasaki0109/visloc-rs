use std::collections::HashMap;
use std::convert::Infallible;

use nalgebra::{UnitQuaternion, Vector2, Vector3};
use visloc_vision::two_view::{
    EightPointEssentialMatrixEstimator, EssentialMatrixEstimator, EssentialRansac,
    RelativePoseEstimator, TwoViewCorrespondence,
};

use crate::{
    Camera, Frame, FrameId, TwoViewMatchSet, VisualOdometryEstimate, VisualOdometryFrontend, SE3,
};

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

/// Lightweight two-view VO frontend that turns externally supplied
/// correspondences into a translation-only relative-pose estimate.
///
/// Note: when this frontend produces a [`VisualOdometryEstimate`], the
/// estimate's `mean_reprojection_error` field stores the mean inlier flow
/// residual in pixels around the median two-view flow, not a 3D reprojection
/// error. The field name is reused so callers and downstream
/// `VisualOdometryPriorProvider` consumers do not need a separate diagnostic
/// type; refer to the value as `mean_flow_residual_px` in user-facing logs and
/// reports for the two-view-match case.
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

/// Configures the essential-matrix VO frontend. The Sampson threshold and
/// RANSAC iteration count tune the inlier search; `default_translation_scale`
/// is the scale applied when no per-pair scale is supplied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssentialMatrixVisualOdometryConfig {
    pub ransac_iterations: usize,
    pub sampson_threshold: f64,
    pub ransac_seed: u64,
    pub default_translation_scale: f64,
    pub min_inliers: usize,
}

impl Default for EssentialMatrixVisualOdometryConfig {
    fn default() -> Self {
        Self {
            ransac_iterations: 256,
            sampson_threshold: 5.0e-3,
            ransac_seed: 7,
            default_translation_scale: 1.0,
            min_inliers: 8,
        }
    }
}

/// Two-view VO frontend backed by the classical essential-matrix RANSAC
/// pipeline in `visloc-vision`. Compared to `TwoViewMatchVisualOdometryFrontend`
/// — which estimates a translation-only flow — this frontend recovers a full
/// SE3 relative pose (rotation + scaled translation) and reports inliers
/// against the Sampson distance.
///
/// The translation is fundamentally up-to-scale. Callers can either:
/// - Use `default_translation_scale` (a fixed metric scale, e.g., the typical
///   inter-frame displacement);
/// - Provide a per-pair scale through [`Self::insert_matches_with_scale`]
///   sourced from GNSS displacement, the previous frame's translation, or
///   another prior.
///
/// `VisualOdometryEstimate.mean_reprojection_error` stores the mean inlier
/// Sampson distance in normalized image-plane units (i.e., divided by focal
/// length). To convert to pixels, multiply by the camera's focal length.
#[derive(Debug, Clone, PartialEq)]
pub struct EssentialMatrixVisualOdometryFrontend {
    estimator: RelativePoseEstimator,
    camera: Camera,
    matches_by_frame_pair: HashMap<(FrameId, FrameId), TwoViewMatchSet>,
    scale_overrides: HashMap<(FrameId, FrameId), f64>,
    min_inliers: usize,
}

impl EssentialMatrixVisualOdometryFrontend {
    pub fn new(camera: Camera, config: EssentialMatrixVisualOdometryConfig) -> Self {
        let estimator = RelativePoseEstimator {
            ransac: EssentialRansac {
                estimator: EightPointEssentialMatrixEstimator::default(),
                config: visloc_vision::two_view::EssentialRansacConfig {
                    iterations: config.ransac_iterations,
                    sampson_threshold: config.sampson_threshold,
                    seed: config.ransac_seed,
                },
            },
            default_translation_scale: config.default_translation_scale,
        };
        Self {
            estimator,
            camera,
            matches_by_frame_pair: HashMap::new(),
            scale_overrides: HashMap::new(),
            min_inliers: config.min_inliers,
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn min_inliers(&self) -> usize {
        self.min_inliers
    }

    pub fn default_translation_scale(&self) -> f64 {
        self.estimator.default_translation_scale
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

    pub fn insert_matches_with_scale(
        &mut self,
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
        matches: TwoViewMatchSet,
        translation_scale: f64,
    ) {
        self.insert_matches(previous_frame_id, current_frame_id, matches);
        self.scale_overrides
            .insert((previous_frame_id, current_frame_id), translation_scale);
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

    pub fn with_matches_and_scale(
        mut self,
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
        matches: TwoViewMatchSet,
        translation_scale: f64,
    ) -> Self {
        self.insert_matches_with_scale(
            previous_frame_id,
            current_frame_id,
            matches,
            translation_scale,
        );
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

    fn translation_scale_for(&self, previous_frame_id: FrameId, current_frame_id: FrameId) -> f64 {
        self.scale_overrides
            .get(&(previous_frame_id, current_frame_id))
            .copied()
            .unwrap_or(self.estimator.default_translation_scale)
    }
}

impl VisualOdometryFrontend for EssentialMatrixVisualOdometryFrontend {
    type Error = Infallible;

    fn estimate_relative_pose(
        &self,
        previous_frame: &Frame,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error> {
        let Some(matches) = self.matches_for(previous_frame.id, current_frame.id) else {
            return Ok(None);
        };
        if matches.len()
            < self
                .estimator
                .ransac
                .estimator
                .minimum_correspondences()
                .max(self.min_inliers)
        {
            return Ok(None);
        }
        let correspondences: Vec<TwoViewCorrespondence> = matches
            .matches()
            .iter()
            .map(|feature_match| TwoViewCorrespondence {
                previous_xy: feature_match.previous_xy,
                current_xy: feature_match.current_xy,
            })
            .collect();
        let scale = self.translation_scale_for(previous_frame.id, current_frame.id);
        let Some(relative_pose) =
            self.estimator
                .estimate_with_scale(&correspondences, &self.camera, scale)
        else {
            return Ok(None);
        };
        if relative_pose.inliers.len() < self.min_inliers {
            return Ok(None);
        }
        let mut estimate = VisualOdometryEstimate::new(
            previous_frame.id,
            current_frame.id,
            relative_pose.previous_to_current,
        );
        estimate.match_count = matches.len();
        estimate.inlier_count = relative_pose.inliers.len();
        estimate.mean_reprojection_error = Some(relative_pose.mean_sampson_error);
        Ok(Some(estimate))
    }
}
