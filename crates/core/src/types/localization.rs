use crate::geometry::Pose;

use super::{CameraId, LandmarkId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalizationFailureReason {
    QueryFeatureShapeMismatch {
        keypoint_count: usize,
        descriptor_count: usize,
    },
    NoCandidateLandmarks,
    NoMapDescriptors,
    NoDescriptorMatches,
    PoseEstimationFailed {
        correspondence_count: usize,
    },
    QualityGateFailed,
    MissingCamera {
        camera_id: CameraId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalizationResult {
    pub success: bool,
    pub pose: Option<Pose>,
    pub failure_reason: Option<LocalizationFailureReason>,
    pub candidate_landmark_count: usize,
    pub match_count: usize,
    pub correspondence_count: usize,
    pub inlier_count: usize,
    pub outlier_count: usize,
    pub inlier_ratio: f64,
    pub reprojection_error: Option<f64>,
    pub median_reprojection_error: Option<f64>,
    pub max_reprojection_error: Option<f64>,
    pub inlier_reprojection_errors: Vec<f64>,
    pub inliers: Vec<usize>,
    pub inlier_query_indices: Vec<usize>,
    pub inlier_landmark_ids: Vec<LandmarkId>,
    pub estimator_diagnostics: Option<PoseEstimatorDiagnostics>,
    pub pose_failure_diagnostics: Option<PoseEstimationFailureDiagnostics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalizationSuccess {
    pub pose: Pose,
    pub candidate_landmark_count: usize,
    pub match_count: usize,
    pub correspondence_count: usize,
    pub inliers: Vec<usize>,
    pub inlier_query_indices: Vec<usize>,
    pub inlier_landmark_ids: Vec<LandmarkId>,
    pub inlier_reprojection_errors: Vec<f64>,
    pub mean_reprojection_error: f64,
    pub median_reprojection_error: f64,
    pub max_reprojection_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseEstimatorDiagnostics {
    pub refinement_applied: bool,
    pub pre_refinement_mean_reprojection_error: Option<f64>,
    pub post_refinement_mean_reprojection_error: Option<f64>,
    pub refinement_error_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseEstimationFailureDiagnostics {
    pub reason: PoseEstimationFailureReason,
    pub correspondence_count: usize,
    pub minimum_correspondence_count: Option<usize>,
    pub ransac_iterations: Option<usize>,
    pub ransac_reprojection_threshold: Option<f64>,
    pub best_inlier_count: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoseEstimationFailureReason {
    InsufficientCorrespondences,
    NoValidPose,
    EstimatorRejected,
}

impl LocalizationResult {
    pub fn failure(
        failure_reason: LocalizationFailureReason,
        candidate_landmark_count: usize,
        match_count: usize,
        correspondence_count: usize,
    ) -> Self {
        Self {
            success: false,
            pose: None,
            failure_reason: Some(failure_reason),
            candidate_landmark_count,
            match_count,
            correspondence_count,
            inlier_count: 0,
            outlier_count: correspondence_count,
            inlier_ratio: 0.0,
            reprojection_error: None,
            median_reprojection_error: None,
            max_reprojection_error: None,
            inlier_reprojection_errors: Vec::new(),
            inliers: Vec::new(),
            inlier_query_indices: Vec::new(),
            inlier_landmark_ids: Vec::new(),
            estimator_diagnostics: None,
            pose_failure_diagnostics: None,
        }
    }

    pub fn success(success: LocalizationSuccess) -> Self {
        let inlier_count = success.inliers.len();
        let outlier_count = success.correspondence_count.saturating_sub(inlier_count);
        let inlier_ratio = if success.correspondence_count == 0 {
            0.0
        } else {
            inlier_count as f64 / success.correspondence_count as f64
        };

        Self {
            success: true,
            pose: Some(success.pose),
            failure_reason: None,
            candidate_landmark_count: success.candidate_landmark_count,
            match_count: success.match_count,
            correspondence_count: success.correspondence_count,
            inlier_count,
            outlier_count,
            inlier_ratio,
            reprojection_error: Some(success.mean_reprojection_error),
            median_reprojection_error: Some(success.median_reprojection_error),
            max_reprojection_error: Some(success.max_reprojection_error),
            inlier_reprojection_errors: success.inlier_reprojection_errors,
            inliers: success.inliers,
            inlier_query_indices: success.inlier_query_indices,
            inlier_landmark_ids: success.inlier_landmark_ids,
            estimator_diagnostics: None,
            pose_failure_diagnostics: None,
        }
    }

    pub fn with_estimator_diagnostics(mut self, diagnostics: PoseEstimatorDiagnostics) -> Self {
        self.estimator_diagnostics = Some(diagnostics);
        self
    }

    pub fn with_pose_failure_diagnostics(
        mut self,
        diagnostics: PoseEstimationFailureDiagnostics,
    ) -> Self {
        self.pose_failure_diagnostics = Some(diagnostics);
        self
    }

    pub fn rejected_by_quality_gate(mut self) -> Self {
        self.success = false;
        self.failure_reason = Some(LocalizationFailureReason::QualityGateFailed);
        self
    }
}
