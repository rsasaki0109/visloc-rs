//! Map-based localization and PnP + RANSAC pose estimation.

use nalgebra::{Point2, Point3};
use numpy::{AllowTypeChange, PyArray1, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use visloc_core::types::{LocalizationFailureReason, LocalizationResult, QueryImage};
use visloc_localization::{AllLandmarksSelector, LocalizationConfig, LocalizationPipeline};
use visloc_vision::matching::BruteForceMatcher;
use visloc_vision::pnp::{Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};

use crate::arrays::{require_finite, rows, vec_to_py};
use crate::camera::PyCamera;
use crate::colmap::PyReconstruction;
use crate::pose::PyPose;

fn failure_reason_name(reason: &LocalizationFailureReason) -> String {
    match reason {
        LocalizationFailureReason::QueryFeatureShapeMismatch { .. } => {
            "query_feature_shape_mismatch"
        }
        LocalizationFailureReason::NoCandidateLandmarks => "no_candidate_landmarks",
        LocalizationFailureReason::NoMapDescriptors => "no_map_descriptors",
        LocalizationFailureReason::NoDescriptorMatches => "no_descriptor_matches",
        LocalizationFailureReason::InvalidProjectionQueryLandmarkRatio => {
            "invalid_projection_query_landmark_ratio"
        }
        LocalizationFailureReason::PoseEstimationFailed { .. } => "pose_estimation_failed",
        LocalizationFailureReason::QualityGateFailed => "quality_gate_failed",
        LocalizationFailureReason::MissingCamera { .. } => "missing_camera",
    }
    .to_owned()
}

fn ransac(
    iterations: usize,
    reprojection_threshold: f64,
    seed: u64,
    refine: bool,
    confidence: Option<f64>,
) -> PyResult<PnPRansac<DltPnP, GaussNewtonPoseRefiner>> {
    if iterations == 0 {
        return Err(PyValueError::new_err("ransac_iterations must be positive"));
    }
    if !(reprojection_threshold.is_finite() && reprojection_threshold > 0.0) {
        return Err(PyValueError::new_err(
            "reprojection_threshold must be a positive finite number of pixels",
        ));
    }
    if let Some(confidence) = confidence {
        if !(confidence > 0.0 && confidence < 1.0) {
            return Err(PyValueError::new_err("confidence must be in (0, 1)"));
        }
    }
    let mut estimator = PnPRansac {
        iterations,
        reprojection_threshold,
        seed,
        confidence,
        ..PnPRansac::default()
    };
    if !refine {
        estimator.pose_refiner = None;
    }
    Ok(estimator)
}

/// Outcome of `localize`: success flag, pose, counts, and inlier details.
#[pyclass(
    module = "visloc",
    name = "LocalizationResult",
    frozen,
    skip_from_py_object
)]
pub struct PyLocalizationResult {
    inner: LocalizationResult,
}

#[pymethods]
impl PyLocalizationResult {
    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    /// Estimated world-to-camera pose, or `None` on failure.
    #[getter]
    fn pose(&self) -> Option<PyPose> {
        self.inner.pose.as_ref().map(PyPose::from_pose)
    }

    /// Machine-readable failure reason (e.g. `"no_descriptor_matches"`), or
    /// `None` on success.
    #[getter]
    fn failure_reason(&self) -> Option<String> {
        self.inner.failure_reason.as_ref().map(failure_reason_name)
    }

    #[getter]
    fn candidate_landmark_count(&self) -> usize {
        self.inner.candidate_landmark_count
    }

    #[getter]
    fn match_count(&self) -> usize {
        self.inner.match_count
    }

    #[getter]
    fn correspondence_count(&self) -> usize {
        self.inner.correspondence_count
    }

    #[getter]
    fn inlier_count(&self) -> usize {
        self.inner.inlier_count
    }

    #[getter]
    fn outlier_count(&self) -> usize {
        self.inner.outlier_count
    }

    #[getter]
    fn inlier_ratio(&self) -> f64 {
        self.inner.inlier_ratio
    }

    /// Mean inlier reprojection error in pixels.
    #[getter]
    fn mean_reprojection_error(&self) -> Option<f64> {
        self.inner.reprojection_error
    }

    #[getter]
    fn median_reprojection_error(&self) -> Option<f64> {
        self.inner.median_reprojection_error
    }

    #[getter]
    fn max_reprojection_error(&self) -> Option<f64> {
        self.inner.max_reprojection_error
    }

    /// Indices into the query keypoints of the inlier correspondences.
    #[getter]
    fn inlier_query_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        vec_to_py(
            py,
            self.inner
                .inlier_query_indices
                .iter()
                .map(|&i| i as i64)
                .collect(),
        )
    }

    /// 3-D point ids of the inlier correspondences.
    #[getter]
    fn inlier_point3d_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        vec_to_py(py, self.inner.inlier_landmark_ids.clone())
    }

    #[getter]
    fn inlier_reprojection_errors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.inner.inlier_reprojection_errors.clone())
    }

    fn __bool__(&self) -> bool {
        self.inner.success
    }

    fn __repr__(&self) -> String {
        if self.inner.success {
            format!(
                "LocalizationResult(success=True, inliers={}/{}, mean_reprojection_error={})",
                self.inner.inlier_count,
                self.inner.correspondence_count,
                self.inner
                    .reprojection_error
                    .map_or_else(|| "None".to_owned(), |e| format!("{e:?}"))
            )
        } else {
            format!(
                "LocalizationResult(success=False, failure_reason={:?}, matches={})",
                self.inner
                    .failure_reason
                    .as_ref()
                    .map_or_else(|| "unknown".to_owned(), failure_reason_name),
                self.inner.match_count
            )
        }
    }
}

/// Localize a query image against a reconstruction whose 3-D points carry
/// descriptors.
///
/// The query is a camera plus `keypoints` `(N, 2)` and `descriptors` `(N, D)`.
/// Descriptors are matched to the map with a brute-force L2 matcher and Lowe's
/// ratio test, then the pose is estimated with DLT PnP inside RANSAC and
/// refined with Gauss-Newton. The GIL is released while localizing.
#[pyfunction]
#[pyo3(signature = (
    camera,
    keypoints,
    descriptors,
    reconstruction,
    *,
    ratio = Some(0.8),
    ransac_iterations = 128,
    reprojection_threshold = 4.0,
    min_inliers = 0,
    min_inlier_ratio = 0.0,
    max_mean_reprojection_error = None,
    max_median_reprojection_error = None,
    max_reprojection_error = None,
    seed = 7,
))]
#[allow(clippy::too_many_arguments)]
pub fn localize(
    py: Python<'_>,
    camera: PyRef<'_, PyCamera>,
    keypoints: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    descriptors: PyArrayLikeDyn<'_, f32, AllowTypeChange>,
    reconstruction: PyRef<'_, PyReconstruction>,
    ratio: Option<f32>,
    ransac_iterations: usize,
    reprojection_threshold: f64,
    min_inliers: usize,
    min_inlier_ratio: f64,
    max_mean_reprojection_error: Option<f64>,
    max_median_reprojection_error: Option<f64>,
    max_reprojection_error: Option<f64>,
    seed: u64,
) -> PyResult<PyLocalizationResult> {
    let keypoint_rows = rows(keypoints.as_array(), 2, "keypoints")?;
    if keypoint_rows.single {
        return Err(PyValueError::new_err("keypoints must have shape (N, 2)"));
    }
    require_finite(&keypoint_rows.data, "keypoints")?;
    let descriptor_view = descriptors.as_array();
    if descriptor_view.ndim() != 2 || descriptor_view.shape()[0] != keypoint_rows.count {
        return Err(PyValueError::new_err(format!(
            "descriptors must have shape ({}, D), got {:?}",
            keypoint_rows.count,
            descriptor_view.shape()
        )));
    }
    if let Some(ratio) = ratio {
        if !(ratio > 0.0 && ratio <= 1.0) {
            return Err(PyValueError::new_err("ratio must be in (0, 1], or None"));
        }
    }
    let query = QueryImage {
        camera: camera.inner.clone(),
        keypoints: keypoint_rows
            .iter()
            .map(|r| Point2::new(r[0], r[1]))
            .collect(),
        descriptors: descriptor_view
            .outer_iter()
            .map(|row| row.iter().copied().collect())
            .collect(),
    };
    let config = LocalizationConfig {
        ratio,
        ransac_iterations,
        reprojection_threshold,
        min_inliers,
        min_inlier_ratio,
        max_mean_reprojection_error,
        max_median_reprojection_error,
        max_reprojection_error,
    };
    let pipeline = LocalizationPipeline::with_pose_estimator(
        BruteForceMatcher { ratio },
        AllLandmarksSelector,
        ransac(ransac_iterations, reprojection_threshold, seed, true, None)?,
        config,
    );
    let map = &reconstruction.map;
    let inner = py.detach(|| pipeline.localize(&query, map));
    Ok(PyLocalizationResult { inner })
}

/// Result of `estimate_pose_pnp_ransac`.
#[pyclass(
    module = "visloc",
    name = "PnPRansacResult",
    frozen,
    skip_from_py_object
)]
pub struct PyPnPRansacResult {
    pose: PyPose,
    inliers: Vec<usize>,
    inlier_reprojection_errors: Vec<f64>,
    mean_reprojection_error: f64,
    median_reprojection_error: f64,
    max_reprojection_error: f64,
    refinement_applied: bool,
}

#[pymethods]
impl PyPnPRansacResult {
    /// Estimated world-to-camera pose.
    #[getter]
    fn pose(&self) -> PyPose {
        self.pose.clone()
    }

    /// Indices of the inlier correspondences.
    #[getter]
    fn inliers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        vec_to_py(py, self.inliers.iter().map(|&i| i as i64).collect())
    }

    #[getter]
    fn inlier_count(&self) -> usize {
        self.inliers.len()
    }

    #[getter]
    fn inlier_reprojection_errors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.inlier_reprojection_errors.clone())
    }

    #[getter]
    fn mean_reprojection_error(&self) -> f64 {
        self.mean_reprojection_error
    }

    #[getter]
    fn median_reprojection_error(&self) -> f64 {
        self.median_reprojection_error
    }

    #[getter]
    fn max_reprojection_error(&self) -> f64 {
        self.max_reprojection_error
    }

    #[getter]
    fn refinement_applied(&self) -> bool {
        self.refinement_applied
    }

    fn __repr__(&self) -> String {
        format!(
            "PnPRansacResult(inliers={}, mean_reprojection_error={:?})",
            self.inliers.len(),
            self.mean_reprojection_error
        )
    }
}

/// Estimate a world-to-camera pose from 2D-3D correspondences.
///
/// `points2d` `(N, 2)` are pixels in `camera`; `points3d` `(N, 3)` are world
/// points. Runs DLT PnP in RANSAC (deterministic for a given `seed`) and, when
/// `refine` is true, Gauss-Newton refinement on the inliers. `confidence`
/// enables COLMAP-style adaptive termination with `ransac_iterations` as the
/// cap. Returns `None` when no pose is found.
#[pyfunction]
#[pyo3(signature = (
    camera,
    points2d,
    points3d,
    *,
    ransac_iterations = 128,
    reprojection_threshold = 4.0,
    seed = 7,
    refine = true,
    confidence = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn estimate_pose_pnp_ransac(
    py: Python<'_>,
    camera: PyRef<'_, PyCamera>,
    points2d: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    points3d: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    ransac_iterations: usize,
    reprojection_threshold: f64,
    seed: u64,
    refine: bool,
    confidence: Option<f64>,
) -> PyResult<Option<PyPnPRansacResult>> {
    let p2 = rows(points2d.as_array(), 2, "points2d")?;
    let p3 = rows(points3d.as_array(), 3, "points3d")?;
    if p2.single || p3.single || p2.count != p3.count {
        return Err(PyValueError::new_err(format!(
            "points2d (N, 2) and points3d (N, 3) must have the same N, got {} and {}",
            p2.count, p3.count
        )));
    }
    require_finite(&p2.data, "points2d")?;
    require_finite(&p3.data, "points3d")?;
    let correspondences = (0..p2.count)
        .map(|i| {
            let a = p2.row(i);
            let b = p3.row(i);
            Correspondence2D3D {
                point2d: Point2::new(a[0], a[1]),
                point3d: Point3::new(b[0], b[1], b[2]),
                confidence: None,
            }
        })
        .collect::<Vec<_>>();
    let estimator = ransac(
        ransac_iterations,
        reprojection_threshold,
        seed,
        refine,
        confidence,
    )?;
    let camera = camera.inner.clone();
    let report = py.detach(|| estimator.estimate(&correspondences, &camera));
    Ok(report.map(|report| PyPnPRansacResult {
        pose: PyPose::from_pose(&report.pose),
        inliers: report.inliers,
        inlier_reprojection_errors: report.inlier_reprojection_errors,
        mean_reprojection_error: report.mean_reprojection_error,
        median_reprojection_error: report.median_reprojection_error,
        max_reprojection_error: report.max_reprojection_error,
        refinement_applied: report.diagnostics.refinement_applied,
    }))
}
