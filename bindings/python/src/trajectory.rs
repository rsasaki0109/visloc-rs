//! Trajectory evaluation: Umeyama alignment, ATE, and RPE.

use std::collections::{HashMap, HashSet};

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayDyn, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use visloc_core::geometry::{Pose, SE3};
use visloc_tracking::{
    umeyama_similarity_transform, PoseTrajectory, RelativePoseErrorConfig,
    RelativePoseErrorStatistics, TrackingEvent, TrackingState, TrajectoryAlignment,
    TrajectorySample, TrajectorySimilarityTransform,
};

use crate::arrays::{matrix3_to_py, matrix4_to_py, require_finite, rows, rows_to_py, vec_to_py};
use crate::pose::{rotation_from_matrix, unit_quaternion};

/// A similarity transform `x -> s R x + t` (scale is 1 for rigid alignment).
#[pyclass(
    module = "visloc",
    name = "SimilarityTransform",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySimilarityTransform {
    inner: TrajectorySimilarityTransform,
}

#[pymethods]
impl PySimilarityTransform {
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix3_to_py(py, self.inner.rotation.matrix())
    }

    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.inner.translation.as_slice().to_vec())
    }

    /// 4x4 homogeneous matrix `[[s R, t], [0, 1]]`.
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let mut m = nalgebra::Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(self.inner.rotation.matrix() * self.inner.scale));
        m.fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.inner.translation);
        matrix4_to_py(py, &m)
    }

    /// Apply to points of shape `(3,)` or `(N, 3)`.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        points: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let input = rows(points.as_array(), 3, "points")?;
        let mut out = Vec::with_capacity(input.count * 3);
        for row in input.iter() {
            let p = self.inner.apply(&Point3::new(row[0], row[1], row[2]));
            out.extend_from_slice(&[p.x, p.y, p.z]);
        }
        rows_to_py(py, out, 3, input.single)
    }

    fn __repr__(&self) -> String {
        let t = self.inner.translation;
        format!(
            "SimilarityTransform(scale={:?}, translation=[{:?}, {:?}, {:?}])",
            self.inner.scale, t.x, t.y, t.z
        )
    }
}

/// Closed-form Umeyama (1991) alignment minimizing `sum ||T(source_i) - target_i||^2`.
///
/// `source` and `target` are `(N, 3)` with `N >= 2`. With `with_scale=False`
/// the result is a rigid SE(3) transform; with `True` a Sim(3) transform.
#[pyfunction]
#[pyo3(signature = (source, target, with_scale = false))]
pub fn umeyama_alignment(
    source: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    target: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    with_scale: bool,
) -> PyResult<PySimilarityTransform> {
    let source = points_from_rows(rows(source.as_array(), 3, "source")?)?;
    let target = points_from_rows(rows(target.as_array(), 3, "target")?)?;
    if source.len() != target.len() {
        return Err(PyValueError::new_err(format!(
            "source and target must have the same number of points, got {} and {}",
            source.len(),
            target.len()
        )));
    }
    umeyama_similarity_transform(&source, &target, with_scale)
        .map(|inner| PySimilarityTransform { inner })
        .ok_or_else(|| {
            PyValueError::new_err(
                "Umeyama alignment is degenerate (need >= 2 points with non-zero spread)",
            )
        })
}

fn points_from_rows(input: crate::arrays::Rows) -> PyResult<Vec<Point3<f64>>> {
    require_finite(&input.data, "points")?;
    Ok(input
        .iter()
        .map(|r| Point3::new(r[0], r[1], r[2]))
        .collect())
}

/// Parse an `(N, 3)` positions, `(N, 7)` TUM `[tx ty tz qx qy qz qw]`, or
/// `(N, 4, 4)` / `(N, 3, 4)` camera-to-world array into camera-to-world poses.
fn camera_to_world_poses(
    view: numpy::ndarray::ArrayViewD<'_, f64>,
    name: &str,
) -> PyResult<Vec<SE3>> {
    let shape = view.shape().to_vec();
    let values = view.iter().copied().collect::<Vec<_>>();
    require_finite(&values, name)?;
    match shape.as_slice() {
        [n, 3] => Ok((0..*n)
            .map(|i| {
                let r = &values[i * 3..i * 3 + 3];
                SE3::new(UnitQuaternion::identity(), Vector3::new(r[0], r[1], r[2]))
            })
            .collect()),
        [n, 7] => (0..*n)
            .map(|i| {
                let r = &values[i * 7..i * 7 + 7];
                Ok(SE3::new(
                    unit_quaternion(r[6], r[3], r[4], r[5])?,
                    Vector3::new(r[0], r[1], r[2]),
                ))
            })
            .collect(),
        [n, rows_, 4] if *rows_ == 3 || *rows_ == 4 => (0..*n)
            .map(|i| {
                let m = &values[i * rows_ * 4..(i + 1) * rows_ * 4];
                let rotation = Matrix3::from_fn(|r, c| m[r * 4 + c]);
                Ok(SE3::new(
                    rotation_from_matrix(&rotation)?,
                    Vector3::new(m[3], m[7], m[11]),
                ))
            })
            .collect(),
        _ => Err(PyValueError::new_err(format!(
            "{name} must have shape (N, 3) positions, (N, 7) [tx, ty, tz, qx, qy, qz, qw], \
             or (N, 4, 4) / (N, 3, 4) camera-to-world matrices; got {shape:?}"
        ))),
    }
}

fn frame_ids(
    ids: Option<PyArrayLikeDyn<'_, u64, AllowTypeChange>>,
    count: usize,
    name: &str,
) -> PyResult<Vec<u64>> {
    let Some(ids) = ids else {
        return Ok((0..count as u64).collect());
    };
    let view = ids.as_array();
    if view.ndim() != 1 || view.len() != count {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({count},), got {:?}",
            view.shape()
        )));
    }
    let ids = view.iter().copied().collect::<Vec<_>>();
    let mut seen = HashSet::with_capacity(ids.len());
    if let Some(duplicate) = ids.iter().find(|id| !seen.insert(**id)) {
        return Err(PyValueError::new_err(format!(
            "{name} contains duplicate frame id {duplicate}"
        )));
    }
    Ok(ids)
}

fn trajectory(
    poses: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    ids: Option<PyArrayLikeDyn<'_, u64, AllowTypeChange>>,
    name: &str,
) -> PyResult<PoseTrajectory> {
    let camera_to_world = camera_to_world_poses(poses.as_array(), name)?;
    let ids = frame_ids(ids, camera_to_world.len(), &format!("{name}_ids"))?;
    let mut trajectory = PoseTrajectory::new();
    for (frame_id, pose) in ids.into_iter().zip(camera_to_world) {
        trajectory.push_sample(TrajectorySample {
            frame_id,
            pose: Pose {
                world_to_camera: pose.inverse(),
            },
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        });
    }
    Ok(trajectory)
}

fn parse_alignment(alignment: &str) -> PyResult<TrajectoryAlignment> {
    match alignment.to_ascii_lowercase().as_str() {
        "none" => Ok(TrajectoryAlignment::None),
        "first" | "first_translation" => Ok(TrajectoryAlignment::FirstMatchedTranslation),
        "se3" | "rigid" => Ok(TrajectoryAlignment::Umeyama),
        "sim3" | "similarity" => Ok(TrajectoryAlignment::UmeyamaWithScale),
        other => Err(PyValueError::new_err(format!(
            "unknown alignment {other:?}; expected 'none', 'first', 'se3', or 'sim3'"
        ))),
    }
}

/// The transform `PoseTrajectory` applies for `alignment` (mirrors the
/// tracking crate: estimated centers in estimated order, identity fallback).
fn alignment_transform(
    estimated: &PoseTrajectory,
    reference: &PoseTrajectory,
    alignment: TrajectoryAlignment,
) -> TrajectorySimilarityTransform {
    let reference_by_id = reference
        .samples()
        .iter()
        .map(|s| (s.frame_id, s.camera_center_world()))
        .collect::<HashMap<_, _>>();
    let matched = estimated
        .samples()
        .iter()
        .filter_map(|s| {
            reference_by_id
                .get(&s.frame_id)
                .map(|r| (s.camera_center_world(), *r))
        })
        .collect::<Vec<_>>();
    match alignment {
        TrajectoryAlignment::None => TrajectorySimilarityTransform::identity(),
        TrajectoryAlignment::FirstMatchedTranslation => matched
            .first()
            .map(|(e, r)| TrajectorySimilarityTransform::pure_translation(r.coords - e.coords))
            .unwrap_or_else(TrajectorySimilarityTransform::identity),
        TrajectoryAlignment::Umeyama | TrajectoryAlignment::UmeyamaWithScale => {
            let (source, target): (Vec<_>, Vec<_>) = matched.into_iter().unzip();
            umeyama_similarity_transform(
                &source,
                &target,
                alignment == TrajectoryAlignment::UmeyamaWithScale,
            )
            .unwrap_or_else(TrajectorySimilarityTransform::identity)
        }
    }
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        0.5 * (sorted[middle - 1] + sorted[middle])
    }
}

/// Absolute trajectory error (translation of camera centers, in input units).
#[pyclass(module = "visloc", name = "AteResult", frozen, skip_from_py_object)]
pub struct PyAteResult {
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    mean: f64,
    #[pyo3(get)]
    median: f64,
    #[pyo3(get)]
    std: f64,
    #[pyo3(get)]
    min: f64,
    #[pyo3(get)]
    max: f64,
    #[pyo3(get)]
    matched_count: usize,
    #[pyo3(get)]
    estimated_count: usize,
    #[pyo3(get)]
    reference_count: usize,
    #[pyo3(get)]
    missing_reference_count: usize,
    #[pyo3(get)]
    missing_estimate_count: usize,
    #[pyo3(get)]
    alignment: PySimilarityTransform,
    errors: Vec<f64>,
    frame_ids: Vec<u64>,
}

#[pymethods]
impl PyAteResult {
    /// Per-frame translation errors after alignment, in estimated order.
    #[getter]
    fn errors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.errors.clone())
    }

    /// Frame ids aligned with `errors`.
    #[getter]
    fn frame_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        vec_to_py(py, self.frame_ids.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "AteResult(rmse={:?}, mean={:?}, max={:?}, matched_count={})",
            self.rmse, self.mean, self.max, self.matched_count
        )
    }
}

/// Absolute trajectory error between two trajectories.
///
/// Each trajectory is `(N, 3)` camera centers, `(N, 7)` TUM-style
/// `[tx, ty, tz, qx, qy, qz, qw]`, or `(N, 4, 4)` / `(N, 3, 4)` camera-to-world
/// matrices. Poses are matched by frame id (`*_ids`, default `0..N`).
/// `alignment` is `"se3"` (rigid Umeyama, default), `"sim3"` (Umeyama with
/// scale, for monocular), `"first"` (first matched translation), or `"none"`.
#[pyfunction]
#[pyo3(signature = (estimated, reference, *, alignment = "se3", estimated_ids = None, reference_ids = None))]
pub fn evaluate_ate(
    estimated: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    reference: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    alignment: &str,
    estimated_ids: Option<PyArrayLikeDyn<'_, u64, AllowTypeChange>>,
    reference_ids: Option<PyArrayLikeDyn<'_, u64, AllowTypeChange>>,
) -> PyResult<PyAteResult> {
    let alignment = parse_alignment(alignment)?;
    let estimated = trajectory(estimated, estimated_ids, "estimated")?;
    let reference = trajectory(reference, reference_ids, "reference")?;
    let errors = estimated.translation_errors_against_with_alignment(&reference, alignment);
    if errors.is_empty() {
        return Err(PyValueError::new_err(
            "no estimated pose shares a frame id with the reference",
        ));
    }
    let summary = estimated.translation_error_summary_against_with_alignment(&reference, alignment);
    let values = errors
        .iter()
        .map(|e| e.translation_error)
        .collect::<Vec<_>>();
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = summary.mean_translation_error.unwrap_or(f64::NAN);
    let rmse = summary.rmse_translation_error.unwrap_or(f64::NAN);
    Ok(PyAteResult {
        rmse,
        mean,
        median: median(&sorted),
        std: (rmse * rmse - mean * mean).max(0.0).sqrt(),
        min: sorted[0],
        max: summary.max_translation_error.unwrap_or(f64::NAN),
        matched_count: summary.matched_pose_count,
        estimated_count: summary.estimated_pose_count,
        reference_count: summary.reference_pose_count,
        missing_reference_count: summary.missing_reference_count,
        missing_estimate_count: summary.missing_estimate_count,
        alignment: PySimilarityTransform {
            inner: alignment_transform(&estimated, &reference, alignment),
        },
        frame_ids: errors.iter().map(|e| e.frame_id).collect(),
        errors: values,
    })
}

fn stats_dict(stats: Option<RelativePoseErrorStatistics>) -> HashMap<&'static str, f64> {
    stats
        .map(|s| {
            HashMap::from([
                ("rmse", s.rmse),
                ("mean", s.mean),
                ("median", s.median),
                ("std", s.std),
                ("min", s.min),
                ("max", s.max),
            ])
        })
        .unwrap_or_default()
}

/// Relative pose error over frame-id-matched pose pairs `delta` steps apart.
#[pyclass(module = "visloc", name = "RpeResult", frozen, skip_from_py_object)]
pub struct PyRpeResult {
    #[pyo3(get)]
    delta: usize,
    #[pyo3(get)]
    pair_count: usize,
    #[pyo3(get)]
    matched_count: usize,
    /// `{"rmse", "mean", "median", "std", "min", "max"}` translational error.
    #[pyo3(get)]
    translation: HashMap<&'static str, f64>,
    /// Same statistics for the rotational error, in degrees.
    #[pyo3(get)]
    rotation_deg: HashMap<&'static str, f64>,
    translation_errors: Vec<f64>,
    rotation_errors_deg: Vec<f64>,
    first_frame_ids: Vec<u64>,
}

#[pymethods]
impl PyRpeResult {
    #[getter]
    fn translation_errors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.translation_errors.clone())
    }

    #[getter]
    fn rotation_errors_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.rotation_errors_deg.clone())
    }

    /// Frame id of the first pose of each pair.
    #[getter]
    fn first_frame_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        vec_to_py(py, self.first_frame_ids.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "RpeResult(delta={}, pair_count={}, translation_rmse={:?}, rotation_rmse_deg={:?})",
            self.delta,
            self.pair_count,
            self.translation.get("rmse").copied().unwrap_or(f64::NAN),
            self.rotation_deg.get("rmse").copied().unwrap_or(f64::NAN)
        )
    }
}

/// Relative pose error (TUM protocol) between two trajectories.
///
/// Inputs are as for `evaluate_ate`; RPE needs no alignment. With `(N, 3)`
/// position-only input the rotations are identity, so the rotational error is
/// zero.
#[pyfunction]
#[pyo3(signature = (estimated, reference, *, delta = 1, start_step = 1, estimated_ids = None, reference_ids = None))]
pub fn evaluate_rpe(
    estimated: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    reference: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    delta: usize,
    start_step: usize,
    estimated_ids: Option<PyArrayLikeDyn<'_, u64, AllowTypeChange>>,
    reference_ids: Option<PyArrayLikeDyn<'_, u64, AllowTypeChange>>,
) -> PyResult<PyRpeResult> {
    if delta == 0 || start_step == 0 {
        return Err(PyValueError::new_err("delta and start_step must be >= 1"));
    }
    let estimated = trajectory(estimated, estimated_ids, "estimated")?;
    let reference = trajectory(reference, reference_ids, "reference")?;
    let summary = estimated
        .relative_pose_error_against(&reference, &RelativePoseErrorConfig { delta, start_step });
    if summary.pair_count == 0 {
        return Err(PyValueError::new_err(format!(
            "no frame-id-matched pose pairs {delta} steps apart ({} matched poses)",
            summary.matched_pose_count
        )));
    }
    Ok(PyRpeResult {
        delta: summary.delta,
        pair_count: summary.pair_count,
        matched_count: summary.matched_pose_count,
        translation: stats_dict(summary.translation),
        rotation_deg: stats_dict(summary.rotation_deg),
        translation_errors: summary.errors.iter().map(|e| e.translation_error).collect(),
        rotation_errors_deg: summary
            .errors
            .iter()
            .map(|e| e.rotation_error_deg)
            .collect(),
        first_frame_ids: summary.errors.iter().map(|e| e.first_frame_id).collect(),
    })
}
