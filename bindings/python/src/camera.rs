//! `visloc.Camera`: COLMAP camera models with projection and unprojection.

use nalgebra::{Matrix3, Point2, Point3};
use numpy::{AllowTypeChange, PyArray2, PyArrayDyn, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use visloc_core::types::{Camera, CameraModel};

use crate::arrays::{matrix3_to_py, require_finite, rows, rows_to_py};
use crate::pose::PyPose;

/// Supported model names and the parameter counts accepted for each.
///
/// The counts are COLMAP's, plus visloc's two-slot radial extension on
/// `PINHOLE` (`[fx, fy, cx, cy, k1, k2]`, see `Camera::pinhole_radial`) and the
/// radial-only 6-parameter `OPENCV`. `DOUBLE_SPHERE` is not a COLMAP model.
const MODELS: &[(&str, &[usize])] = &[
    ("SIMPLE_PINHOLE", &[3]),
    ("PINHOLE", &[4, 6]),
    ("SIMPLE_RADIAL", &[4]),
    ("RADIAL", &[5]),
    ("OPENCV", &[6, 8]),
    ("FULL_OPENCV", &[12]),
    ("OPENCV_FISHEYE", &[8]),
    ("SIMPLE_RADIAL_FISHEYE", &[4]),
    ("RADIAL_FISHEYE", &[5]),
    ("FOV", &[5]),
    ("DOUBLE_SPHERE", &[6]),
];

pub(crate) fn model_name(model: &CameraModel) -> String {
    match model {
        CameraModel::DoubleSphere => "DOUBLE_SPHERE".to_owned(),
        other => other.colmap_name().unwrap_or("UNKNOWN").to_owned(),
    }
}

fn parse_model(name: &str, param_count: usize) -> PyResult<CameraModel> {
    let upper = name.trim().to_ascii_uppercase();
    let Some((_, counts)) = MODELS.iter().find(|(model, _)| *model == upper) else {
        let supported = MODELS.iter().map(|(m, _)| *m).collect::<Vec<_>>();
        return Err(PyValueError::new_err(format!(
            "unsupported camera model {name:?}; expected one of {supported:?}"
        )));
    };
    if !counts.contains(&param_count) {
        return Err(PyValueError::new_err(format!(
            "camera model {upper} expects {counts:?} params, got {param_count}"
        )));
    }
    Ok(if upper == "DOUBLE_SPHERE" {
        CameraModel::DoubleSphere
    } else {
        CameraModel::from_colmap_name(&upper)
    })
}

/// A camera with a COLMAP model, image size, and parameter vector.
///
/// `params` uses COLMAP's per-model layout, e.g. `PINHOLE` is
/// `[fx, fy, cx, cy]` and `OPENCV` is `[fx, fy, cx, cy, k1, k2, p1, p2]`.
#[pyclass(module = "visloc", name = "Camera", eq, frozen, skip_from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyCamera {
    pub(crate) inner: Camera,
}

#[pymethods]
impl PyCamera {
    #[new]
    #[pyo3(signature = (model, width, height, params, id = 1))]
    fn new(model: &str, width: u32, height: u32, params: Vec<f64>, id: u64) -> PyResult<Self> {
        require_finite(&params, "params")?;
        let model = parse_model(model, params.len())?;
        Ok(Self {
            inner: Camera {
                id,
                model,
                width,
                height,
                params,
            },
        })
    }

    /// Distortion-free pinhole camera `[fx, fy, cx, cy]`.
    #[staticmethod]
    #[pyo3(signature = (fx, fy, cx, cy, width, height, id = 1))]
    fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32, id: u64) -> Self {
        Self {
            inner: Camera::pinhole(id, width, height, fx, fy, cx, cy),
        }
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }

    /// COLMAP model name (or `"DOUBLE_SPHERE"`).
    #[getter]
    fn model(&self) -> String {
        model_name(&self.inner.model)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::arrays::vec_to_py(py, self.inner.params.clone())
    }

    /// `(fx, fy, cx, cy)`, or `None` for an unknown model.
    #[getter]
    fn intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.intrinsics()
    }

    /// 3x3 calibration matrix `K` built from [`intrinsics`].
    fn calibration_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (fx, fy, cx, cy) = self
            .inner
            .intrinsics()
            .ok_or_else(|| PyValueError::new_err("camera model has no pinhole intrinsics"))?;
        let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        Ok(matrix3_to_py(py, &k))
    }

    /// Project 3-D points to pixels.
    ///
    /// `points` has shape `(3,)` or `(N, 3)`. Points are in the camera frame,
    /// or in the world frame when `pose` (world-to-camera) is given. Points
    /// that cannot be projected (behind the camera, outside the model's
    /// domain) yield `NaN` rows.
    #[pyo3(signature = (points, pose = None))]
    fn project<'py>(
        &self,
        py: Python<'py>,
        points: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
        pose: Option<PyRef<'py, PyPose>>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let input = rows(points.as_array(), 3, "points")?;
        let mut out = Vec::with_capacity(input.count * 2);
        for row in input.iter() {
            let mut point = Point3::new(row[0], row[1], row[2]);
            if let Some(pose) = &pose {
                point = pose.inner.transform_point(&point);
            }
            match self.inner.project(&point) {
                Some(pixel) => out.extend_from_slice(&[pixel.x, pixel.y]),
                None => out.extend_from_slice(&[f64::NAN, f64::NAN]),
            }
        }
        rows_to_py(py, out, 2, input.single)
    }

    /// Back-project pixels to unit-norm camera-frame rays.
    ///
    /// `pixels` has shape `(2,)` or `(N, 2)`; the result has shape `(3,)` or
    /// `(N, 3)`, with `NaN` rows where the model cannot unproject. Multiply a
    /// ray by a depth along it to recover a 3-D point.
    fn unproject<'py>(
        &self,
        py: Python<'py>,
        pixels: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let input = rows(pixels.as_array(), 2, "pixels")?;
        let mut out = Vec::with_capacity(input.count * 3);
        for row in input.iter() {
            match self.inner.unit_ray_from_pixel(&Point2::new(row[0], row[1])) {
                Some(ray) => out.extend_from_slice(&[ray.x, ray.y, ray.z]),
                None => out.extend_from_slice(&[f64::NAN; 3]),
            }
        }
        rows_to_py(py, out, 3, input.single)
    }

    /// Undistort pixels to normalized image coordinates `(x/z, y/z)`.
    fn normalize<'py>(
        &self,
        py: Python<'py>,
        pixels: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let input = rows(pixels.as_array(), 2, "pixels")?;
        let mut out = Vec::with_capacity(input.count * 2);
        for row in input.iter() {
            match self.inner.normalize_pixel(&Point2::new(row[0], row[1])) {
                Some(xy) => out.extend_from_slice(&[xy.x, xy.y]),
                None => out.extend_from_slice(&[f64::NAN; 2]),
            }
        }
        rows_to_py(py, out, 2, input.single)
    }

    fn __repr__(&self) -> String {
        format!(
            "Camera(model={:?}, width={}, height={}, params={:?}, id={})",
            model_name(&self.inner.model),
            self.inner.width,
            self.inner.height,
            self.inner.params,
            self.inner.id
        )
    }
}
