//! `visloc.Pose`: rigid SE(3) transforms.

use nalgebra::{Matrix3, Point3, Quaternion, Rotation3, UnitQuaternion, Vector3, Vector6};
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayDyn, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use visloc_core::geometry::{Pose, SE3};

use crate::arrays::{
    matrix3_from_view, matrix3_to_py, matrix4_to_py, require_finite, rows, rows_to_py, vec_to_py,
    vector3_from_view,
};

const ROTATION_TOLERANCE: f64 = 1e-6;

pub(crate) fn rotation_from_matrix(matrix: &Matrix3<f64>) -> PyResult<UnitQuaternion<f64>> {
    require_finite(matrix.as_slice(), "rotation")?;
    let orthogonality = (matrix.transpose() * matrix - Matrix3::identity()).norm();
    if orthogonality > ROTATION_TOLERANCE || matrix.determinant() <= 0.0 {
        return Err(PyValueError::new_err(
            "rotation must be a proper orthonormal 3x3 matrix (R^T R = I, det(R) = +1)",
        ));
    }
    Ok(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(*matrix),
    ))
}

pub(crate) fn unit_quaternion(w: f64, x: f64, y: f64, z: f64) -> PyResult<UnitQuaternion<f64>> {
    require_finite(&[w, x, y, z], "quaternion")?;
    UnitQuaternion::try_new(Quaternion::new(w, x, y, z), 1e-12)
        .ok_or_else(|| PyValueError::new_err("quaternion norm is too small"))
}

/// A rigid transform in SE(3): `T * p = R p + t`.
///
/// When used as a camera pose (COLMAP images, localization results) it is the
/// world-to-camera transform `T_cw`; `camera_center()` returns `-R^T t`.
#[pyclass(module = "visloc", name = "Pose", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPose {
    pub(crate) inner: SE3,
}

impl PyPose {
    pub(crate) fn from_pose(pose: &Pose) -> Self {
        Self {
            inner: pose.world_to_camera.clone(),
        }
    }

    pub(crate) fn to_pose(&self) -> Pose {
        Pose {
            world_to_camera: self.inner.clone(),
        }
    }
}

#[pymethods]
impl PyPose {
    /// Build from a 3x3 rotation matrix and a translation (both optional;
    /// identity by default).
    #[new]
    #[pyo3(signature = (rotation = None, translation = None))]
    fn new(
        rotation: Option<PyArrayLikeDyn<'_, f64, AllowTypeChange>>,
        translation: Option<PyArrayLikeDyn<'_, f64, AllowTypeChange>>,
    ) -> PyResult<Self> {
        let rotation = match rotation {
            Some(r) => rotation_from_matrix(&matrix3_from_view(r.as_array(), "rotation")?)?,
            None => UnitQuaternion::identity(),
        };
        let translation = match translation {
            Some(t) => {
                let t = vector3_from_view(t.as_array(), "translation")?;
                require_finite(t.as_slice(), "translation")?;
                t
            }
            None => Vector3::zeros(),
        };
        Ok(Self {
            inner: SE3::new(rotation, translation),
        })
    }

    #[staticmethod]
    fn identity() -> Self {
        Self {
            inner: SE3::identity(),
        }
    }

    /// Build from a quaternion and a translation.
    ///
    /// The quaternion is `(w, x, y, z)` (COLMAP order) by default, or
    /// `(x, y, z, w)` (TUM / SciPy order) with `scalar_first=False`. It is
    /// normalized.
    #[staticmethod]
    #[pyo3(signature = (quaternion, translation = None, *, scalar_first = true))]
    fn from_quaternion(
        quaternion: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
        translation: Option<PyArrayLikeDyn<'_, f64, AllowTypeChange>>,
        scalar_first: bool,
    ) -> PyResult<Self> {
        let q = quaternion.as_array();
        if q.shape() != [4] {
            return Err(PyValueError::new_err(format!(
                "quaternion must have shape (4,), got {:?}",
                q.shape()
            )));
        }
        let rotation = if scalar_first {
            unit_quaternion(q[[0]], q[[1]], q[[2]], q[[3]])?
        } else {
            unit_quaternion(q[[3]], q[[0]], q[[1]], q[[2]])?
        };
        let translation = match translation {
            Some(t) => vector3_from_view(t.as_array(), "translation")?,
            None => Vector3::zeros(),
        };
        require_finite(translation.as_slice(), "translation")?;
        Ok(Self {
            inner: SE3::new(rotation, translation),
        })
    }

    /// Build from a 4x4 (or 3x4) homogeneous matrix `[R | t]`.
    #[staticmethod]
    fn from_matrix(matrix: PyArrayLikeDyn<'_, f64, AllowTypeChange>) -> PyResult<Self> {
        let m = matrix.as_array();
        let shape = m.shape().to_vec();
        if shape != [4, 4] && shape != [3, 4] {
            return Err(PyValueError::new_err(format!(
                "matrix must have shape (4, 4) or (3, 4), got {shape:?}"
            )));
        }
        let rotation = Matrix3::from_fn(|r, c| m[[r, c]]);
        let translation = Vector3::new(m[[0, 3]], m[[1, 3]], m[[2, 3]]);
        require_finite(translation.as_slice(), "translation")?;
        Ok(Self {
            inner: SE3::new(rotation_from_matrix(&rotation)?, translation),
        })
    }

    /// SE(3) exponential of a tangent `[rho; omega]` (translation first).
    #[staticmethod]
    fn exp(tangent: PyArrayLikeDyn<'_, f64, AllowTypeChange>) -> PyResult<Self> {
        let v = tangent.as_array();
        if v.shape() != [6] {
            return Err(PyValueError::new_err(format!(
                "tangent must have shape (6,), got {:?}",
                v.shape()
            )));
        }
        let xi = Vector6::from_fn(|i, _| v[[i]]);
        require_finite(xi.as_slice(), "tangent")?;
        Ok(Self {
            inner: SE3::exp(&xi),
        })
    }

    /// SE(3) logarithm `[rho; omega]` (translation first).
    fn log<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.inner.log().as_slice().to_vec())
    }

    /// 3x3 rotation matrix `R`.
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix3_to_py(py, &self.inner.rotation.to_rotation_matrix().into_inner())
    }

    /// Translation `t`, shape `(3,)`.
    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_py(py, self.inner.translation.as_slice().to_vec())
    }

    /// Unit quaternion `(w, x, y, z)`; `scalar_first=False` gives `(x, y, z, w)`.
    #[pyo3(signature = (*, scalar_first = true))]
    fn quaternion<'py>(&self, py: Python<'py>, scalar_first: bool) -> Bound<'py, PyArray1<f64>> {
        let q = self.inner.rotation.quaternion();
        let values = if scalar_first {
            vec![q.w, q.i, q.j, q.k]
        } else {
            vec![q.i, q.j, q.k, q.w]
        };
        vec_to_py(py, values)
    }

    /// 4x4 homogeneous matrix.
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix4_to_py(py, &self.inner.matrix())
    }

    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    /// `self * other` (apply `other` first, then `self`).
    fn compose(&self, other: PyRef<'_, PyPose>) -> Self {
        Self {
            inner: self.inner.compose(&other.inner),
        }
    }

    fn __matmul__(&self, other: PyRef<'_, PyPose>) -> Self {
        self.compose(other)
    }

    /// Transform points of shape `(3,)` or `(N, 3)`: `R p + t`.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        points: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let input = rows(points.as_array(), 3, "points")?;
        let mut out = Vec::with_capacity(input.count * 3);
        for row in input.iter() {
            let p = self
                .inner
                .transform_point(&Point3::new(row[0], row[1], row[2]));
            out.extend_from_slice(&[p.x, p.y, p.z]);
        }
        rows_to_py(py, out, 3, input.single)
    }

    /// Camera center in the world frame, `-R^T t`, for a world-to-camera pose.
    fn camera_center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let center = self.to_pose().camera_center_world();
        vec_to_py(py, vec![center.x, center.y, center.z])
    }

    fn __repr__(&self) -> String {
        let q = self.inner.rotation.quaternion();
        let t = self.inner.translation;
        format!(
            "Pose(quaternion_wxyz=[{:?}, {:?}, {:?}, {:?}], translation=[{:?}, {:?}, {:?}])",
            q.w, q.i, q.j, q.k, t.x, t.y, t.z
        )
    }
}
