//! NumPy conversion helpers shared by the binding modules.

use nalgebra::{Matrix3, Matrix4, Vector3};
use numpy::ndarray::{Array1, Array2, ArrayViewD};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Row-major rows of a `(width,)` or `(N, width)` input array.
pub(crate) struct Rows {
    pub(crate) data: Vec<f64>,
    pub(crate) count: usize,
    pub(crate) width: usize,
    /// `true` when the input was a single 1-D row; outputs mirror that shape.
    pub(crate) single: bool,
}

impl Rows {
    pub(crate) fn row(&self, index: usize) -> &[f64] {
        &self.data[index * self.width..(index + 1) * self.width]
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks_exact(self.width.max(1)).take(self.count)
    }
}

pub(crate) fn rows(view: ArrayViewD<'_, f64>, width: usize, name: &str) -> PyResult<Rows> {
    let shape = view.shape().to_vec();
    let (count, single) = match shape.as_slice() {
        [w] if *w == width => (1, true),
        [n, w] if *w == width => (*n, false),
        _ => {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({width},) or (N, {width}), got {shape:?}"
            )))
        }
    };
    Ok(Rows {
        data: view.iter().copied().collect(),
        count,
        width,
        single,
    })
}

/// Return `(N, width)` (or `(width,)` when `single`) as a NumPy array.
pub(crate) fn rows_to_py<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    width: usize,
    single: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let count = data.len() / width.max(1);
    let shape = if single {
        vec![width]
    } else {
        vec![count, width]
    };
    let array = numpy::ndarray::ArrayD::from_shape_vec(shape, data)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

pub(crate) fn matrix3_from_view(view: ArrayViewD<'_, f64>, name: &str) -> PyResult<Matrix3<f64>> {
    if view.shape() != [3, 3] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (3, 3), got {:?}",
            view.shape()
        )));
    }
    Ok(Matrix3::from_fn(|r, c| view[[r, c]]))
}

pub(crate) fn vector3_from_view(view: ArrayViewD<'_, f64>, name: &str) -> PyResult<Vector3<f64>> {
    if view.shape() != [3] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (3,), got {:?}",
            view.shape()
        )));
    }
    Ok(Vector3::new(view[[0]], view[[1]], view[[2]]))
}

pub(crate) fn matrix3_to_py<'py>(py: Python<'py>, m: &Matrix3<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((3, 3), |(r, c)| m[(r, c)]).into_pyarray(py)
}

pub(crate) fn matrix4_to_py<'py>(py: Python<'py>, m: &Matrix4<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((4, 4), |(r, c)| m[(r, c)]).into_pyarray(py)
}

pub(crate) fn vec_to_py<'py, T: numpy::Element>(
    py: Python<'py>,
    values: Vec<T>,
) -> Bound<'py, PyArray1<T>> {
    Array1::from(values).into_pyarray(py)
}

pub(crate) fn require_finite(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )))
    }
}
