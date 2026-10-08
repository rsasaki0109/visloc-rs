#![forbid(unsafe_code)]
//! Python bindings for `visloc-rs`.
//!
//! The extension module `visloc._visloc` is re-exported by the pure-Python
//! package `visloc` (see `python/visloc/__init__.py`). It binds the
//! stable-intent surface documented in `docs/api_stability.md`: cameras,
//! SE(3) poses, COLMAP text/binary IO, map-based localization (PnP + RANSAC),
//! and trajectory evaluation (ATE / RPE with Umeyama alignment).

use pyo3::prelude::*;

mod arrays;
mod camera;
mod colmap;
mod localization;
mod pose;
mod trajectory;

#[pymodule]
fn _visloc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<camera::PyCamera>()?;
    m.add_class::<pose::PyPose>()?;
    m.add_class::<colmap::PyImage>()?;
    m.add_class::<colmap::PyReconstruction>()?;
    m.add_class::<localization::PyLocalizationResult>()?;
    m.add_class::<localization::PyPnPRansacResult>()?;
    m.add_function(wrap_pyfunction!(localization::localize, m)?)?;
    m.add_function(wrap_pyfunction!(localization::estimate_pose_pnp_ransac, m)?)?;
    m.add_class::<trajectory::PySimilarityTransform>()?;
    m.add_class::<trajectory::PyAteResult>()?;
    m.add_class::<trajectory::PyRpeResult>()?;
    m.add_function(wrap_pyfunction!(trajectory::umeyama_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(trajectory::evaluate_ate, m)?)?;
    m.add_function(wrap_pyfunction!(trajectory::evaluate_rpe, m)?)?;
    Ok(())
}
