//! `visloc.Image` / `visloc.Reconstruction`: COLMAP sparse models.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use nalgebra::{Point2, Point3};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLikeDyn};
use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyValueError};
use pyo3::prelude::*;
use visloc_core::types::{Frame, Keyframe, Landmark, Observation, VisualMap};
use visloc_io::colmap::{
    read_colmap_binary_model, read_colmap_text_model, write_colmap_binary_model,
    write_colmap_text_model, ColmapError,
};
use visloc_io::descriptors::{read_landmark_descriptors_txt, DescriptorStoreError};

use crate::arrays::{require_finite, rows, vec_to_py};
use crate::camera::PyCamera;
use crate::pose::PyPose;

pub(crate) fn colmap_error(error: ColmapError) -> PyErr {
    match error {
        ColmapError::Io(io) => PyIOError::new_err(io.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn descriptor_error(error: DescriptorStoreError) -> PyErr {
    match error {
        DescriptorStoreError::Io(io) => PyIOError::new_err(io.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn ids_from_array(view: numpy::ndarray::ArrayViewD<'_, i64>, name: &str) -> PyResult<Vec<i64>> {
    if view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "{name} must be one-dimensional, got shape {:?}",
            view.shape()
        )));
    }
    Ok(view.iter().copied().collect())
}

/// A registered image: camera id, world-to-camera pose, and 2-D keypoints with
/// their observed 3-D point ids (`-1` when a keypoint is untriangulated).
///
/// visloc's map model does not keep COLMAP image names; written models use
/// `image_<id>.jpg`.
#[pyclass(module = "visloc", name = "Image", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyImage {
    pub(crate) inner: Keyframe,
}

#[pymethods]
impl PyImage {
    #[new]
    #[pyo3(signature = (id, camera_id, pose, keypoints = None, point3d_ids = None))]
    fn new(
        id: u64,
        camera_id: u64,
        pose: PyRef<'_, PyPose>,
        keypoints: Option<PyArrayLikeDyn<'_, f64, AllowTypeChange>>,
        point3d_ids: Option<PyArrayLikeDyn<'_, i64, AllowTypeChange>>,
    ) -> PyResult<Self> {
        let mut frame = Frame::new(id, camera_id);
        frame.pose = Some(pose.to_pose());
        if let Some(keypoints) = keypoints {
            let view = keypoints.as_array();
            if view.ndim() == 2 && view.shape()[0] == 0 {
                // Allow an empty (0, 2) or (0, k) array.
            } else {
                let input = rows(view, 2, "keypoints")?;
                if input.single {
                    return Err(PyValueError::new_err("keypoints must have shape (N, 2)"));
                }
                require_finite(&input.data, "keypoints")?;
                frame.keypoints = input.iter().map(|r| Point2::new(r[0], r[1])).collect();
            }
        }
        let mut observations = Vec::new();
        if let Some(ids) = point3d_ids {
            let ids = ids_from_array(ids.as_array(), "point3d_ids")?;
            if ids.len() != frame.keypoints.len() {
                return Err(PyValueError::new_err(format!(
                    "point3d_ids has {} entries but there are {} keypoints",
                    ids.len(),
                    frame.keypoints.len()
                )));
            }
            for (keypoint_index, &point_id) in ids.iter().enumerate() {
                if point_id >= 0 {
                    observations.push(Observation {
                        frame_id: id,
                        landmark_id: point_id as u64,
                        keypoint_index,
                        xy: frame.keypoints[keypoint_index],
                    });
                } else if point_id != -1 {
                    return Err(PyValueError::new_err(
                        "point3d_ids must be >= 0, or -1 for an untriangulated keypoint",
                    ));
                }
            }
        }
        Ok(Self {
            inner: Keyframe {
                frame,
                observations,
            },
        })
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.frame.id
    }

    #[getter]
    fn camera_id(&self) -> u64 {
        self.inner.frame.camera_id
    }

    /// World-to-camera pose (COLMAP `QW QX QY QZ TX TY TZ`), if known.
    #[getter]
    fn pose(&self) -> Option<PyPose> {
        self.inner.frame.pose.as_ref().map(PyPose::from_pose)
    }

    /// Keypoints, shape `(N, 2)`.
    #[getter]
    fn keypoints<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let keypoints = &self.inner.frame.keypoints;
        Array2::from_shape_fn((keypoints.len(), 2), |(i, c)| {
            if c == 0 {
                keypoints[i].x
            } else {
                keypoints[i].y
            }
        })
        .into_pyarray(py)
    }

    /// Per-keypoint 3-D point id, shape `(N,)`, `-1` when untriangulated.
    #[getter]
    fn point3d_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let mut ids = vec![-1i64; self.inner.frame.keypoints.len()];
        for observation in &self.inner.observations {
            if let Some(slot) = ids.get_mut(observation.keypoint_index) {
                if *slot == -1 {
                    *slot = observation.landmark_id as i64;
                }
            }
        }
        vec_to_py(py, ids)
    }

    #[getter]
    fn num_observations(&self) -> usize {
        self.inner.observations.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Image(id={}, camera_id={}, keypoints={}, observations={})",
            self.inner.frame.id,
            self.inner.frame.camera_id,
            self.inner.frame.keypoints.len(),
            self.inner.observations.len()
        )
    }
}

/// A sparse COLMAP model: cameras, registered images, and 3-D points (with
/// optional per-point descriptors used by `localize`).
#[pyclass(module = "visloc", name = "Reconstruction", skip_from_py_object)]
#[derive(Clone, Default)]
pub struct PyReconstruction {
    pub(crate) map: VisualMap,
}

impl PyReconstruction {
    fn sorted_landmarks(&self) -> Vec<&Landmark> {
        let mut landmarks = self.map.landmarks.values().collect::<Vec<_>>();
        landmarks.sort_by_key(|landmark| landmark.id);
        landmarks
    }

    /// Copy of the map whose landmark `TRACK[]`s are rebuilt from the image
    /// observations, so written `points3D` files agree with `images` files.
    fn map_for_writing(&self) -> VisualMap {
        let mut map = self.map.clone();
        let mut tracks: BTreeMap<u64, Vec<Observation>> = BTreeMap::new();
        for keyframe in map.keyframes.values() {
            for observation in &keyframe.observations {
                tracks
                    .entry(observation.landmark_id)
                    .or_default()
                    .push(observation.clone());
            }
        }
        for landmark in map.landmarks.values_mut() {
            if landmark.observations.is_empty() {
                landmark.observations = tracks.remove(&landmark.id).unwrap_or_default();
            }
        }
        map
    }
}

#[pymethods]
impl PyReconstruction {
    /// An empty reconstruction.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Read a COLMAP text model (`cameras.txt`, `images.txt`, `points3D.txt`).
    #[staticmethod]
    fn read_text(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let map = py
            .detach(|| read_colmap_text_model(&path))
            .map_err(colmap_error)?;
        Ok(Self { map })
    }

    /// Read a COLMAP binary model (`cameras.bin`, `images.bin`, `points3D.bin`).
    #[staticmethod]
    fn read_binary(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let map = py
            .detach(|| read_colmap_binary_model(&path))
            .map_err(colmap_error)?;
        Ok(Self { map })
    }

    /// Read a COLMAP model directory, preferring the binary files when present.
    #[staticmethod]
    fn read(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        if path.join("cameras.bin").is_file() {
            Self::read_binary(py, path)
        } else if path.join("cameras.txt").is_file() {
            Self::read_text(py, path)
        } else {
            Err(PyFileNotFoundError::new_err(format!(
                "no cameras.bin or cameras.txt in {}",
                path.display()
            )))
        }
    }

    /// Write a COLMAP text model into `path` (created if needed).
    fn write_text(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let map = self.map_for_writing();
        py.detach(|| write_colmap_text_model(&map, &path))
            .map_err(colmap_error)
    }

    /// Write a COLMAP binary model into `path` (created if needed).
    fn write_binary(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let map = self.map_for_writing();
        py.detach(|| write_colmap_binary_model(&map, &path))
            .map_err(colmap_error)
    }

    /// Cameras keyed by id (a copy; use `add_camera` to modify).
    #[getter]
    fn cameras(&self) -> BTreeMap<u64, PyCamera> {
        self.map
            .cameras
            .iter()
            .map(|(id, camera)| {
                (
                    *id,
                    PyCamera {
                        inner: camera.clone(),
                    },
                )
            })
            .collect()
    }

    /// Images keyed by id (a copy; use `add_image` to modify).
    #[getter]
    fn images(&self) -> BTreeMap<u64, PyImage> {
        self.map
            .keyframes
            .iter()
            .map(|(id, keyframe)| {
                (
                    *id,
                    PyImage {
                        inner: keyframe.clone(),
                    },
                )
            })
            .collect()
    }

    /// 3-D point ids sorted ascending, shape `(M,)`.
    #[getter]
    fn point3d_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        vec_to_py(
            py,
            self.sorted_landmarks()
                .iter()
                .map(|l| l.id)
                .collect::<Vec<_>>(),
        )
    }

    /// 3-D point positions aligned with `point3d_ids`, shape `(M, 3)`.
    #[getter]
    fn points3d<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let landmarks = self.sorted_landmarks();
        Array2::from_shape_fn((landmarks.len(), 3), |(i, c)| landmarks[i].position[c])
            .into_pyarray(py)
    }

    /// Per-point descriptors aligned with `point3d_ids`, shape `(M, D)`, or
    /// `None` unless every point has a descriptor of the same length.
    #[getter]
    fn point3d_descriptors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f32>>> {
        let landmarks = self.sorted_landmarks();
        let dim = landmarks.first()?.descriptor.as_ref()?.len();
        let mut data = Vec::with_capacity(landmarks.len() * dim);
        for landmark in &landmarks {
            let descriptor = landmark.descriptor.as_ref()?;
            if descriptor.len() != dim {
                return None;
            }
            data.extend_from_slice(descriptor);
        }
        Array2::from_shape_vec((landmarks.len(), dim), data)
            .ok()
            .map(|array| array.into_pyarray(py))
    }

    #[getter]
    fn num_cameras(&self) -> usize {
        self.map.cameras.len()
    }

    #[getter]
    fn num_images(&self) -> usize {
        self.map.keyframes.len()
    }

    #[getter]
    fn num_points3d(&self) -> usize {
        self.map.landmarks.len()
    }

    /// Insert or replace a camera (keyed by `camera.id`).
    fn add_camera(&mut self, camera: PyRef<'_, PyCamera>) {
        self.map
            .cameras
            .insert(camera.inner.id, camera.inner.clone());
    }

    /// Insert or replace an image (keyed by `image.id`).
    fn add_image(&mut self, image: PyRef<'_, PyImage>) {
        self.map
            .keyframes
            .insert(image.inner.frame.id, image.inner.clone());
    }

    /// Replace all 3-D points.
    ///
    /// `ids` has shape `(M,)` (unique, non-negative), `points` `(M, 3)`, and the
    /// optional `descriptors` `(M, D)` (cast to float32).
    #[pyo3(signature = (ids, points, descriptors = None))]
    fn set_points3d(
        &mut self,
        ids: PyArrayLikeDyn<'_, i64, AllowTypeChange>,
        points: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
        descriptors: Option<PyArrayLikeDyn<'_, f32, AllowTypeChange>>,
    ) -> PyResult<()> {
        let ids = ids_from_array(ids.as_array(), "ids")?;
        let view = points.as_array();
        let positions = if view.ndim() == 2 && view.shape()[0] == 0 {
            Vec::new()
        } else {
            let input = rows(view, 3, "points")?;
            if input.single {
                return Err(PyValueError::new_err("points must have shape (M, 3)"));
            }
            require_finite(&input.data, "points")?;
            input
                .iter()
                .map(|r| Point3::new(r[0], r[1], r[2]))
                .collect()
        };
        if ids.len() != positions.len() {
            return Err(PyValueError::new_err(format!(
                "ids has {} entries but points has {} rows",
                ids.len(),
                positions.len()
            )));
        }
        let descriptor_rows = match descriptors {
            Some(descriptors) => {
                let view = descriptors.as_array();
                if view.ndim() != 2 || view.shape()[0] != ids.len() {
                    return Err(PyValueError::new_err(format!(
                        "descriptors must have shape ({}, D), got {:?}",
                        ids.len(),
                        view.shape()
                    )));
                }
                Some(
                    view.outer_iter()
                        .map(|row| row.iter().copied().collect::<Vec<f32>>())
                        .collect::<Vec<_>>(),
                )
            }
            None => None,
        };
        let mut seen = HashSet::with_capacity(ids.len());
        let mut landmarks = std::collections::HashMap::with_capacity(ids.len());
        for (index, (&id, position)) in ids.iter().zip(positions).enumerate() {
            let id = u64::try_from(id)
                .map_err(|_| PyValueError::new_err("point ids must be non-negative"))?;
            if !seen.insert(id) {
                return Err(PyValueError::new_err(format!("duplicate point id {id}")));
            }
            let mut landmark = Landmark::new(id, position);
            if let Some(rows) = &descriptor_rows {
                landmark.descriptor = Some(rows[index].clone());
            }
            landmarks.insert(id, landmark);
        }
        self.map.landmarks = landmarks;
        self.map.landmark_position_covariances.clear();
        Ok(())
    }

    /// Attach descriptors from a landmark-descriptor text file
    /// (`LANDMARK_ID D0 D1 ...` per line). Returns the number of points that
    /// received a descriptor; ids without a matching point are ignored.
    fn load_point3d_descriptors(&mut self, py: Python<'_>, path: PathBuf) -> PyResult<usize> {
        let store = py
            .detach(|| read_landmark_descriptors_txt(&path))
            .map_err(descriptor_error)?;
        let mut attached = 0;
        for (id, descriptor) in store.iter() {
            if let Some(landmark) = self.map.landmarks.get_mut(&id) {
                landmark.descriptor = Some(descriptor.to_vec());
                attached += 1;
            }
        }
        Ok(attached)
    }

    /// Structural validation issues (empty when the model is consistent).
    fn validate(&self) -> Vec<String> {
        self.map
            .validate()
            .issues
            .iter()
            .map(|issue| format!("{issue:?}"))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Reconstruction(cameras={}, images={}, points3d={})",
            self.map.cameras.len(),
            self.map.keyframes.len(),
            self.map.landmarks.len()
        )
    }
}
