use crate::geometry::Pose;
use nalgebra::Point2;

use super::{Camera, CameraId, LandmarkId};

pub type FrameId = u64;

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub id: FrameId,
    pub camera_id: CameraId,
    pub keypoints: Vec<Point2<f64>>,
    pub descriptors: Vec<Vec<f32>>,
    pub pose: Option<Pose>,
}

impl Frame {
    pub fn new(id: FrameId, camera_id: CameraId) -> Self {
        Self {
            id,
            camera_id,
            keypoints: Vec::new(),
            descriptors: Vec::new(),
            pose: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Keyframe {
    pub frame: Frame,
    pub observations: Vec<Observation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub frame_id: FrameId,
    pub landmark_id: LandmarkId,
    pub keypoint_index: usize,
    pub xy: Point2<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryImage {
    pub camera: Camera,
    pub keypoints: Vec<Point2<f64>>,
    pub descriptors: Vec<Vec<f32>>,
}

impl QueryImage {
    pub fn from_frame(frame: &Frame, camera: Camera) -> Self {
        Self {
            camera,
            keypoints: frame.keypoints.clone(),
            descriptors: frame.descriptors.clone(),
        }
    }
}
