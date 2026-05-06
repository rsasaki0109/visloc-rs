use crate::types::Camera;
use nalgebra::{Point2, Point3};

use super::Pose;

pub fn reproject(camera: &Camera, pose: &Pose, point_world: &Point3<f64>) -> Option<Point2<f64>> {
    let point_camera = pose.transform_world_point(point_world);
    camera.project(&point_camera)
}
