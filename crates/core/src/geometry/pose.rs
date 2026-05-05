use super::SE3;
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub world_to_camera: SE3,
}

impl Pose {
    pub fn identity() -> Self {
        Self {
            world_to_camera: SE3::identity(),
        }
    }

    pub fn from_world_to_camera(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        Self {
            world_to_camera: SE3::new(rotation, translation),
        }
    }

    pub fn transform_world_point(&self, point: &Point3<f64>) -> Point3<f64> {
        self.world_to_camera.transform_point(point)
    }

    pub fn camera_to_world(&self) -> SE3 {
        self.world_to_camera.inverse()
    }

    pub fn camera_center_world(&self) -> Point3<f64> {
        Point3::from(self.camera_to_world().translation)
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        self.world_to_camera.matrix()
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self::identity()
    }
}
