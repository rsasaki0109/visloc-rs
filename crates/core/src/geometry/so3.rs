use nalgebra::{Matrix3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, PartialEq)]
pub struct SO3 {
    rotation: UnitQuaternion<f64>,
}

impl SO3 {
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
        }
    }

    pub fn from_quaternion(rotation: UnitQuaternion<f64>) -> Self {
        Self { rotation }
    }

    pub fn from_matrix(matrix: &Matrix3<f64>) -> Self {
        Self {
            rotation: UnitQuaternion::from_matrix(matrix),
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            rotation: self.rotation.inverse(),
        }
    }

    pub fn transform_vector(&self, vector: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transform_vector(vector)
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        self.rotation.to_rotation_matrix().into_inner()
    }

    pub fn quaternion(&self) -> &UnitQuaternion<f64> {
        &self.rotation
    }
}

impl Default for SO3 {
    fn default() -> Self {
        Self::identity()
    }
}
