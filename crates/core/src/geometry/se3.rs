use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};

#[derive(Debug, Clone, PartialEq)]
pub struct SE3 {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl SE3 {
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn new(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.rotation.transform_point(point).coords + self.translation)
    }

    pub fn transform_vector(&self, vector: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.transform_vector(vector)
    }

    pub fn compose(&self, other: &SE3) -> Self {
        Self::new(
            self.rotation * other.rotation,
            self.rotation.transform_vector(&other.translation) + self.translation,
        )
    }

    pub fn inverse(&self) -> Self {
        let rotation_inv = self.rotation.inverse();
        let translation_inv = -(rotation_inv.transform_vector(&self.translation));
        Self::new(rotation_inv, translation_inv)
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        let mut matrix = Matrix4::identity();
        matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.to_rotation_matrix().into_inner());
        matrix
            .fixed_view_mut::<3, 1>(0, 3)
            .copy_from(&self.translation);
        matrix
    }
}

impl Default for SE3 {
    fn default() -> Self {
        Self::identity()
    }
}
