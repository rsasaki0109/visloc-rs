use nalgebra::{DMatrix, DVector, Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;

#[derive(Debug, Clone, PartialEq)]
pub struct Correspondence2D3D {
    pub point2d: Point2<f64>,
    pub point3d: Point3<f64>,
}

pub trait PoseEstimator {
    fn estimate_pose(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<Pose>;
    fn minimum_correspondences(&self) -> usize;
}

pub trait PoseRefiner {
    fn refine_pose(
        &self,
        initial_pose: &Pose,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<Pose>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DltPnP {
    pub min_correspondences: usize,
}

impl Default for DltPnP {
    fn default() -> Self {
        Self {
            min_correspondences: 6,
        }
    }
}

impl PoseEstimator for DltPnP {
    fn estimate_pose(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<Pose> {
        if correspondences.len() < self.minimum_correspondences() {
            return None;
        }

        let mut a = DMatrix::<f64>::zeros(correspondences.len() * 2, 12);
        for (i, correspondence) in correspondences.iter().enumerate() {
            let normalized = camera.normalize_pixel(&correspondence.point2d)?;
            let x = normalized.x;
            let y = normalized.y;
            let world = correspondence.point3d;
            let row = i * 2;

            a[(row, 0)] = -world.x;
            a[(row, 1)] = -world.y;
            a[(row, 2)] = -world.z;
            a[(row, 3)] = -1.0;
            a[(row, 8)] = x * world.x;
            a[(row, 9)] = x * world.y;
            a[(row, 10)] = x * world.z;
            a[(row, 11)] = x;

            a[(row + 1, 4)] = -world.x;
            a[(row + 1, 5)] = -world.y;
            a[(row + 1, 6)] = -world.z;
            a[(row + 1, 7)] = -1.0;
            a[(row + 1, 8)] = y * world.x;
            a[(row + 1, 9)] = y * world.y;
            a[(row + 1, 10)] = y * world.z;
            a[(row + 1, 11)] = y;
        }

        let svd = a.svd(true, true);
        let v_t = svd.v_t?;
        let last_row = v_t.row(v_t.nrows() - 1);

        let mut projection = [0.0_f64; 12];
        for i in 0..12 {
            projection[i] = last_row[i];
        }

        let mut m = Matrix3::new(
            projection[0],
            projection[1],
            projection[2],
            projection[4],
            projection[5],
            projection[6],
            projection[8],
            projection[9],
            projection[10],
        );
        let mut translation = Vector3::new(projection[3], projection[7], projection[11]);

        let row_scale = (m.row(0).norm() + m.row(1).norm() + m.row(2).norm()) / 3.0;
        if row_scale <= f64::EPSILON {
            return None;
        }

        let mut scale = row_scale;
        if (m / scale).determinant() < 0.0 {
            scale = -scale;
        }
        m /= scale;
        translation /= scale;

        let rotation_svd = m.svd(true, true);
        let u = rotation_svd.u?;
        let v_t = rotation_svd.v_t?;
        let mut rotation = u * v_t;
        if rotation.determinant() < 0.0 {
            rotation = -rotation;
        }

        Some(Pose::from_world_to_camera(
            UnitQuaternion::from_matrix(&rotation),
            translation,
        ))
    }

    fn minimum_correspondences(&self) -> usize {
        self.min_correspondences
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussNewtonPoseRefiner {
    pub iterations: usize,
    pub damping: f64,
    pub finite_difference_epsilon: f64,
    pub min_error_reduction: f64,
}

impl Default for GaussNewtonPoseRefiner {
    fn default() -> Self {
        Self {
            iterations: 8,
            damping: 1.0e-6,
            finite_difference_epsilon: 1.0e-6,
            min_error_reduction: 1.0e-9,
        }
    }
}

impl PoseRefiner for GaussNewtonPoseRefiner {
    fn refine_pose(
        &self,
        initial_pose: &Pose,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<Pose> {
        if correspondences.is_empty() {
            return None;
        }

        let mut pose = initial_pose.clone();
        let mut best_error = mean_squared_reprojection_error(&pose, correspondences, camera)?;

        for _ in 0..self.iterations {
            let residual = reprojection_residual(&pose, correspondences, camera)?;
            let jacobian = numeric_reprojection_jacobian(
                &pose,
                correspondences,
                camera,
                self.finite_difference_epsilon,
            )?;
            let j_t = jacobian.transpose();
            let mut hessian = &j_t * &jacobian;
            for diagonal in 0..hessian.nrows().min(hessian.ncols()) {
                hessian[(diagonal, diagonal)] += self.damping;
            }
            let gradient = &j_t * residual;
            let step = hessian.lu().solve(&(-gradient))?;
            if !step.iter().all(|value| value.is_finite()) {
                return Some(pose);
            }

            let candidate = perturb_pose(&pose, &step);
            let Some(candidate_error) =
                mean_squared_reprojection_error(&candidate, correspondences, camera)
            else {
                break;
            };

            if candidate_error + self.min_error_reduction >= best_error {
                break;
            }

            pose = candidate;
            best_error = candidate_error;
        }

        Some(pose)
    }
}

fn mean_squared_reprojection_error(
    pose: &Pose,
    correspondences: &[Correspondence2D3D],
    camera: &Camera,
) -> Option<f64> {
    let residual = reprojection_residual(pose, correspondences, camera)?;
    Some(residual.norm_squared() / correspondences.len() as f64)
}

fn reprojection_residual(
    pose: &Pose,
    correspondences: &[Correspondence2D3D],
    camera: &Camera,
) -> Option<DVector<f64>> {
    let mut residual = DVector::<f64>::zeros(correspondences.len() * 2);
    for (index, correspondence) in correspondences.iter().enumerate() {
        let projected = camera.project(&pose.transform_world_point(&correspondence.point3d))?;
        let offset = index * 2;
        residual[offset] = projected.x - correspondence.point2d.x;
        residual[offset + 1] = projected.y - correspondence.point2d.y;
    }
    Some(residual)
}

fn numeric_reprojection_jacobian(
    pose: &Pose,
    correspondences: &[Correspondence2D3D],
    camera: &Camera,
    epsilon: f64,
) -> Option<DMatrix<f64>> {
    if epsilon <= 0.0 {
        return None;
    }

    let base = reprojection_residual(pose, correspondences, camera)?;
    let mut jacobian = DMatrix::<f64>::zeros(base.len(), 6);
    for parameter_index in 0..6 {
        let mut delta = DVector::<f64>::zeros(6);
        delta[parameter_index] = epsilon;
        let perturbed_pose = perturb_pose(pose, &delta);
        let perturbed = reprojection_residual(&perturbed_pose, correspondences, camera)?;
        let column = (perturbed - &base) / epsilon;
        jacobian.set_column(parameter_index, &column);
    }
    Some(jacobian)
}

fn perturb_pose(pose: &Pose, delta: &DVector<f64>) -> Pose {
    let rotation_delta =
        UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2]));
    let translation_delta = Vector3::new(delta[3], delta[4], delta[5]);
    Pose::from_world_to_camera(
        rotation_delta * pose.world_to_camera.rotation,
        pose.world_to_camera.translation + translation_delta,
    )
}

#[cfg(test)]
mod tests {
    use super::{Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner, PoseEstimator, PoseRefiner};
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::Camera;

    #[test]
    fn estimates_identity_pose_from_projected_points() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let points = vec![
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 5.5),
            Point3::new(0.0, 0.0, 6.0),
            Point3::new(0.5, -0.25, 7.0),
        ];
        let correspondences = points
            .into_iter()
            .map(|point3d| Correspondence2D3D {
                point2d: camera
                    .project(&pose.transform_world_point(&point3d))
                    .unwrap(),
                point3d,
            })
            .collect::<Vec<_>>();

        let estimated = DltPnP::default()
            .estimate_pose(&correspondences, &camera)
            .unwrap();
        assert!(estimated.world_to_camera.translation.norm() < 1.0e-6);
    }

    #[test]
    fn gauss_newton_refiner_reduces_reprojection_error() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let true_pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let initial_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.005)),
            Vector3::new(0.03, -0.02, 0.05),
        );
        let points = vec![
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 5.5),
            Point3::new(0.0, 0.0, 6.0),
            Point3::new(0.5, -0.25, 7.0),
            Point3::new(-0.4, 0.6, 6.5),
            Point3::new(0.8, 0.3, 8.0),
        ];
        let correspondences = points
            .into_iter()
            .map(|point3d| Correspondence2D3D {
                point2d: camera
                    .project(&true_pose.transform_world_point(&point3d))
                    .unwrap(),
                point3d,
            })
            .collect::<Vec<_>>();

        let before = mean_pixel_error(&camera, &initial_pose, &correspondences);
        let refined = GaussNewtonPoseRefiner::default()
            .refine_pose(&initial_pose, &correspondences, &camera)
            .unwrap();
        let after = mean_pixel_error(&camera, &refined, &correspondences);

        assert!(after < before * 0.1, "before={before} after={after}");
    }

    fn mean_pixel_error(
        camera: &Camera,
        pose: &Pose,
        correspondences: &[Correspondence2D3D],
    ) -> f64 {
        let errors = correspondences
            .iter()
            .map(|correspondence| {
                let projected = camera
                    .project(&pose.transform_world_point(&correspondence.point3d))
                    .unwrap_or(Point2::new(f64::INFINITY, f64::INFINITY));
                (projected - correspondence.point2d).norm()
            })
            .collect::<Vec<_>>();
        errors.iter().sum::<f64>() / errors.len() as f64
    }
}
