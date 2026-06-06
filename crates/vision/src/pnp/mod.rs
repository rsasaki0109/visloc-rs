use nalgebra::{DMatrix, DVector, Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;

mod p3p;
pub use p3p::P3PGrunert;

#[derive(Debug, Clone, PartialEq)]
pub struct Correspondence2D3D {
    pub point2d: Point2<f64>,
    pub point3d: Point3<f64>,
    /// Optional matcher confidence for RANSAC sampling. `None` means the
    /// correspondence has no confidence signal and should be treated uniformly.
    pub confidence: Option<f32>,
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

        // Hartley-normalise the 3D points (centroid to origin, mean distance to
        // √3) before the DLT. Without this the design matrix mixes world
        // coordinates of magnitude ‖p‖ with the homogeneous `1` and the
        // intrinsics-normalised pixels (magnitude ~1); when the points sit far
        // from the origin relative to the scene scale (e.g. depth ≫ baseline,
        // as in a 2-view SfM seed) the system is badly conditioned and the
        // smallest right-singular vector is dominated by numerical noise — the
        // recovered projection is garbage even from an all-inlier sample. The
        // recovered projection `P'` acts on normalised points `p' = S·p`, so the
        // un-normalised projection is `P = P'·S`.
        let n = correspondences.len();
        let centroid = correspondences
            .iter()
            .fold(Vector3::zeros(), |acc, c| acc + c.point3d.coords)
            / n as f64;
        let mean_dist = correspondences
            .iter()
            .map(|c| (c.point3d.coords - centroid).norm())
            .sum::<f64>()
            / n as f64;
        let s = if mean_dist > f64::EPSILON {
            (3.0_f64).sqrt() / mean_dist
        } else {
            1.0
        };

        let mut a = DMatrix::<f64>::zeros(correspondences.len() * 2, 12);
        for (i, correspondence) in correspondences.iter().enumerate() {
            let normalized = camera.normalize_pixel(&correspondence.point2d)?;
            let x = normalized.x;
            let y = normalized.y;
            // Normalised world point p' = s·(p − centroid).
            let wx = s * (correspondence.point3d.x - centroid.x);
            let wy = s * (correspondence.point3d.y - centroid.y);
            let wz = s * (correspondence.point3d.z - centroid.z);
            let row = i * 2;

            a[(row, 0)] = -wx;
            a[(row, 1)] = -wy;
            a[(row, 2)] = -wz;
            a[(row, 3)] = -1.0;
            a[(row, 8)] = x * wx;
            a[(row, 9)] = x * wy;
            a[(row, 10)] = x * wz;
            a[(row, 11)] = x;

            a[(row + 1, 4)] = -wx;
            a[(row + 1, 5)] = -wy;
            a[(row + 1, 6)] = -wz;
            a[(row + 1, 7)] = -1.0;
            a[(row + 1, 8)] = y * wx;
            a[(row + 1, 9)] = y * wy;
            a[(row + 1, 10)] = y * wz;
            a[(row + 1, 11)] = y;
        }

        let svd = a.svd(true, true);
        let v_t = svd.v_t?;
        let last_row = v_t.row(v_t.nrows() - 1);

        let mut projection = [0.0_f64; 12];
        for i in 0..12 {
            projection[i] = last_row[i];
        }

        // Recovered projection acts on normalised points: P' (3×4). Compose with
        // the normalisation S (p' = s·p − s·centroid) to get P = P'·S so that M
        // and t below are for the original world frame.
        let m_prime = Matrix3::new(
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
        let t_prime = Vector3::new(projection[3], projection[7], projection[11]);
        // P = P'·S: M = s·M', t = t' − M'·(s·centroid).
        let mut m = m_prime * s;
        let mut translation = t_prime - m_prime * (s * centroid);

        // The DLT recovers `[M|t]` only up to an overall sign. Resolve it by
        // cheirality: the projective depth `(M·X + t).z` must be positive for
        // points in front of the camera. (The old determinant-only test fixed a
        // proper rotation but not the sign, so a valid-looking pose could place
        // every point behind the camera — reprojection ∞.)
        let depth_sign: f64 = correspondences
            .iter()
            .map(|c| ((m * c.point3d.coords).z + translation.z).signum())
            .sum();
        if depth_sign < 0.0 {
            m = -m;
            translation = -translation;
        }

        let row_scale = (m.row(0).norm() + m.row(1).norm() + m.row(2).norm()) / 3.0;
        if row_scale <= f64::EPSILON {
            return None;
        }
        m /= row_scale;
        translation /= row_scale;

        let rotation_svd = m.svd(true, true);
        let u = rotation_svd.u?;
        let v_t = rotation_svd.v_t?;
        let mut rotation = u * v_t;
        if rotation.determinant() < 0.0 {
            // Nearest *proper* rotation: flip the sign of the column tied to the
            // smallest singular value (Kabsch), not the whole matrix — negating
            // `R` would yield a different rotation and undo the cheirality sign.
            let mut u_fixed = u;
            let last = u_fixed.ncols() - 1;
            for r in 0..u_fixed.nrows() {
                u_fixed[(r, last)] = -u_fixed[(r, last)];
            }
            rotation = u_fixed * v_t;
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
                confidence: None,
            })
            .collect::<Vec<_>>();

        let estimated = DltPnP::default()
            .estimate_pose(&correspondences, &camera)
            .unwrap();
        assert!(estimated.world_to_camera.translation.norm() < 1.0e-6);
    }

    #[test]
    fn estimates_pose_for_points_far_from_origin() {
        // SfM-seed geometry: points sit far from the origin (depth ~14) relative
        // to the scene/baseline scale (~1). Without Hartley-normalising the 3D
        // points the DLT design matrix is badly conditioned and returns a
        // garbage projection; this regression-tests the normalisation.
        let camera = Camera::pinhole(1, 3072, 2304, 2560.0, 2560.0, 1536.0, 1152.0);
        let true_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.05, -0.08, 0.03)),
            Vector3::new(0.4, -0.3, 0.2),
        );
        let mut points = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                // Non-coplanar depth (a paraboloid, not a plane) so the DLT is
                // not degenerate — what makes this a *conditioning* test rather
                // than a planar-PnP test.
                let (xf, yf) = (xi as f64, yi as f64);
                let z = 13.0 + (xf * xf + yf * yf) * 0.6 + xf * 0.3;
                points.push(Point3::new(xf * 2.5, yf * 2.0, z));
            }
        }
        let correspondences = points
            .into_iter()
            .map(|point3d| Correspondence2D3D {
                point2d: camera
                    .project(&true_pose.transform_world_point(&point3d))
                    .unwrap(),
                point3d,
                confidence: None,
            })
            .collect::<Vec<_>>();

        let estimated = DltPnP::default()
            .estimate_pose(&correspondences, &camera)
            .expect("DLT should recover a pose for far-from-origin points");
        // Compare camera centres and rotations to the truth.
        let est_center = estimated.world_to_camera.inverse().translation;
        let true_center = true_pose.world_to_camera.inverse().translation;
        assert!(
            (est_center - true_center).norm() < 1.0e-3,
            "camera centre off: est {est_center:?} vs {true_center:?}"
        );
        let r_err = (estimated.world_to_camera.rotation
            * true_pose.world_to_camera.rotation.inverse())
        .angle();
        assert!(r_err < 1.0e-3, "rotation off by {r_err} rad");
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
                confidence: None,
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
