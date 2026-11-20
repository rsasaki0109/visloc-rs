//! Absolute pose **and focal length** from 2D-3D correspondences (P4Pf), for a
//! camera whose focal length is unknown.
//!
//! ```text
//! pixel + 3D point pairs -> centre at the principal point -> P4Pf hypotheses (4 points, pose + f)
//!                        -> pixel reprojection score (seeded RANSAC)
//!                        -> Levenberg-Marquardt refit of (R, t, f) on the consensus
//!                        -> pose, focal length, inliers
//! ```
//!
//! The camera is a pinhole with square pixels, no distortion and a known
//! principal point; the focal length is the unknown. Use it to register an
//! image whose focal length is not known well (no EXIF, a cropped or zoomed
//! photo): [`crate::pnp::P3PGrunert`] with a wrong focal-length prior sees a
//! consistently distorted bundle of rays, which shrinks the inlier set or
//! makes the registration fail.
//!
//! The four-point minimal solver ([`p4pf_minimal`]) is vendored unchanged from
//! [OpenMVG-rs](https://github.com/openMVG/openmvg-rs)
//! (`openmvg-geometry/src/pose/p4pf`, MIT OR Apache-2.0), where it is checked
//! bit for bit against a Julia reference: a DLT null basis turned into three
//! quadrics in three unknowns (E3Q3), solved through an 8x8 companion
//! eigenproblem, plus a planar seed for (near-)coplanar points, each candidate
//! polished by Gauss-Newton on the four reprojection residuals. The refit
//! follows OpenMVG-rs's `refit_focal_pose`; the RANSAC loop around it is this
//! crate's usual seeded one, so the result is deterministic for a given
//! `seed`.
//!
//! Poses are world-to-camera, as everywhere in visloc-rs:
//! `X_camera = R X_world + t`, and a point projects to
//! `f (X_camera.x, X_camera.y) / X_camera.z + principal_point`.

mod kernel;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, SMatrix, SVector, UnitQuaternion, Vector3};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use visloc_core::geometry::Pose;

use super::Correspondence2D3D;

/// Largest number of poses the minimal solver returns for one sample.
pub const P4PF_MAX_SOLUTIONS: usize = kernel::MAX_POSES;

const MINIMAL_SAMPLE: usize = 4;
const REFIT_ITERATIONS: usize = 20;
const MIN_REFIT_POINTS: usize = 6;

/// A pose together with the focal length (in pixels) it was estimated with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalPose {
    /// World-to-camera rotation.
    pub rotation: Matrix3<f64>,
    /// World-to-camera translation.
    pub translation: Vector3<f64>,
    /// Focal length in pixels (square pixels).
    pub focal_px: f64,
}

impl FocalPose {
    /// The pose as visloc-rs's [`Pose`] (the rotation is re-orthonormalised).
    pub fn pose(&self) -> Pose {
        let rotation =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(&self.rotation));
        Pose::from_world_to_camera(rotation, self.translation)
    }

    /// Pixel reprojection error of `point3d` against `pixel`; infinite when
    /// the point is not in front of the camera.
    pub fn reprojection_error(
        &self,
        principal_point: &Point2<f64>,
        point3d: &Point3<f64>,
        pixel: &Point2<f64>,
    ) -> f64 {
        let camera_point = self.rotation * point3d.coords + self.translation;
        if !(camera_point.z.is_finite() && camera_point.z > 0.0) {
            return f64::INFINITY;
        }
        let x = self.focal_px * camera_point.x / camera_point.z + principal_point.x;
        let y = self.focal_px * camera_point.y / camera_point.z + principal_point.y;
        (x - pixel.x).hypot(y - pixel.y)
    }

    fn is_valid(&self) -> bool {
        self.focal_px.is_finite()
            && self.focal_px > 0.0
            && self.translation.iter().all(|v| v.is_finite())
            && self.rotation.iter().all(|v| v.is_finite())
    }
}

/// Minimal P4Pf: the poses and focal lengths consistent with four 2D-3D
/// correspondences, best (lowest reprojection error) first. Empty when the
/// points are degenerate (repeated, collinear, non-finite) or no candidate
/// passes the solver's gates (positive focal length, points in front, small
/// reprojection error).
pub fn p4pf_minimal(
    points3d: &[Point3<f64>; 4],
    pixels: &[Point2<f64>; 4],
    principal_point: &Point2<f64>,
) -> Vec<FocalPose> {
    let mut out = Vec::new();
    for_each_p4pf_solution(points3d, pixels, principal_point, |pose| out.push(pose));
    out
}

fn for_each_p4pf_solution(
    points3d: &[Point3<f64>; 4],
    pixels: &[Point2<f64>; 4],
    principal_point: &Point2<f64>,
    mut emit: impl FnMut(FocalPose),
) {
    let world: [[f64; 3]; 4] =
        std::array::from_fn(|i| [points3d[i].x, points3d[i].y, points3d[i].z]);
    let image: [[f64; 2]; 4] = std::array::from_fn(|i| {
        [
            pixels[i].x - principal_point.x,
            pixels[i].y - principal_point.y,
        ]
    });
    let mut poses = [kernel::Pose::default(); kernel::MAX_POSES];
    let mut status = kernel::FRONT_END;
    let count = kernel::p4pf(&world, &image, &mut poses, &mut status);
    for pose in &poses[..count] {
        let candidate = FocalPose {
            rotation: Matrix3::from_row_slice(&pose.r),
            translation: Vector3::from(pose.t),
            focal_px: pose.f,
        };
        if candidate.is_valid() {
            emit(candidate);
        }
    }
}

/// Settings of [`estimate_pose_with_focal`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct P4PfRansacConfig {
    /// Inlier threshold on the pixel reprojection error.
    pub max_error_px: f64,
    /// Upper bound on the number of RANSAC samples.
    pub max_iterations: usize,
    /// Lower bound on the number of RANSAC samples.
    pub min_iterations: usize,
    /// Probability of drawing at least one all-inlier sample, for the adaptive
    /// stopping rule.
    pub confidence: f64,
    /// Fewest inliers an accepted estimate needs.
    pub min_inliers: usize,
    /// Plausible focal-length range in pixels; hypotheses outside it are
    /// discarded. `(0, inf)` accepts any positive focal length.
    pub focal_range_px: (f64, f64),
    /// RNG seed: the same seed gives the same result.
    pub seed: u64,
}

impl Default for P4PfRansacConfig {
    fn default() -> Self {
        Self {
            max_error_px: 4.0,
            max_iterations: 10_000,
            min_iterations: 100,
            confidence: 0.9999,
            min_inliers: 12,
            focal_range_px: (0.0, f64::INFINITY),
            seed: 7,
        }
    }
}

/// Result of [`estimate_pose_with_focal`].
#[derive(Debug, Clone, PartialEq)]
pub struct P4PfRansacReport {
    /// The refined pose and focal length.
    pub estimate: FocalPose,
    /// Indices of the correspondences within `max_error_px` of the estimate.
    pub inliers: Vec<usize>,
    /// RANSAC samples drawn.
    pub iterations: usize,
}

impl P4PfRansacReport {
    /// The pose as visloc-rs's [`Pose`].
    pub fn pose(&self) -> Pose {
        self.estimate.pose()
    }
}

/// Robust absolute pose and focal length from 2D-3D correspondences: seeded
/// RANSAC over [`p4pf_minimal`] samples, scored by pixel reprojection error
/// (MSAC: inlier count, ties broken by truncated squared error), then a
/// Levenberg-Marquardt refit of rotation, translation and focal length on the
/// consensus set, repeated until the inlier set stops changing.
///
/// `principal_point` is in the same pixel frame as the correspondences'
/// `point2d`; for an uncalibrated photo the image centre is the usual choice.
/// Returns `None` when there are fewer than four correspondences or no
/// estimate reaches `config.min_inliers`.
pub fn estimate_pose_with_focal(
    correspondences: &[Correspondence2D3D],
    principal_point: Point2<f64>,
    config: &P4PfRansacConfig,
) -> Option<P4PfRansacReport> {
    let n = correspondences.len();
    if n < MINIMAL_SAMPLE {
        return None;
    }
    let threshold = config.max_error_px;
    let threshold_sq = threshold * threshold;
    let in_range = |pose: &FocalPose| {
        pose.focal_px >= config.focal_range_px.0 && pose.focal_px <= config.focal_range_px.1
    };
    let score = |pose: &FocalPose| -> (usize, f64) {
        let mut count = 0;
        let mut cost = 0.0;
        for c in correspondences {
            let e = pose.reprojection_error(&principal_point, &c.point3d, &c.point2d);
            if e <= threshold {
                count += 1;
                cost += e * e;
            } else {
                cost += threshold_sq;
            }
        }
        (count, cost)
    };

    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut best: Option<(FocalPose, usize, f64)> = None;
    let mut required = config.max_iterations;
    let mut iterations = 0;
    while iterations
        < required
            .max(config.min_iterations)
            .min(config.max_iterations)
    {
        iterations += 1;
        let sample = rand::seq::index::sample(&mut rng, n, MINIMAL_SAMPLE);
        let points3d: [Point3<f64>; 4] =
            std::array::from_fn(|i| correspondences[sample.index(i)].point3d);
        let pixels: [Point2<f64>; 4] =
            std::array::from_fn(|i| correspondences[sample.index(i)].point2d);
        let mut improved = false;
        for_each_p4pf_solution(&points3d, &pixels, &principal_point, |candidate| {
            if !in_range(&candidate) {
                return;
            }
            let (count, cost) = score(&candidate);
            let better = match &best {
                None => true,
                Some((_, best_count, best_cost)) => {
                    count > *best_count || (count == *best_count && cost < *best_cost)
                }
            };
            if better {
                best = Some((candidate, count, cost));
                improved = true;
            }
        });
        if improved {
            let inlier_ratio = best.as_ref().map_or(0, |b| b.1) as f64 / n as f64;
            required = required_iterations(inlier_ratio, config.confidence, config.max_iterations);
        }
    }

    let (mut estimate, _, _) = best?;
    let mut inliers = inlier_indices(&estimate, correspondences, &principal_point, threshold);
    // Refit on the consensus until the inlier set settles (a few rounds at most).
    for _ in 0..4 {
        let Some(refit) = refit_focal_pose(&estimate, correspondences, &inliers, &principal_point)
        else {
            break;
        };
        if !in_range(&refit) {
            break;
        }
        let refit_inliers = inlier_indices(&refit, correspondences, &principal_point, threshold);
        if refit_inliers.len() < inliers.len() {
            break;
        }
        let settled = refit_inliers == inliers;
        estimate = refit;
        inliers = refit_inliers;
        if settled {
            break;
        }
    }
    (inliers.len() >= config.min_inliers.max(MINIMAL_SAMPLE)).then_some(P4PfRansacReport {
        estimate,
        inliers,
        iterations,
    })
}

fn required_iterations(inlier_ratio: f64, confidence: f64, max_iterations: usize) -> usize {
    let p_good = inlier_ratio.powi(MINIMAL_SAMPLE as i32);
    if p_good >= 1.0 {
        return 1;
    }
    if p_good <= 0.0 {
        return max_iterations;
    }
    let n = (1.0 - confidence).ln() / (1.0 - p_good).ln();
    if n.is_finite() {
        (n.ceil() as usize).min(max_iterations)
    } else {
        max_iterations
    }
}

fn inlier_indices(
    pose: &FocalPose,
    correspondences: &[Correspondence2D3D],
    principal_point: &Point2<f64>,
    threshold: f64,
) -> Vec<usize> {
    correspondences
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            pose.reprojection_error(principal_point, &c.point3d, &c.point2d) <= threshold
        })
        .map(|(i, _)| i)
        .collect()
}

/// Levenberg-Marquardt on the pixel reprojection error over `(R, t, f)`,
/// following OpenMVG-rs's `refit_focal_pose`: a left rotation increment, the
/// translation and the focal length, seven unknowns.
fn refit_focal_pose(
    seed: &FocalPose,
    correspondences: &[Correspondence2D3D],
    inliers: &[usize],
    principal_point: &Point2<f64>,
) -> Option<FocalPose> {
    type Normal = SMatrix<f64, 7, 7>;
    type Gradient = SVector<f64, 7>;

    if inliers.len() < MIN_REFIT_POINTS {
        return None;
    }
    let cost = |model: &FocalPose| -> f64 {
        inliers
            .iter()
            .map(|&i| {
                let c = &correspondences[i];
                let e = model.reprojection_error(principal_point, &c.point3d, &c.point2d);
                e * e
            })
            .sum()
    };
    let mut model = *seed;
    let mut model_cost = cost(&model);
    if !model_cost.is_finite() {
        return None;
    }
    let mut damping = 1e-3;
    for _ in 0..REFIT_ITERATIONS {
        let mut normal = Normal::zeros();
        let mut gradient = Gradient::zeros();
        for &i in inliers {
            let c = &correspondences[i];
            let rotated = model.rotation * c.point3d.coords;
            let camera_point = rotated + model.translation;
            if !(camera_point.z.is_finite() && camera_point.z > 0.0) {
                return None;
            }
            let inverse_depth = 1.0 / camera_point.z;
            let nx = camera_point.x * inverse_depth;
            let ny = camera_point.y * inverse_depth;
            let residual = [
                model.focal_px * nx + principal_point.x - c.point2d.x,
                model.focal_px * ny + principal_point.y - c.point2d.y,
            ];
            // d(pixel)/d(camera point), chained through a left rotation
            // increment w (d(camera point) = -[R X]x w), the translation and f.
            let skew = Matrix3::new(
                0.0, -rotated.z, rotated.y, rotated.z, 0.0, -rotated.x, -rotated.y, rotated.x, 0.0,
            );
            let d_point = [
                Vector3::new(
                    model.focal_px * inverse_depth,
                    0.0,
                    -model.focal_px * nx * inverse_depth,
                ),
                Vector3::new(
                    0.0,
                    model.focal_px * inverse_depth,
                    -model.focal_px * ny * inverse_depth,
                ),
            ];
            for (row, r) in residual.iter().enumerate() {
                let d_rotation = -(skew.transpose() * d_point[row]);
                let mut jacobian = Gradient::zeros();
                jacobian.fixed_rows_mut::<3>(0).copy_from(&d_rotation);
                jacobian.fixed_rows_mut::<3>(3).copy_from(&d_point[row]);
                jacobian[6] = if row == 0 { nx } else { ny };
                normal += jacobian * jacobian.transpose();
                gradient += jacobian * *r;
            }
        }
        let mut accepted = false;
        for _ in 0..8 {
            let mut damped = normal;
            for k in 0..7 {
                damped[(k, k)] += damping * normal[(k, k)].max(1e-12);
            }
            let Some(step) = damped.cholesky().map(|f| f.solve(&(-gradient))) else {
                damping *= 10.0;
                continue;
            };
            let rotation_step =
                Rotation3::from_scaled_axis(Vector3::new(step[0], step[1], step[2]));
            let candidate = FocalPose {
                rotation: *rotation_step.matrix() * model.rotation,
                translation: model.translation + Vector3::new(step[3], step[4], step[5]),
                focal_px: model.focal_px + step[6],
            };
            let candidate_cost = cost(&candidate);
            if candidate_cost.is_finite() && candidate_cost < model_cost && candidate.focal_px > 0.0
            {
                let converged = (model_cost - candidate_cost) <= 1e-12 * model_cost.max(1e-30);
                model = candidate;
                model_cost = candidate_cost;
                damping = (damping * 0.3).max(1e-9);
                accepted = true;
                if converged {
                    return model.is_valid().then_some(model);
                }
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            break;
        }
    }
    model.is_valid().then_some(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn rotation(rx: f64, ry: f64, rz: f64) -> Matrix3<f64> {
        *Rotation3::from_euler_angles(rx, ry, rz).matrix()
    }

    fn project(pose: &FocalPose, pp: &Point2<f64>, p: &Point3<f64>) -> Point2<f64> {
        let c = pose.rotation * p.coords + pose.translation;
        Point2::new(
            pose.focal_px * c.x / c.z + pp.x,
            pose.focal_px * c.y / c.z + pp.y,
        )
    }

    fn scene(rng: &mut SmallRng, n: usize) -> Vec<Point3<f64>> {
        (0..n)
            .map(|_| {
                Point3::new(
                    rng.gen_range(-2.0..2.0),
                    rng.gen_range(-1.5..1.5),
                    rng.gen_range(-1.0..1.0),
                )
            })
            .collect()
    }

    fn truth() -> (FocalPose, Point2<f64>) {
        (
            FocalPose {
                rotation: rotation(0.1, -0.25, 0.05),
                translation: Vector3::new(0.3, -0.2, 6.0),
                focal_px: 830.0,
            },
            Point2::new(640.0, 360.0),
        )
    }

    #[test]
    fn minimal_solver_recovers_pose_and_focal() {
        let (gt, pp) = truth();
        let mut rng = SmallRng::seed_from_u64(1);
        let mut checked = 0;
        for _ in 0..50 {
            let pts = scene(&mut rng, 4);
            let points3d: [Point3<f64>; 4] = std::array::from_fn(|i| pts[i]);
            let pixels: [Point2<f64>; 4] = std::array::from_fn(|i| project(&gt, &pp, &pts[i]));
            let solutions = p4pf_minimal(&points3d, &pixels, &pp);
            let found = solutions.iter().any(|s| {
                (s.focal_px - gt.focal_px).abs() < 1e-6 * gt.focal_px
                    && (s.rotation - gt.rotation).norm() < 1e-6
                    && (s.translation - gt.translation).norm() < 1e-6
            });
            assert!(
                found,
                "ground truth not among {} solutions",
                solutions.len()
            );
            checked += 1;
        }
        assert_eq!(checked, 50);
    }

    #[test]
    fn minimal_solver_handles_planar_points() {
        let (gt, pp) = truth();
        let points3d = [
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.2, -0.8, 0.0),
            Point3::new(0.9, 1.1, 0.0),
            Point3::new(-1.1, 0.7, 0.0),
        ];
        let pixels: [Point2<f64>; 4] = std::array::from_fn(|i| project(&gt, &pp, &points3d[i]));
        let solutions = p4pf_minimal(&points3d, &pixels, &pp);
        assert!(solutions
            .iter()
            .any(|s| (s.focal_px - gt.focal_px).abs() < 1e-4 * gt.focal_px));
    }

    #[test]
    fn minimal_solver_declines_collinear_points() {
        let (gt, pp) = truth();
        let points3d: [Point3<f64>; 4] =
            std::array::from_fn(|i| Point3::new(i as f64 * 0.5, i as f64 * 0.25, 0.1 * i as f64));
        let pixels: [Point2<f64>; 4] = std::array::from_fn(|i| project(&gt, &pp, &points3d[i]));
        assert!(p4pf_minimal(&points3d, &pixels, &pp).is_empty());
    }

    #[test]
    fn ransac_recovers_focal_with_noise_and_outliers() {
        let (gt, pp) = truth();
        let mut rng = SmallRng::seed_from_u64(42);
        let pts = scene(&mut rng, 200);
        let correspondences: Vec<Correspondence2D3D> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut px = project(&gt, &pp, p);
                if i % 10 < 4 {
                    // 40% outliers anywhere in a 1280x720 image
                    px = Point2::new(rng.gen_range(0.0..1280.0), rng.gen_range(0.0..720.0));
                } else {
                    px.x += rng.gen_range(-0.5..0.5);
                    px.y += rng.gen_range(-0.5..0.5);
                }
                Correspondence2D3D {
                    point2d: px,
                    point3d: *p,
                    confidence: None,
                }
            })
            .collect();
        let report = estimate_pose_with_focal(&correspondences, pp, &P4PfRansacConfig::default())
            .expect("P4Pf RANSAC should succeed");
        let est = report.estimate;
        assert!(
            (est.focal_px - gt.focal_px).abs() < 0.01 * gt.focal_px,
            "focal {} vs {}",
            est.focal_px,
            gt.focal_px
        );
        assert!((est.rotation - gt.rotation).norm() < 5e-3);
        assert!((est.translation - gt.translation).norm() < 0.05);
        // All 120 inliers (and almost no outlier) are kept.
        let true_inliers = report.inliers.iter().filter(|&&i| i % 10 >= 4).count();
        assert!(true_inliers >= 118, "kept {true_inliers} of 120 inliers");
        assert!(report.inliers.len() - true_inliers <= 3);

        // Same seed, same answer.
        let again =
            estimate_pose_with_focal(&correspondences, pp, &P4PfRansacConfig::default()).unwrap();
        assert_eq!(again, report);
    }

    #[test]
    fn ransac_respects_focal_range() {
        let (gt, pp) = truth();
        let mut rng = SmallRng::seed_from_u64(5);
        let correspondences: Vec<Correspondence2D3D> = scene(&mut rng, 60)
            .iter()
            .map(|p| Correspondence2D3D {
                point2d: project(&gt, &pp, p),
                point3d: *p,
                confidence: None,
            })
            .collect();
        let config = P4PfRansacConfig {
            focal_range_px: (1500.0, 3000.0),
            ..P4PfRansacConfig::default()
        };
        // Focal length and depth trade off on a shallow scene, so a hypothesis
        // inside the range can still gather inliers; whatever is returned
        // must respect the range.
        if let Some(report) = estimate_pose_with_focal(&correspondences, pp, &config) {
            let f = report.estimate.focal_px;
            assert!((1500.0..=3000.0).contains(&f), "focal {f} outside the range");
            assert!(report.inliers.len() < correspondences.len());
        }
    }
}
