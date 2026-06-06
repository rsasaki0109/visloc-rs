//! Grunert's Perspective-Three-Point (P3P) minimal solver.
//!
//! The [`DltPnP`](super::DltPnP) 6-point solver is a *linear* method, and like
//! every linear PnP it is **degenerate on coplanar points**: a flat building
//! façade, a planar calibration target, or any near-planar patch collapses the
//! DLT design matrix and returns a garbage pose. COLMAP and every serious SfM
//! frontend therefore use a *minimal* geometric solver — P3P — inside RANSAC,
//! because P3P is well-posed for any three non-collinear points whether or not
//! they (and the rest of the scene) lie on a plane.
//!
//! This is the classic Grunert formulation as reviewed by Haralick et al.
//! (1994), "Review and Analysis of Solutions of the Three Point Perspective
//! Pose Estimation Problem". From three world points and their three image
//! bearings it forms a quartic in one length ratio, solves it, recovers the
//! three camera-frame point depths for each real root, and reads off the rigid
//! pose by absolute orientation (Kabsch) on the three 3D-3D pairs. A P3P sample
//! yields up to four candidate poses; the extra correspondence(s) in the RANSAC
//! sample disambiguate by reprojection.

use nalgebra::{Matrix3, Matrix4, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;

use super::{Correspondence2D3D, PoseEstimator};

/// Grunert's P3P solver, usable anywhere a [`PoseEstimator`] is expected (e.g.
/// as the minimal estimator inside [`PnPRansac`](crate::ransac::PnPRansac)).
///
/// `minimum_correspondences()` is 4 — three points define the P3P geometry and
/// the fourth disambiguates the up-to-four solutions. When handed more than
/// four (the RANSAC refit on the full inlier set) it solves from a well-spread
/// triple and scores every candidate against all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct P3PGrunert;

impl Default for P3PGrunert {
    fn default() -> Self {
        Self
    }
}

impl PoseEstimator for P3PGrunert {
    fn estimate_pose(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<Pose> {
        if correspondences.len() < self.minimum_correspondences() {
            return None;
        }

        // Pick three world points that span a well-conditioned triangle. For
        // the minimal RANSAC sample this is just the three points; for the
        // refit-on-all-inliers call it avoids a near-collinear leading triple.
        let (i0, i1, i2) = spread_triple(correspondences)?;

        // Unit bearing vectors in the camera frame for the solve triple.
        let f0 = bearing(camera, &correspondences[i0])?;
        let f1 = bearing(camera, &correspondences[i1])?;
        let f2 = bearing(camera, &correspondences[i2])?;

        let p0 = correspondences[i0].point3d;
        let p1 = correspondences[i1].point3d;
        let p2 = correspondences[i2].point3d;

        let candidates = solve_grunert(&[p0, p1, p2], &[f0, f1, f2])?;

        // Disambiguate by total reprojection error over the whole sample.
        let mut best: Option<Pose> = None;
        let mut best_error = f64::INFINITY;
        for pose in candidates {
            let mut error = 0.0;
            let mut valid = true;
            for c in correspondences {
                let camera_point = pose.transform_world_point(&c.point3d);
                let Some(projected) = camera.project(&camera_point) else {
                    valid = false;
                    break;
                };
                error += (projected - c.point2d).norm_squared();
            }
            if valid && error < best_error {
                best_error = error;
                best = Some(pose);
            }
        }
        best
    }

    fn minimum_correspondences(&self) -> usize {
        4
    }
}

/// Unit bearing vector (camera frame) for a correspondence's image point.
fn bearing(camera: &Camera, c: &Correspondence2D3D) -> Option<Vector3<f64>> {
    let n = camera.normalize_pixel(&c.point2d)?;
    let v = Vector3::new(n.x, n.y, 1.0);
    let norm = v.norm();
    if norm <= f64::EPSILON {
        return None;
    }
    Some(v / norm)
}

/// Choose three indices forming a well-spread (non-collinear) triangle. With
/// exactly three correspondences it returns them directly; otherwise it greedily
/// picks the farthest-apart pair-plus-apex so the refit on a large inlier set
/// never solves P3P from a near-degenerate triple.
fn spread_triple(correspondences: &[Correspondence2D3D]) -> Option<(usize, usize, usize)> {
    let n = correspondences.len();
    if n < 3 {
        return None;
    }
    if n == 3 {
        return Some((0, 1, 2));
    }
    let pts: Vec<Vector3<f64>> = correspondences.iter().map(|c| c.point3d.coords).collect();

    // Farthest-apart pair (i0, i1).
    let (mut i0, mut i1, mut best_d2) = (0usize, 1usize, -1.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let d2 = (pts[i] - pts[j]).norm_squared();
            if d2 > best_d2 {
                best_d2 = d2;
                i0 = i;
                i1 = j;
            }
        }
    }
    if best_d2 <= f64::EPSILON {
        return None;
    }

    // Apex i2 maximising distance to the line through (i0, i1).
    let line = pts[i1] - pts[i0];
    let line_norm = line.norm();
    let (mut i2, mut best_h2) = (usize::MAX, -1.0);
    for k in 0..n {
        if k == i0 || k == i1 {
            continue;
        }
        let h2 = (pts[k] - pts[i0]).cross(&line).norm_squared() / (line_norm * line_norm);
        if h2 > best_h2 {
            best_h2 = h2;
            i2 = k;
        }
    }
    if i2 == usize::MAX || best_h2 <= f64::EPSILON {
        return None;
    }
    Some((i0, i1, i2))
}

/// Solve Grunert's P3P for the camera-frame depths and return the candidate
/// poses (world→camera) recovered by absolute orientation. Up to four.
fn solve_grunert(points: &[Point3<f64>; 3], rays: &[Vector3<f64>; 3]) -> Option<Vec<Pose>> {
    let p1 = points[0].coords;
    let p2 = points[1].coords;
    let p3 = points[2].coords;

    // Pairwise world distances. (a = |P2−P3|, b = |P1−P3|, c = |P1−P2|.)
    let a = (p2 - p3).norm();
    let b = (p1 - p3).norm();
    let c = (p1 - p2).norm();
    if a <= f64::EPSILON || b <= f64::EPSILON || c <= f64::EPSILON {
        return None;
    }

    // Cosines of the angles subtended at the camera centre between the bearings.
    let cos_alpha = rays[1].dot(&rays[2]); // between f2, f3
    let cos_beta = rays[0].dot(&rays[2]); // between f1, f3
    let cos_gamma = rays[0].dot(&rays[1]); // between f1, f2

    let a_sq = a * a;
    let b_sq = b * b;
    let c_sq = c * c;

    // Haralick's normalised ratios.
    let p = (a_sq - c_sq) / b_sq; // "(a²−c²)/b²"
    let q = (a_sq + c_sq) / b_sq; // "(a²+c²)/b²"

    let ca = cos_alpha;
    let cb = cos_beta;
    let cg = cos_gamma;

    // Quartic A4 v⁴ + A3 v³ + A2 v² + A1 v + A0 = 0 (Grunert).
    let a4 = (p - 1.0) * (p - 1.0) - 4.0 * c_sq / b_sq * ca * ca;
    let a3 = 4.0 * (p * (1.0 - p) * cb - (1.0 - q) * ca * cg + 2.0 * c_sq / b_sq * ca * ca * cb);
    let a2 = 2.0
        * (p * p - 1.0 + 2.0 * p * p * cb * cb + 2.0 * (b_sq - c_sq) / b_sq * ca * ca
            - 4.0 * q * ca * cb * cg
            + 2.0 * (b_sq - a_sq) / b_sq * cg * cg);
    let a1 = 4.0 * (-p * (1.0 + p) * cb + 2.0 * a_sq / b_sq * cg * cg * cb - (1.0 - q) * ca * cg);
    let a0 = (1.0 + p) * (1.0 + p) - 4.0 * a_sq / b_sq * cg * cg;

    let roots = real_quartic_roots(a4, a3, a2, a1, a0)?;

    let mut poses = Vec::new();
    for v in roots {
        if v <= 0.0 {
            continue;
        }
        // u from the eliminated quadratic, given v.
        let denom = 2.0 * (cg - v * ca);
        if denom.abs() <= f64::EPSILON {
            continue;
        }
        let u = ((p - 1.0) * v * v - 2.0 * p * cb * v + 1.0 + p) / denom;
        if u <= 0.0 {
            continue;
        }

        // s1² from 1 + v² − 2v·cosβ = b²/s1².
        let denom_s1 = 1.0 + v * v - 2.0 * v * cb;
        if denom_s1 <= f64::EPSILON {
            continue;
        }
        let s1 = (b_sq / denom_s1).sqrt();
        let s2 = u * s1;
        let s3 = v * s1;
        if !(s1.is_finite() && s2.is_finite() && s3.is_finite()) {
            continue;
        }

        // Camera-frame reconstructions of the three points.
        let q1 = (rays[0] * s1).into();
        let q2 = (rays[1] * s2).into();
        let q3 = (rays[2] * s3).into();

        if let Some(pose) = kabsch_three(&[points[0], points[1], points[2]], &[q1, q2, q3]) {
            poses.push(pose);
        }
    }

    if poses.is_empty() {
        None
    } else {
        Some(poses)
    }
}

/// Real roots of `a4 x⁴ + a3 x³ + a2 x² + a1 x + a0` via the companion-matrix
/// eigenvalues of the monic polynomial. Returns `None` if the quartic
/// degenerates (leading coefficient ~0) or no eigenvalue is real.
fn real_quartic_roots(a4: f64, a3: f64, a2: f64, a1: f64, a0: f64) -> Option<Vec<f64>> {
    if a4.abs() <= f64::EPSILON {
        return None;
    }
    let (c3, c2, c1, c0) = (a3 / a4, a2 / a4, a1 / a4, a0 / a4);
    // Companion matrix of x⁴ + c3 x³ + c2 x² + c1 x + c0.
    let companion = Matrix4::new(
        -c3, -c2, -c1, -c0, //
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    );
    let eigen = companion.complex_eigenvalues();
    let mut roots = Vec::new();
    for e in eigen.iter() {
        if e.im.abs() < 1.0e-9 * (1.0 + e.re.abs()) {
            roots.push(e.re);
        }
    }
    if roots.is_empty() {
        None
    } else {
        Some(roots)
    }
}

/// Closed-form Kabsch on exactly three 3D-3D pairs: returns the world→camera
/// pose `Q ≈ R·P + t`. Degenerate (collinear) triples fall out as a non-proper
/// rotation, which the reprojection disambiguation then rejects.
fn kabsch_three(world: &[Point3<f64>; 3], camera: &[Point3<f64>; 3]) -> Option<Pose> {
    let cw = (world[0].coords + world[1].coords + world[2].coords) / 3.0;
    let cc = (camera[0].coords + camera[1].coords + camera[2].coords) / 3.0;
    let mut h = Matrix3::<f64>::zeros();
    for i in 0..3 {
        let w = world[i].coords - cw;
        let q = camera[i].coords - cc;
        h += w * q.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let v = v_t.transpose();
    let mut s = Matrix3::<f64>::identity();
    if (v * u.transpose()).determinant() < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let r_mat = v * s * u.transpose();
    if !r_mat.iter().all(|x| x.is_finite()) {
        return None;
    }
    let translation = cc - r_mat * cw;
    let rotation = UnitQuaternion::from_matrix(&r_mat);
    Some(Pose::from_world_to_camera(rotation, translation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    fn make_corr(camera: &Camera, pose: &Pose, point3d: Point3<f64>) -> Correspondence2D3D {
        Correspondence2D3D {
            point2d: camera
                .project(&pose.transform_world_point(&point3d))
                .unwrap(),
            point3d,
            confidence: None,
        }
    }

    #[test]
    fn recovers_pose_from_general_points() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let true_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.15, 0.05)),
            Vector3::new(0.3, -0.2, 0.4),
        );
        let points = vec![
            Point3::new(-1.0, -1.0, 5.0),
            Point3::new(1.2, -0.8, 6.0),
            Point3::new(-0.5, 1.1, 5.5),
            Point3::new(0.8, 0.9, 7.0),
            Point3::new(0.0, 0.0, 6.5),
        ];
        let corrs: Vec<_> = points
            .into_iter()
            .map(|p| make_corr(&camera, &true_pose, p))
            .collect();

        let est = P3PGrunert
            .estimate_pose(&corrs, &camera)
            .expect("P3P should recover a pose");
        let ec = est.world_to_camera.inverse().translation;
        let tc = true_pose.world_to_camera.inverse().translation;
        assert!((ec - tc).norm() < 1.0e-6, "centre off: {ec:?} vs {tc:?}");
        let r_err =
            (est.world_to_camera.rotation * true_pose.world_to_camera.rotation.inverse()).angle();
        assert!(r_err < 1.0e-6, "rotation off by {r_err}");
    }

    #[test]
    fn recovers_pose_from_coplanar_points() {
        // The whole point of P3P: a planar façade where the DLT is degenerate.
        let camera = Camera::pinhole(1, 1024, 768, 900.0, 900.0, 512.0, 384.0);
        let true_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.07, 0.12, -0.03)),
            Vector3::new(-0.4, 0.25, 0.1),
        );
        // All points on the plane z = 8 (a flat wall facing the camera).
        let mut corrs = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                let p = Point3::new(xi as f64 * 0.8, yi as f64 * 0.8, 8.0);
                corrs.push(make_corr(&camera, &true_pose, p));
            }
        }
        let est = P3PGrunert
            .estimate_pose(&corrs, &camera)
            .expect("P3P should recover a pose on coplanar points");
        let ec = est.world_to_camera.inverse().translation;
        let tc = true_pose.world_to_camera.inverse().translation;
        assert!(
            (ec - tc).norm() < 1.0e-5,
            "coplanar centre off: {ec:?} vs {tc:?}"
        );
        let r_err =
            (est.world_to_camera.rotation * true_pose.world_to_camera.rotation.inverse()).angle();
        assert!(r_err < 1.0e-5, "coplanar rotation off by {r_err}");
    }

    #[test]
    fn rejects_collinear_triple() {
        // Three points on a line is the genuine P3P degeneracy; with only three
        // collinear correspondences there is no valid triangle.
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let corrs: Vec<_> = (0..3)
            .map(|i| make_corr(&camera, &pose, Point3::new(i as f64, i as f64, 5.0)))
            .collect();
        // minimum_correspondences is 4, so three is below the floor anyway.
        assert!(P3PGrunert.estimate_pose(&corrs, &camera).is_none());
    }

    #[test]
    fn far_from_origin_seed_geometry() {
        // The conditioning case that broke the DLT: depth ≫ baseline.
        let camera = Camera::pinhole(1, 3072, 2304, 2560.0, 2560.0, 1536.0, 1152.0);
        let true_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.05, -0.08, 0.03)),
            Vector3::new(0.4, -0.3, 0.2),
        );
        let mut corrs = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                let (xf, yf) = (xi as f64, yi as f64);
                let z = 13.0 + (xf * xf + yf * yf) * 0.6 + xf * 0.3;
                corrs.push(make_corr(
                    &camera,
                    &true_pose,
                    Point3::new(xf * 2.5, yf * 2.0, z),
                ));
            }
        }
        let est = P3PGrunert.estimate_pose(&corrs, &camera).unwrap();
        let ec = est.world_to_camera.inverse().translation;
        let tc = true_pose.world_to_camera.inverse().translation;
        assert!((ec - tc).norm() < 1.0e-4, "centre off: {ec:?} vs {tc:?}");
    }
}
