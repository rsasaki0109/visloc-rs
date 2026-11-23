//! Runtime-selectable minimal PnP solver.
//!
//! [`PnPRansac`](crate::ransac::PnPRansac) is generic over its minimal
//! [`PoseEstimator`], so picking DLT or P3P normally means picking a type at
//! compile time. [`PnpMinimalSolver`] wraps both behind one value, so a
//! configuration struct or a command-line flag can choose the solver at run
//! time without changing the pipeline's type.
//!
//! Choose [`PnpMinimalSolver::P3p`] for scenes dominated by walls, façades or
//! other planar structure: the 6-point DLT is degenerate when the sampled map
//! points are coplanar (a view of a single wall), while P3P is well-posed for
//! any three non-collinear points. [`PnpMinimalSolver::Dlt`] stays the default
//! so existing results do not change.

use std::fmt;
use std::str::FromStr;

use visloc_core::geometry::Pose;
use visloc_core::types::Camera;

use super::{Correspondence2D3D, DltPnP, P3PGrunert, PoseEstimator};

/// Minimal PnP solver chosen at run time; see the [module docs](self).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PnpMinimalSolver {
    /// 6-point Direct Linear Transform ([`DltPnP`]). Linear and fast, but
    /// degenerate on coplanar points. The default.
    #[default]
    Dlt,
    /// Grunert's Perspective-Three-Point solver ([`P3PGrunert`]), 4-point
    /// samples. Well-posed on planar scenes; use it where the map is mostly
    /// walls.
    P3p,
}

impl PnpMinimalSolver {
    /// The solver's name as [`FromStr`] accepts it (`"dlt"` or `"p3p"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dlt => "dlt",
            Self::P3p => "p3p",
        }
    }
}

impl PoseEstimator for PnpMinimalSolver {
    fn estimate_pose(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<Pose> {
        match self {
            Self::Dlt => DltPnP::default().estimate_pose(correspondences, camera),
            Self::P3p => P3PGrunert.estimate_pose(correspondences, camera),
        }
    }

    fn minimum_correspondences(&self) -> usize {
        match self {
            Self::Dlt => DltPnP::default().minimum_correspondences(),
            Self::P3p => P3PGrunert.minimum_correspondences(),
        }
    }
}

impl fmt::Display for PnpMinimalSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PnpMinimalSolver {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.to_ascii_lowercase().as_str() {
            "dlt" => Ok(Self::Dlt),
            "p3p" => Ok(Self::P3p),
            other => Err(format!("unknown PnP solver {other} (dlt or p3p)")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    fn wall_correspondences(camera: &Camera, pose: &Pose) -> Vec<Correspondence2D3D> {
        // Every point on the plane z = 8: a single wall facing the camera.
        let mut corrs = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                let point3d = Point3::new(xi as f64 * 0.8, yi as f64 * 0.8, 8.0);
                corrs.push(Correspondence2D3D {
                    point2d: camera
                        .project(&pose.transform_world_point(&point3d))
                        .unwrap(),
                    point3d,
                    confidence: None,
                });
            }
        }
        corrs
    }

    #[test]
    fn default_is_dlt_and_names_round_trip() {
        assert_eq!(PnpMinimalSolver::default(), PnpMinimalSolver::Dlt);
        for solver in [PnpMinimalSolver::Dlt, PnpMinimalSolver::P3p] {
            assert_eq!(solver.to_string().parse::<PnpMinimalSolver>(), Ok(solver));
        }
        assert_eq!("P3P".parse::<PnpMinimalSolver>(), Ok(PnpMinimalSolver::P3p));
        assert!("epnp".parse::<PnpMinimalSolver>().is_err());
    }

    #[test]
    fn minimum_correspondences_follow_the_solver() {
        assert_eq!(PnpMinimalSolver::Dlt.minimum_correspondences(), 6);
        assert_eq!(PnpMinimalSolver::P3p.minimum_correspondences(), 4);
    }

    #[test]
    fn p3p_recovers_a_wall_view() {
        let camera = Camera::pinhole(1, 1024, 768, 900.0, 900.0, 512.0, 384.0);
        let true_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.07, 0.12, -0.03)),
            Vector3::new(-0.4, 0.25, 0.1),
        );
        let corrs = wall_correspondences(&camera, &true_pose);
        let est = PnpMinimalSolver::P3p
            .estimate_pose(&corrs, &camera)
            .expect("P3P solves a planar view");
        let ec = est.world_to_camera.inverse().translation;
        let tc = true_pose.world_to_camera.inverse().translation;
        assert!((ec - tc).norm() < 1.0e-5, "centre off: {ec:?} vs {tc:?}");
    }
}
