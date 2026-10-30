//! Metric 3D objects from object tracks and metric camera poses (Phase 3 of
//! `docs/object_metric_slam_plan.md`, static objects).
//!
//! The monocular-inertial VIO's poses are metric thanks to the IMU. So feature
//! tracks triangulated with them are metric too, and so is an upright box
//! fitted to those points. The pipeline:
//!
//! 1. Collect the feature tracks seen inside an object's 2D boxes.
//! 2. Triangulate each one from its observations ([`triangulate`]): camera
//!    poses (camera to world, from the VIO) and unit bearings (from the camera
//!    model, which this module does not need to know).
//! 3. Fit a gravity-aligned box to the points ([`fit_upright_box`]). The world
//!    z axis is up, as in Basalt's gravity-aligned world frame.
//!
//! Every estimate records where its scale came from ([`ScaleSource`]). Class
//! size priors ([`plausible_size`]) are used only to flag implausible
//! results, never to set a scale: with a wrong scale, the sizes come out
//! wrong.

use nalgebra::{Isometry3, Matrix3, Point3, Vector2, Vector3};

/// One observation of a feature: the camera pose (camera to world) and the
/// unit bearing of the feature in the camera frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BearingObservation {
    pub camera_to_world: Isometry3<f64>,
    pub bearing: Vector3<f64>,
}

/// Acceptance tests of [`triangulate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangulationParams {
    /// Observations a track needs (3).
    pub min_observations: usize,
    /// Largest angle between two of its world rays must exceed this (2°).
    pub min_parallax_rad: f64,
    /// Every ray must pass within this angle of the point (0.3°).
    pub max_ray_error_rad: f64,
    /// The point must be at least this far in front of every camera (0.1 m).
    pub min_depth: f64,
}

impl Default for TriangulationParams {
    fn default() -> Self {
        Self {
            min_observations: 3,
            min_parallax_rad: 2f64.to_radians(),
            max_ray_error_rad: 0.3f64.to_radians(),
            min_depth: 0.1,
        }
    }
}

/// The point closest to all the observation rays (least squares), if the
/// rays diverge enough and the point is in front of, and close to, every ray.
pub fn triangulate(
    observations: &[BearingObservation],
    params: &TriangulationParams,
) -> Option<Point3<f64>> {
    if observations.len() < params.min_observations.max(2) {
        return None;
    }
    let rays: Vec<(Vector3<f64>, Vector3<f64>)> = observations
        .iter()
        .map(|o| {
            let direction = (o.camera_to_world.rotation * o.bearing).try_normalize(1e-12)?;
            Some((o.camera_to_world.translation.vector, direction))
        })
        .collect::<Option<_>>()?;
    let parallax = rays
        .iter()
        .enumerate()
        .flat_map(|(i, a)| rays[i + 1..].iter().map(move |b| a.1.angle(&b.1)))
        .fold(0.0, f64::max);
    if parallax < params.min_parallax_rad {
        return None;
    }
    let mut a = Matrix3::zeros();
    let mut b = Vector3::zeros();
    for (center, direction) in &rays {
        let projector = Matrix3::identity() - direction * direction.transpose();
        a += projector;
        b += projector * center;
    }
    let point = a.cholesky()?.solve(&b);
    for (center, direction) in &rays {
        let offset = point - center;
        if offset.dot(direction) < params.min_depth
            || offset.angle(direction) > params.max_ray_error_rad
        {
            return None;
        }
    }
    Some(Point3::from(point))
}

/// A gravity-aligned box. `length` (the longer side) runs along the heading
/// `yaw` (radians from world x towards world y), `width` across it, `height`
/// along world z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UprightBox {
    pub center: Point3<f64>,
    pub length: f64,
    pub width: f64,
    pub height: f64,
    pub yaw: f64,
}

/// Robustness settings of [`fit_upright_box`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxFitParams {
    /// Points needed after outlier removal (6).
    pub min_points: usize,
    /// Points farther from the median point than this many scaled median
    /// absolute deviations are outliers, such as background seen through or
    /// beside the object (3).
    pub outlier_mads: f64,
    /// Extents are taken between this fraction and one minus it of the
    /// points along each axis (0.02), so a few stray points do not inflate
    /// the box.
    pub extent_quantile: f64,
}

impl Default for BoxFitParams {
    fn default() -> Self {
        Self {
            min_points: 6,
            outlier_mads: 3.0,
            extent_quantile: 0.02,
        }
    }
}

/// Fit an upright box to `points` (world z up): drop outliers far from the
/// median point, then take the heading whose quantile rectangle has the
/// smallest area, and the quantile extents along it, across it and up. Points on an object's
/// visible surfaces only cover part of it, so the box can be smaller than the
/// object.
pub fn fit_upright_box(points: &[Point3<f64>], params: &BoxFitParams) -> Option<UprightBox> {
    let median_point = Point3::new(
        median(points.iter().map(|p| p.x).collect())?,
        median(points.iter().map(|p| p.y).collect())?,
        median(points.iter().map(|p| p.z).collect())?,
    );
    let distances: Vec<f64> = points.iter().map(|p| (p - median_point).norm()).collect();
    let spread = 1.4826 * median(distances.clone())?;
    let inliers: Vec<Point3<f64>> = points
        .iter()
        .zip(&distances)
        .filter(|(_, d)| spread == 0.0 || **d <= params.outlier_mads * spread)
        .map(|(p, _)| *p)
        .collect();
    if inliers.len() < params.min_points {
        return None;
    }
    // Heading: the yaw whose rectangle of quantile extents has the smallest
    // area, searched in 0.5° steps over a quarter turn (the main axis of the
    // scatter is unstable for near-square footprints).
    let q = params.extent_quantile;
    let extents = |axis: Vector2<f64>| -> Option<(f64, f64)> {
        let values: Vec<f64> = inliers
            .iter()
            .map(|p| axis.dot(&Vector2::new(p.x, p.y)))
            .collect();
        Some((quantile(values.clone(), q)?, quantile(values, 1.0 - q)?))
    };
    let axes = |yaw: f64| {
        (
            Vector2::new(yaw.cos(), yaw.sin()),
            Vector2::new(-yaw.sin(), yaw.cos()),
        )
    };
    let area = |yaw: f64| -> Option<f64> {
        let (heading, across) = axes(yaw);
        let ((l0, l1), (w0, w1)) = (extents(heading)?, extents(across)?);
        Some((l1 - l0) * (w1 - w0))
    };
    let mut yaw = (0..180)
        .map(|step| (step as f64 * 0.5).to_radians())
        .filter_map(|yaw| Some((yaw, area(yaw)?)))
        .min_by(|a, b| a.1.total_cmp(&b.1))?
        .0;
    let (heading, across) = axes(yaw);
    let ((l0, l1), (w0, w1)) = (extents(heading)?, extents(across)?);
    let center_xy = heading * (0.5 * (l0 + l1)) + across * (0.5 * (w0 + w1));
    let (mut length, mut width) = (l1 - l0, w1 - w0);
    if width > length {
        // Report the longer side as the length.
        std::mem::swap(&mut length, &mut width);
        yaw += std::f64::consts::FRAC_PI_2;
    }
    let heights: Vec<f64> = inliers.iter().map(|p| p.z).collect();
    let (h0, h1) = (quantile(heights.clone(), q)?, quantile(heights, 1.0 - q)?);
    Some(UprightBox {
        center: Point3::new(center_xy.x, center_xy.y, 0.5 * (h0 + h1)),
        length,
        width,
        height: h1 - h0,
        yaw,
    })
}

/// Where an estimate's metric scale comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleSource {
    /// Feature tracks triangulated with the metric VIO poses (static objects).
    Triangulated,
    /// The camera's metric height above the ground, through the object's
    /// ground contact (objects that move; [`crate::ground`]).
    GroundContact,
}

/// A metric object estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectEstimate {
    pub object_id: u64,
    pub class_id: u32,
    pub upright_box: UprightBox,
    /// Triangulated points that made it into the box fit.
    pub points: usize,
    pub scale_source: ScaleSource,
}

/// Triangulate an object's feature tracks and fit its upright box.
pub fn estimate_static_object(
    object_id: u64,
    class_id: u32,
    tracks: &[Vec<BearingObservation>],
    triangulation: &TriangulationParams,
    fit: &BoxFitParams,
) -> Option<ObjectEstimate> {
    let points: Vec<Point3<f64>> = tracks
        .iter()
        .filter_map(|track| triangulate(track, triangulation))
        .collect();
    let upright_box = fit_upright_box(&points, fit)?;
    Some(ObjectEstimate {
        object_id,
        class_id,
        upright_box,
        points: points.len(),
        scale_source: ScaleSource::Triangulated,
    })
}

/// Rough size ranges of common COCO classes, in metres: the largest
/// dimension (length, width or height) an object of the class should have.
/// Only for flagging implausible estimates; see the module docs.
pub fn plausible_size(class_id: u32) -> Option<(f64, f64)> {
    Some(match class_id {
        0 => (0.9, 2.2),    // person (height, or length when lying)
        1 => (1.2, 2.2),    // bicycle
        2 => (3.0, 6.0),    // car
        3 => (1.5, 2.6),    // motorcycle
        5 => (8.0, 20.0),   // bus
        7 => (4.0, 20.0),   // truck
        16 => (0.3, 1.3),   // dog
        39 => (0.15, 0.45), // bottle
        41 => (0.06, 0.25), // cup
        56 => (0.5, 1.3),   // chair
        57 => (1.2, 3.0),   // couch
        58 => (0.2, 1.5),   // potted plant
        60 => (0.7, 2.5),   // dining table
        62 => (0.3, 2.0),   // tv / monitor
        63 => (0.25, 0.5),  // laptop
        66 => (0.25, 0.6),  // keyboard
        73 => (0.12, 0.45), // book
        74 => (0.15, 0.6),  // clock
        _ => return None,
    })
}

fn median(values: Vec<f64>) -> Option<f64> {
    quantile(values, 0.5)
}

/// Linear-interpolated quantile of `values`, `q` in [0, 1].
fn quantile(mut values: Vec<f64>, q: f64) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let position = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let (low, high) = (position.floor() as usize, position.ceil() as usize);
    Some(values[low] + (values[high] - values[low]) * (position - low as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// A camera at `position` looking at `target` (camera z forward, y down,
    /// world z up), as camera-to-world.
    fn look_at(position: Vector3<f64>, target: Vector3<f64>) -> Isometry3<f64> {
        let z = (target - position).normalize();
        let x = z.cross(&Vector3::z()).normalize();
        let y = z.cross(&x);
        let rotation = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[x, y, z])),
        );
        Isometry3::from_parts(Translation3::from(position), rotation)
    }

    fn observe(pose: &Isometry3<f64>, point: &Point3<f64>) -> BearingObservation {
        BearingObservation {
            camera_to_world: *pose,
            bearing: (pose.inverse() * point).coords.normalize(),
        }
    }

    #[test]
    fn triangulates_exact_rays_and_rejects_bad_geometry() {
        let point = Point3::new(1.0, 3.0, 0.5);
        let poses: Vec<_> = (0..4)
            .map(|k| look_at(Vector3::new(0.2 * k as f64, 0.0, 1.0), point.coords))
            .collect();
        let observations: Vec<_> = poses.iter().map(|p| observe(p, &point)).collect();
        let found = triangulate(&observations, &TriangulationParams::default()).unwrap();
        assert!((found - point).norm() < 1e-9);
        // The same camera position four times: no parallax.
        let still: Vec<_> = (0..4).map(|_| observe(&poses[0], &point)).collect();
        assert!(triangulate(&still, &TriangulationParams::default()).is_none());
        // A ray pointing well away from the others.
        let mut bent = observations.clone();
        bent[2].bearing = (bent[2].bearing + Vector3::new(0.05, 0.0, 0.0)).normalize();
        assert!(triangulate(&bent, &TriangulationParams::default()).is_none());
        // Too few observations.
        assert!(triangulate(&observations[..2], &TriangulationParams::default()).is_none());
    }

    /// Points spread over the faces of a `length × width × height` box
    /// standing on the ground at `center_xy`, turned by `yaw`.
    fn box_points(
        rng: &mut SmallRng,
        center_xy: (f64, f64),
        size: (f64, f64, f64),
        yaw: f64,
        count: usize,
    ) -> Vec<Point3<f64>> {
        let (l, w, h) = size;
        (0..count)
            .map(|_| {
                let mut local = Vector3::new(
                    rng.gen_range(-0.5..0.5) * l,
                    rng.gen_range(-0.5..0.5) * w,
                    rng.gen_range(0.0..1.0) * h,
                );
                // Put each point on one of the four sides or the top.
                match rng.gen_range(0..5) {
                    0 => local.x = -0.5 * l,
                    1 => local.x = 0.5 * l,
                    2 => local.y = -0.5 * w,
                    3 => local.y = 0.5 * w,
                    _ => local.z = h,
                }
                let (s, c) = yaw.sin_cos();
                Point3::new(
                    center_xy.0 + c * local.x - s * local.y,
                    center_xy.1 + s * local.x + c * local.y,
                    local.z,
                )
            })
            .collect()
    }

    #[test]
    fn fits_a_turned_box_and_ignores_outliers() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut points = box_points(&mut rng, (2.0, 5.0), (1.2, 0.5, 0.8), 0.4, 400);
        // Background seen beside the object.
        points.extend((0..15).map(|i| Point3::new(8.0 + i as f64 * 0.1, 12.0, 1.0)));
        let fitted = fit_upright_box(&points, &BoxFitParams::default()).unwrap();
        assert!((fitted.length - 1.2).abs() < 0.06, "{fitted:?}");
        assert!((fitted.width - 0.5).abs() < 0.04, "{fitted:?}");
        assert!((fitted.height - 0.8).abs() < 0.04, "{fitted:?}");
        let yaw_error = (fitted.yaw - 0.4).rem_euclid(std::f64::consts::PI);
        assert!(
            yaw_error.min(std::f64::consts::PI - yaw_error) < 0.05,
            "{fitted:?}"
        );
        assert!(
            (fitted.center - Point3::new(2.0, 5.0, 0.4)).norm() < 0.05,
            "{fitted:?}"
        );
    }

    /// A camera walks 1.5 m past a chair-sized box 3 m away and observes 80
    /// surface points with 0.5 px noise (focal length 460 px).
    fn walk_past(rng: &mut SmallRng, translation_scale: f64) -> Vec<Vec<BearingObservation>> {
        let size = (0.5, 0.45, 0.9);
        let points = box_points(rng, (0.0, 3.0), size, 0.3, 80);
        let target = Vector3::new(0.0, 3.0, 0.45);
        let poses: Vec<_> = (0..30)
            .map(|k| {
                let position = Vector3::new(-0.75 + 1.5 * k as f64 / 29.0, 0.0, 1.4);
                let mut pose = look_at(position, target);
                // A wrongly scaled trajectory: every position scaled about the origin.
                pose.translation.vector *= translation_scale;
                pose
            })
            .collect();
        let true_poses: Vec<_> = (0..30)
            .map(|k| {
                look_at(
                    Vector3::new(-0.75 + 1.5 * k as f64 / 29.0, 0.0, 1.4),
                    target,
                )
            })
            .collect();
        let pixel_angle = 0.5 / 460.0;
        points
            .iter()
            .map(|p| {
                true_poses
                    .iter()
                    .zip(&poses)
                    .map(|(truth, used)| {
                        let mut o = observe(truth, p);
                        o.bearing = (o.bearing
                            + Vector3::new(
                                rng.gen_range(-1.0..1.0),
                                rng.gen_range(-1.0..1.0),
                                0.0,
                            ) * pixel_angle)
                            .normalize();
                        o.camera_to_world = *used;
                        o
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_static_object_comes_out_in_metres_with_metric_poses() {
        let mut rng = SmallRng::seed_from_u64(11);
        let tracks = walk_past(&mut rng, 1.0);
        let estimate = estimate_static_object(
            7,
            56,
            &tracks,
            &TriangulationParams::default(),
            &BoxFitParams::default(),
        )
        .unwrap();
        let b = estimate.upright_box;
        assert_eq!(estimate.scale_source, ScaleSource::Triangulated);
        assert!(estimate.points > 60, "{estimate:?}");
        assert!((b.height - 0.9).abs() < 0.05, "{b:?}");
        let (long, short) = (b.length.max(b.width), b.length.min(b.width));
        assert!(
            (long - 0.5).abs() < 0.05 && (short - 0.45).abs() < 0.05,
            "{b:?}"
        );
        assert!(
            (b.center - Point3::new(0.0, 3.0, 0.45)).norm() < 0.05,
            "{b:?}"
        );
        let (low, high) = plausible_size(56).unwrap();
        assert!((low..=high).contains(&b.height.max(long)));
    }

    #[test]
    fn a_wrong_trajectory_scale_gives_wrong_sizes() {
        // The same walk with positions doubled, as a monocular VO without
        // the IMU might report: the box comes out about twice as large.
        let mut rng = SmallRng::seed_from_u64(11);
        let tracks = walk_past(&mut rng, 2.0);
        let b = estimate_static_object(
            7,
            56,
            &tracks,
            &TriangulationParams::default(),
            &BoxFitParams::default(),
        )
        .unwrap()
        .upright_box;
        assert!((b.height / 0.9 - 2.0).abs() < 0.15, "{b:?}");
        let (low, high) = plausible_size(56).unwrap();
        assert!(
            !(low..=high).contains(&b.height.max(b.length).max(b.width)),
            "flagged by the prior"
        );
    }
}
