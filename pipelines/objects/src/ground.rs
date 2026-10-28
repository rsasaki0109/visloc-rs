//! Metric size, position and speed of objects standing on the ground (Phase 3
//! of `docs/object_metric_slam_plan.md`, moving objects).
//!
//! An object that moves while the camera moves cannot be triangulated: its
//! rays do not meet. Ground contact places it anyway:
//! - **The ground plane** is horizontal in the gravity-aligned world (z up)
//!   and lies at a metric height, found from the VIO's static scene points
//!   ([`estimate_ground_height`]).
//! - **Each frame** ([`place_on_ground`]): the ray through the middle of the
//!   2D box's bottom edge meets the ground at the object's foot. That fixes
//!   its distance, and the box's rays cut at that distance give its height
//!   and width. "Bottom" is the edge lowest in the world, so a camera
//!   mounted sideways works as well as an upright one.
//! - **Over a track** ([`summarize_ground_track`]): the median size, the path
//!   of the foot, and the speed of a line fitted to it.
//!
//! The scale comes from the camera's metric height above the ground, so it
//! is as good as the VIO's. The object must touch the ground where the box
//! ends: anything carried, stacked or seen over an edge breaks it.

use nalgebra::{Point3, UnitQuaternion, Vector2, Vector3};

use crate::BBox;

/// Settings of [`estimate_ground_height`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundParams {
    /// Height bin of the histogram (0.05 m).
    pub bin: f64,
    /// Only points this far below the lowest camera position count (0.3 m).
    pub min_below_camera: f64,
    /// Points the densest bin needs (20).
    pub min_points: usize,
}

impl Default for GroundParams {
    fn default() -> Self {
        Self {
            bin: 0.05,
            min_below_camera: 0.3,
            min_points: 20,
        }
    }
}

/// The ground's height (world z): the mean height of the points in the
/// densest `bin`-high layer below `lowest_camera_z - min_below_camera`.
pub fn estimate_ground_height(
    points: &[Point3<f64>],
    lowest_camera_z: f64,
    params: &GroundParams,
) -> Option<f64> {
    let ceiling = lowest_camera_z - params.min_below_camera;
    let below: Vec<f64> = points
        .iter()
        .map(|p| p.z)
        .filter(|z| z.is_finite() && *z < ceiling)
        .collect();
    let mut bins: std::collections::HashMap<i64, Vec<f64>> = std::collections::HashMap::new();
    for z in below {
        bins.entry((z / params.bin).floor() as i64)
            .or_default()
            .push(z);
    }
    let (_, layer) = bins
        .into_iter()
        .filter(|(_, zs)| zs.len() >= params.min_points)
        .max_by_key(|(bin, zs)| (zs.len(), std::cmp::Reverse(*bin)))?;
    Some(layer.iter().sum::<f64>() / layer.len() as f64)
}

/// An object placed on the ground in one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundObservation {
    pub timestamp_ns: i64,
    /// Where the object touches the ground.
    pub contact: Point3<f64>,
    pub height: f64,
    pub width: f64,
    /// Horizontal distance from the camera.
    pub distance: f64,
}

/// World-frame rays through a 2D box: the middles of its four edges and its
/// four corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxRays {
    pub edge_midpoints: [Vector3<f64>; 4],
    pub corners: [Vector3<f64>; 4],
}

impl BoxRays {
    /// The rays of `bbox`, unprojected by `unproject` (pixel to camera-frame
    /// ray) and turned into the world by `camera_to_world`. `None` if a point
    /// does not unproject.
    pub fn from_box(
        bbox: &BBox,
        camera_to_world: &UnitQuaternion<f64>,
        unproject: impl Fn(f64, f64) -> Option<Vector3<f64>>,
    ) -> Option<Self> {
        let (xm, ym) = bbox.center();
        let ray = |x: f64, y: f64| unproject(x, y).map(|r| camera_to_world * r);
        Some(Self {
            edge_midpoints: [
                ray(xm, bbox.y_min)?,
                ray(bbox.x_max, ym)?,
                ray(xm, bbox.y_max)?,
                ray(bbox.x_min, ym)?,
            ],
            corners: [
                ray(bbox.x_min, bbox.y_min)?,
                ray(bbox.x_max, bbox.y_min)?,
                ray(bbox.x_max, bbox.y_max)?,
                ray(bbox.x_min, bbox.y_max)?,
            ],
        })
    }
}

/// Place an upright object standing on the ground `z = ground_z`, seen from
/// `camera_center` through the rays of its 2D box. `None` if the camera is
/// not above the ground or the box's bottom edge does not point down.
pub fn place_on_ground(
    camera_center: &Point3<f64>,
    rays: &BoxRays,
    ground_z: f64,
    timestamp_ns: i64,
) -> Option<GroundObservation> {
    let height_above = camera_center.z - ground_z;
    if height_above <= 0.0 {
        return None;
    }
    let unit = |r: &Vector3<f64>| r.try_normalize(1e-12);
    let foot = rays
        .edge_midpoints
        .iter()
        .filter_map(unit)
        .min_by(|a, b| a.z.total_cmp(&b.z))?;
    if foot.z >= -1e-6 {
        return None;
    }
    let contact = camera_center + foot * (height_above / -foot.z);
    // The object's "billboard": the vertical plane through the contact
    // point, facing the camera.
    let normal = Vector3::new(
        contact.x - camera_center.x,
        contact.y - camera_center.y,
        0.0,
    );
    let distance = normal.norm();
    let normal = normal.try_normalize(1e-9)?;
    let across = Vector3::new(-normal.y, normal.x, 0.0);
    let depth = normal.dot(&(contact - camera_center));
    let hits: Vec<Point3<f64>> = rays
        .corners
        .iter()
        .chain(&rays.edge_midpoints)
        .filter_map(unit)
        .filter(|r| normal.dot(r) > 1e-6)
        .map(|r| camera_center + r * (depth / normal.dot(&r)))
        .collect();
    let top = hits.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
    let sideways: Vec<f64> = hits.iter().map(|p| across.dot(&p.coords)).collect();
    let width = sideways.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - sideways.iter().copied().fold(f64::INFINITY, f64::min);
    Some(GroundObservation {
        timestamp_ns,
        contact,
        height: top - ground_z,
        width,
        distance,
    })
}

/// An object's track on the ground.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundTrack {
    pub object_id: u64,
    pub class_id: u32,
    /// Median over the frames.
    pub height: f64,
    pub width: f64,
    pub distance: f64,
    /// The contact point per frame.
    pub path: Vec<(i64, Point3<f64>)>,
    /// Horizontal speed of a straight line fitted to the path (m/s).
    pub speed: f64,
}

/// Median size and the speed of the path, from at least three frames
/// spanning `min_seconds`.
pub fn summarize_ground_track(
    object_id: u64,
    class_id: u32,
    observations: &[GroundObservation],
    min_seconds: f64,
) -> Option<GroundTrack> {
    if observations.len() < 3 {
        return None;
    }
    let t0 = observations.iter().map(|o| o.timestamp_ns).min()?;
    let times: Vec<f64> = observations
        .iter()
        .map(|o| (o.timestamp_ns - t0) as f64 * 1e-9)
        .collect();
    let mean_t = times.iter().sum::<f64>() / times.len() as f64;
    let spread: f64 = times.iter().map(|t| (t - mean_t).powi(2)).sum();
    if times.iter().copied().fold(0.0, f64::max) < min_seconds || spread <= 0.0 {
        return None;
    }
    let mean_xy = observations
        .iter()
        .map(|o| Vector2::new(o.contact.x, o.contact.y))
        .sum::<Vector2<f64>>()
        / observations.len() as f64;
    let velocity = observations
        .iter()
        .zip(&times)
        .map(|(o, t)| (Vector2::new(o.contact.x, o.contact.y) - mean_xy) * (t - mean_t))
        .sum::<Vector2<f64>>()
        / spread;
    let median = |f: fn(&GroundObservation) -> f64| {
        let mut v: Vec<f64> = observations.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Some(GroundTrack {
        object_id,
        class_id,
        height: median(|o| o.height),
        width: median(|o| o.width),
        distance: median(|o| o.distance),
        path: observations
            .iter()
            .map(|o| (o.timestamp_ns, o.contact))
            .collect(),
        speed: velocity.norm(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Matrix3, Point2, Rotation3, Translation3, UnitQuaternion};

    const F: f64 = 400.0;
    const C: (f64, f64) = (320.0, 240.0);

    /// Camera-to-world looking along `forward` (horizontal), z up, rolled
    /// by `roll` about its optical axis (a sideways-mounted camera is 90°).
    fn camera(position: Vector3<f64>, forward: Vector3<f64>, roll: f64) -> Isometry3<f64> {
        let z = forward.normalize();
        let x = z.cross(&Vector3::z()).normalize();
        let y = z.cross(&x);
        let base = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[x, y, z]));
        let rolled = base * Rotation3::from_axis_angle(&Vector3::z_axis(), roll);
        Isometry3::from_parts(
            Translation3::from(position),
            UnitQuaternion::from_rotation_matrix(&rolled),
        )
    }

    fn project(pose: &Isometry3<f64>, p: &Point3<f64>) -> Point2<f64> {
        let c = pose.inverse() * p;
        Point2::new(F * c.x / c.z + C.0, F * c.y / c.z + C.1)
    }

    /// The rays of the 2D box of an upright `width × depth × height` cuboid
    /// standing at `foot`.
    fn outline(pose: &Isometry3<f64>, foot: Point3<f64>, size: (f64, f64, f64)) -> BoxRays {
        let (w, d, h) = size;
        let corners: Vec<Point2<f64>> = [-0.5, 0.5]
            .iter()
            .flat_map(|&a| {
                [-0.5, 0.5]
                    .iter()
                    .flat_map(move |&b| [0.0, 1.0].iter().map(move |&c| (a, b, c)))
            })
            .map(|(a, b, c)| project(pose, &(foot + Vector3::new(a * w, b * d, c * h))))
            .collect();
        let (x0, x1) = corners.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p.x), hi.max(p.x))
        });
        let (y0, y1) = corners.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p.y), hi.max(p.y))
        });
        BoxRays::from_box(&BBox::new(x0, y0, x1, y1), &pose.rotation, |x, y| {
            Some(Vector3::new((x - C.0) / F, (y - C.1) / F, 1.0))
        })
        .unwrap()
    }

    /// A person 1.75 m tall (0.5 m wide, 0.3 m deep) walks diagonally at
    /// (1.2, 0.8) m/s, about 5 m from a camera 1.5 m above the ground that
    /// itself walks forward. Returns the summarized track, with every camera position
    /// and the ground scaled by `scale` (a wrongly scaled trajectory).
    const SPEED: f64 = 1.442_220_510_185_596; // |(1.2, 0.8)|

    fn walk(roll: f64, scale: f64) -> GroundTrack {
        let observations: Vec<GroundObservation> = (0..30)
            .map(|k| {
                let t = k as f64 / 10.0;
                let camera_position = Vector3::new(0.0, 0.8 * t, 1.5);
                let true_pose = camera(camera_position, Vector3::new(0.3, 1.0, 0.0), roll);
                let foot = Point3::new(-1.8 + 1.2 * t, 5.0 + 0.8 * t, 0.0);
                let rays = outline(&true_pose, foot, (0.5, 0.3, 1.75));
                place_on_ground(
                    &Point3::from(camera_position * scale),
                    &rays,
                    0.0,
                    (t * 1e9) as i64,
                )
                .unwrap()
            })
            .collect();
        summarize_ground_track(1, 0, &observations, 1.0).unwrap()
    }

    #[test]
    fn a_walking_person_comes_out_in_metres_and_metres_per_second() {
        for roll in [0.0, std::f64::consts::FRAC_PI_2] {
            let track = walk(roll, 1.0);
            // The 2D box of a cuboid is a little larger than its silhouette.
            assert!((track.height - 1.75).abs() < 0.1, "roll {roll}: {track:?}");
            assert!((track.width - 0.5).abs() < 0.15, "roll {roll}: {track:?}");
            assert!((track.speed - SPEED).abs() < 0.05, "roll {roll}: {track:?}");
            // The foot is the near edge of the footprint, not its centre.
            let (start, end) = (track.path[0].1, track.path[29].1);
            assert!(
                (start - Point3::new(-1.8, 5.0, 0.0)).norm() < 0.35,
                "{start}"
            );
            assert!(
                (end - Point3::new(-1.8 + 3.48, 7.32, 0.0)).norm() < 0.35,
                "{end}"
            );
        }
    }

    #[test]
    fn a_wrong_scale_makes_the_person_a_giant() {
        let track = walk(0.0, 2.0);
        assert!((track.height / 1.75 - 2.0).abs() < 0.1, "{track:?}");
        assert!((track.speed / SPEED - 2.0).abs() < 0.1, "{track:?}");
    }

    #[test]
    fn rejects_a_camera_below_ground_and_rays_that_do_not_go_down() {
        let rays = |z: f64| BoxRays {
            edge_midpoints: [Vector3::new(0.0, 1.0, z); 4],
            corners: [Vector3::new(0.1, 1.0, z); 4],
        };
        assert!(place_on_ground(&Point3::new(0.0, 0.0, 1.5), &rays(0.1), 0.0, 0).is_none());
        assert!(place_on_ground(&Point3::new(0.0, 0.0, -0.5), &rays(-0.3), 0.0, 0).is_none());
        assert!(place_on_ground(&Point3::new(0.0, 0.0, 1.5), &rays(-0.3), 0.0, 0).is_some());
    }

    #[test]
    fn finds_the_floor_under_clutter() {
        let mut points = Vec::new();
        for i in 0..200 {
            let a = i as f64 * 0.37;
            points.push(Point3::new(
                a.sin() * 4.0,
                a.cos() * 4.0,
                -1.52 + 0.01 * (a * 7.0).sin(),
            ));
        }
        // A table top and some walls: fewer points per layer.
        points.extend((0..60).map(|i| Point3::new(i as f64 * 0.05, 2.0, -0.78)));
        points.extend((0..300).map(|i| Point3::new(3.0, i as f64 * 0.01, -1.4 + i as f64 * 0.01)));
        let ground = estimate_ground_height(&points, 0.0, &GroundParams::default()).unwrap();
        assert!((ground + 1.52).abs() < 0.02, "{ground}");
    }
}
