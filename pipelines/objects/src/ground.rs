//! Metric size, position and speed of objects standing on the ground (Phase 3
//! of `docs/object_metric_slam_plan.md`, moving objects).
//!
//! An object that moves while the camera moves cannot be triangulated: its
//! rays do not meet. Ground contact places it anyway:
//! - **The ground** is found from the VIO's static scene points in the
//!   gravity-aligned world (z up). [`PathGround`] follows the camera's path:
//!   the camera moves over the ground, so a point's height is measured from
//!   the camera where it passed nearest, which keeps roads that climb or fall
//!   sharp. [`estimate_ground_height`] is the flat-ground version.
//! - **Each frame** ([`place_on_ground`]): the ray through the middle of the
//!   2D box's bottom edge meets the ground at the object's foot. That fixes
//!   its distance, and the box's rays cut at that distance give its height
//!   and width. "Bottom" is the edge lowest in the world, so a camera
//!   mounted sideways works as well as an upright one.
//! - **The ground under the object** ([`place_on_local_ground`]): stairs, a
//!   ramp or a kerb can put an object on another level than the camera's
//!   ground, which makes it look small and near (a lower level) or large and
//!   far (a higher one). The ground moves to the layer of scene points under
//!   the object's placed foot until it settles.
//! - **Over a track** ([`summarize_ground_track`]): the median size, the path
//!   of the foot, and the velocity of a line fitted to it, weighted by how
//!   far each foot can be off; a velocity within that noise is still.
//!
//! The scale comes from the camera's metric height above the ground, so it
//! is as good as the VIO's. The object must touch the ground where the box
//! ends: anything carried, stacked or seen over an edge breaks it.

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Point3, UnitQuaternion, Vector2, Vector3, Vector4};

use crate::BBox;

/// Settings of [`estimate_ground_height`], [`PathGround`] and
/// [`place_on_local_ground`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundParams {
    /// Height bin of the histogram (0.05 m).
    pub bin: f64,
    /// Only points this far below the lowest camera position count (0.3 m).
    pub min_below_camera: f64,
    /// Points the densest bin needs (20).
    pub min_points: usize,
    /// The ground under an object: the points horizontally nearest to its
    /// foot (12)...
    pub local_neighbours: usize,
    /// ...within this distance (1.5 m)...
    pub local_radius: f64,
    /// ...of which its densest bin needs this many (3)...
    pub local_min_points: usize,
    /// ...and is taken if it is this much higher or lower (0.1 m).
    pub local_min_shift: f64,
    /// [`PathGround`]: the stretch of path, either way, whose points give the
    /// camera's height above the ground (20 m).
    pub path_stretch: f64,
}

impl Default for GroundParams {
    fn default() -> Self {
        Self {
            bin: 0.05,
            min_below_camera: 0.3,
            min_points: 20,
            local_neighbours: 12,
            local_radius: 1.5,
            local_min_points: 3,
            local_min_shift: 0.1,
            path_stretch: 20.0,
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
    densest_layer(
        points.iter().map(|p| p.z),
        lowest_camera_z - params.min_below_camera,
        params.bin,
        params.min_points,
    )
}

/// The mean of the densest `bin`-high layer of the heights below `ceiling`,
/// if it has `min_points`; ties go to the lower layer.
fn densest_layer(
    heights: impl Iterator<Item = f64>,
    ceiling: f64,
    bin: f64,
    min_points: usize,
) -> Option<f64> {
    densest_band(heights, ceiling, bin, 0, min_points)
}

/// [`densest_layer`] over bands of `2 * spread + 1` neighbouring bins, for
/// noisier heights.
fn densest_band(
    heights: impl Iterator<Item = f64>,
    ceiling: f64,
    bin: f64,
    spread: i64,
    min_points: usize,
) -> Option<f64> {
    let mut bins: std::collections::HashMap<i64, Vec<f64>> = std::collections::HashMap::new();
    for z in heights.filter(|z| z.is_finite() && *z < ceiling) {
        bins.entry((z / bin).floor() as i64).or_default().push(z);
    }
    let band = |k: i64| {
        (k - spread..=k + spread)
            .filter_map(|j| bins.get(&j))
            .flatten()
    };
    let (best, count) = bins
        .keys()
        .map(|&k| (k, band(k).count()))
        .filter(|(_, n)| *n >= min_points)
        .max_by_key(|(k, n)| (*n, std::cmp::Reverse(*k)))?;
    Some(band(best).sum::<f64>() / count as f64)
}

/// The ground along the camera's path. The camera moves over the ground, so
/// a scene point's height is measured from the camera where it passed
/// nearest: on a road that climbs or falls, the ground then lies at one
/// depth below the path, the camera's height above it. That height is found
/// for each stretch of the path (`path_stretch` metres either way), so a
/// camera whose height changes (a hand lowering it) is followed too. The
/// ground under any spot is the camera's height below the nearest point of
/// the path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathGround {
    /// Camera centres along the path, in order.
    pub path: Vec<Point3<f64>>,
    /// The camera's height above the ground at each of them.
    pub camera_heights: Vec<f64>,
}

impl PathGround {
    /// For each point of the path, the densest band of three `bin`-high
    /// layers of the heights of the scene `points` whose nearest path point
    /// lies within `path_stretch` along the path, each measured from that
    /// nearest point, at least `min_below_camera` down, with `min_points`. A stretch without
    /// such a layer takes the nearest stretch's. `None` if none has one.
    pub fn estimate(
        path: &[Point3<f64>],
        points: &[Point3<f64>],
        params: &GroundParams,
    ) -> Option<Self> {
        path.first()?;
        let mut along = vec![0.0];
        for pair in path.windows(2) {
            let step =
                (Vector2::new(pair[1].x, pair[1].y) - Vector2::new(pair[0].x, pair[0].y)).norm();
            along.push(along.last()? + step);
        }
        let ground = Self {
            path: path.to_vec(),
            camera_heights: Vec::new(),
        };
        let below: Vec<(usize, f64)> = points
            .iter()
            .map(|p| {
                let i = ground.nearest(&Vector2::new(p.x, p.y));
                (i, p.z - path[i].z)
            })
            .collect();
        let found: Vec<Option<f64>> = along
            .iter()
            .map(|s| {
                let heights = below
                    .iter()
                    .filter(|(i, _)| (along[*i] - s).abs() <= params.path_stretch)
                    .map(|(_, z)| *z);
                densest_band(
                    heights,
                    -params.min_below_camera,
                    params.bin,
                    1,
                    params.min_points,
                )
                .map(|layer| -layer)
            })
            .collect();
        let camera_heights = (0..found.len())
            .map(|i| {
                (0..found.len())
                    .filter_map(|j| found[j].map(|h| ((along[i] - along[j]).abs(), h)))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, h)| h)
            })
            .collect::<Option<Vec<f64>>>()?;
        Some(Self {
            camera_heights,
            ..ground
        })
    }

    /// The ground's height under `xy`.
    pub fn height_at(&self, xy: &Vector2<f64>) -> f64 {
        let i = self.nearest(xy);
        self.path[i].z - self.camera_heights[i]
    }

    /// The camera's height above the ground where it passed nearest to `xy`.
    pub fn camera_height_at(&self, xy: &Vector2<f64>) -> f64 {
        self.camera_heights[self.nearest(xy)]
    }

    fn nearest(&self, xy: &Vector2<f64>) -> usize {
        let d = |i: &usize| (Vector2::new(self.path[*i].x, self.path[*i].y) - xy).norm_squared();
        (0..self.path.len())
            .min_by(|a, b| d(a).total_cmp(&d(b)))
            .expect("a path")
    }
}

/// An object placed on the ground in one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundObservation {
    pub timestamp_ns: i64,
    /// Where the object touches the ground.
    pub contact: Point3<f64>,
    /// The camera it was seen from.
    pub camera: Point3<f64>,
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
        camera: *camera_center,
        height: top - ground_z,
        width,
        distance,
    })
}

/// Rough height ranges of COCO classes that stand on the ground, in metres.
/// The height is what ground contact measures best (the width depends on
/// the side the object is seen from). Only for flagging implausible
/// estimates, never to set a scale.
pub fn plausible_height(class_id: u32) -> Option<(f64, f64)> {
    Some(match class_id {
        0 => (1.0, 2.1),  // person, children included
        1 => (0.8, 1.4),  // bicycle
        2 => (1.2, 2.1),  // car, vans included
        3 => (0.9, 1.6),  // motorcycle
        5 => (2.7, 4.5),  // bus
        6 => (2.8, 4.8),  // train or tram
        7 => (2.0, 4.2),  // truck
        16 => (0.2, 1.0), // dog
        17 => (1.2, 2.0), // horse
        _ => return None,
    })
}

/// One view of an object: where the camera was and the rays of its box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundView {
    pub timestamp_ns: i64,
    pub camera_center: Point3<f64>,
    pub rays: BoxRays,
}

/// Place an object's views on the ground under it. Starting from `ground_z`,
/// the views are placed ([`place_on_ground`]), and the ground moves to the
/// densest layer of the `local_neighbours` scene `points` horizontally
/// nearest to the median foot (within `local_radius`, below the lowest
/// camera), while that layer lies more than `local_min_shift` away (a few
/// rounds at most). Nearest points rather than all within a radius, so that
/// over steps the ground walks down (or up) one step at a time. Returns the ground used and
/// the placed views; `None` if no view could be placed.
pub fn place_on_local_ground(
    views: &[GroundView],
    points: &[Point3<f64>],
    ground_z: f64,
    params: &GroundParams,
) -> Option<(f64, Vec<GroundObservation>)> {
    const ROUNDS: usize = 5;
    let ceiling = views
        .iter()
        .map(|v| v.camera_center.z)
        .fold(f64::INFINITY, f64::min)
        - params.min_below_camera;
    let mut ground = ground_z;
    for round in 0.. {
        let placed: Vec<GroundObservation> = views
            .iter()
            .filter_map(|v| place_on_ground(&v.camera_center, &v.rays, ground, v.timestamp_ns))
            .collect();
        if placed.is_empty() {
            return None;
        }
        let median = |f: fn(&Point3<f64>) -> f64| {
            let mut v: Vec<f64> = placed.iter().map(|o| f(&o.contact)).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let foot = Vector2::new(median(|p| p.x), median(|p| p.y));
        let mut near: Vec<(f64, f64)> = points
            .iter()
            .filter(|p| p.z < ceiling)
            .map(|p| ((Vector2::new(p.x, p.y) - foot).norm(), p.z))
            .filter(|(d, _)| *d < params.local_radius)
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        near.truncate(params.local_neighbours);
        let local = densest_layer(
            near.iter().map(|(_, z)| *z),
            ceiling,
            params.bin,
            params.local_min_points,
        );
        match local {
            Some(z) if (z - ground).abs() > params.local_min_shift && round + 1 < ROUNDS => {
                ground = z
            }
            _ => return Some((ground, placed)),
        }
    }
    unreachable!()
}

/// Settings of [`summarize_ground_track`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundTrackParams {
    /// A track needs three frames over this long (1 s).
    pub min_seconds: f64,
    /// How far a box's bottom edge is off, as an angle (0.004 rad, about
    /// 3 px at a focal length of 720 px): along the ray, the foot is off by
    /// this times `(h² + d²) / h` for a camera `h` above the ground and `d`
    /// from the foot, across it by this times `d`.
    pub angular_noise: f64,
    /// And by this share of the object's width more either way, for not
    /// knowing which point of the object the box's bottom edge falls on
    /// (0.5)...
    pub extent_noise: f64,
    /// ...but at least this much (0.2 m).
    pub min_extent_noise: f64,
    /// A fitted velocity this unlikely under the noise (chi-square, two
    /// degrees of freedom) or more counts as moving (9.21: 1 %).
    pub still_chi2: f64,
    /// A foot this far off the fitted line (the same chi-square) is dropped
    /// (9.21).
    pub outlier_chi2: f64,
}

impl Default for GroundTrackParams {
    fn default() -> Self {
        Self {
            min_seconds: 1.0,
            angular_noise: 0.004,
            extent_noise: 0.5,
            min_extent_noise: 0.2,
            still_chi2: 9.21,
            outlier_chi2: 9.21,
        }
    }
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
    /// Horizontal velocity of a straight line fitted to the path (m/s).
    pub velocity: Vector2<f64>,
    /// Its standard deviation along its direction (m/s).
    pub speed_sigma: f64,
    /// Whether the velocity stands out of the noise.
    pub moving: bool,
    /// The speed: `velocity`'s, or 0 if not `moving` (m/s).
    pub speed: f64,
}

/// Median size, and the velocity of the path, from at least three frames
/// spanning `min_seconds`.
///
/// The velocity is a straight line fitted by weighted least squares. Each
/// foot is weighted by how far it can be off: little across the ray, a lot
/// along it far away, where a pixel moves the foot by metres
/// ([`GroundTrackParams`]). A camera driving towards a still car that is
/// placed a few percent near at range would otherwise see it approach. Feet
/// far off the line (a 2D track that jumped to a neighbour) are dropped. If
/// the velocity does not stand out of the noise, the object is still.
pub fn summarize_ground_track(
    object_id: u64,
    class_id: u32,
    observations: &[GroundObservation],
    params: &GroundTrackParams,
) -> Option<GroundTrack> {
    if observations.len() < 3 {
        return None;
    }
    let t0 = observations.iter().map(|o| o.timestamp_ns).min()?;
    let times: Vec<f64> = observations
        .iter()
        .map(|o| (o.timestamp_ns - t0) as f64 * 1e-9)
        .collect();
    if times.iter().copied().fold(0.0, f64::max) < params.min_seconds {
        return None;
    }
    let mut widths: Vec<f64> = observations.iter().map(|o| o.width).collect();
    widths.sort_by(f64::total_cmp);
    let extent = (params.extent_noise * widths[widths.len() / 2]).max(params.min_extent_noise);
    // Each foot's position and 2x2 information.
    let feet: Vec<(Vector2<f64>, Matrix2<f64>)> = observations
        .iter()
        .map(|o| {
            let foot = Vector2::new(o.contact.x, o.contact.y);
            let ray = foot - Vector2::new(o.camera.x, o.camera.y);
            let d = ray.norm();
            let along = ray.try_normalize(1e-9).unwrap_or_else(Vector2::x);
            let across = Vector2::new(-along.y, along.x);
            let h = (o.camera.z - o.contact.z).max(0.3);
            let sigma_along = params.angular_noise * (h * h + d * d) / h + extent;
            let sigma_across = params.angular_noise * d + extent;
            let information = along * along.transpose() / (sigma_along * sigma_along)
                + across * across.transpose() / (sigma_across * sigma_across);
            (foot, information)
        })
        .collect();
    // p_i = p0 + v (t_i - mean_t) by weighted least squares; feet beyond
    // `outlier_chi2` of the line (a 2D track that jumped to a neighbour) are
    // dropped and the line fitted again, a few times.
    let mut keep = vec![true; feet.len()];
    let mut fit = None;
    for _ in 0..4 {
        let kept: Vec<usize> = (0..feet.len()).filter(|&i| keep[i]).collect();
        let mean_t = kept.iter().map(|&i| times[i]).sum::<f64>() / kept.len() as f64;
        let mut normal = Matrix4::<f64>::zeros();
        let mut rhs = Vector4::<f64>::zeros();
        for &i in &kept {
            let (foot, information) = &feet[i];
            let mut jacobian = Matrix2x4::<f64>::zeros();
            jacobian
                .fixed_view_mut::<2, 2>(0, 0)
                .copy_from(&Matrix2::identity());
            jacobian
                .fixed_view_mut::<2, 2>(0, 2)
                .copy_from(&(Matrix2::identity() * (times[i] - mean_t)));
            normal += jacobian.transpose() * information * jacobian;
            rhs += jacobian.transpose() * information * foot;
        }
        let covariance = normal.try_inverse()?;
        let solution = covariance * rhs;
        let inliers: Vec<bool> = (0..feet.len())
            .map(|i| {
                let (foot, information) = &feet[i];
                let r = foot
                    - Vector2::new(solution[0], solution[1])
                    - Vector2::new(solution[2], solution[3]) * (times[i] - mean_t);
                r.dot(&(information * r)) < params.outlier_chi2
            })
            .collect();
        fit = Some((covariance, solution));
        if inliers == keep || inliers.iter().filter(|&&k| k).count() < 3 {
            break;
        }
        keep = inliers;
    }
    let (covariance, solution) = fit?;
    let velocity = Vector2::new(solution[2], solution[3]);
    let velocity_covariance: Matrix2<f64> = covariance.fixed_view::<2, 2>(2, 2).into();
    let chi2 = velocity.dot(&(velocity_covariance.try_inverse()? * velocity));
    let direction = velocity.try_normalize(1e-12).unwrap_or_else(Vector2::x);
    let speed_sigma = direction
        .dot(&(velocity_covariance * direction))
        .max(0.0)
        .sqrt();
    let moving = chi2 > params.still_chi2;
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
        velocity,
        speed_sigma,
        moving,
        speed: if moving { velocity.norm() } else { 0.0 },
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
        padded_outline(pose, foot, size, 0.0)
    }

    /// [`outline`] with the box's bottom edge `pad` pixels low, as a loose
    /// detector box has it (upright camera).
    fn padded_outline(
        pose: &Isometry3<f64>,
        foot: Point3<f64>,
        size: (f64, f64, f64),
        pad: f64,
    ) -> BoxRays {
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
        BoxRays::from_box(&BBox::new(x0, y0, x1, y1 + pad), &pose.rotation, |x, y| {
            Some(Vector3::new((x - C.0) / F, (y - C.1) / F, 1.0))
        })
        .unwrap()
    }

    const SPEED: f64 = 1.442_220_510_185_596; // |(1.2, 0.8)|

    /// A person 1.75 m tall (0.5 m wide, 0.3 m deep) walks diagonally at
    /// (1.2, 0.8) m/s, about 5 m from a camera 1.5 m above the ground that
    /// itself walks forward. Returns the summarized track, with every camera
    /// position and the ground scaled by `scale` (a wrongly scaled
    /// trajectory).
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
        summarize_ground_track(1, 0, &observations, &GroundTrackParams::default()).unwrap()
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

    /// A car 4.5 x 1.8 x 1.5 m, its box 1.5 px low (the angular noise at this
    /// focal length), seen for 3 s from a camera
    /// 1.6 m up driving at 10 m/s, from 40 m to 10 m, 3 m to the side; the
    /// car drives at `velocity` (m/s, along x, across the camera's path).
    fn drive_past(velocity: f64) -> GroundTrack {
        let observations: Vec<GroundObservation> = (0..30)
            .map(|k| {
                let t = k as f64 / 10.0;
                let position = Vector3::new(0.0, 10.0 * t, 1.6);
                let pose = camera(position, Vector3::new(0.0, 1.0, 0.0), 0.0);
                let foot = Point3::new(3.0 + velocity * t, 40.0, 0.0);
                let rays = padded_outline(&pose, foot, (1.8, 4.5, 1.5), 1.5);
                place_on_ground(&Point3::from(position), &rays, 0.0, (t * 1e9) as i64).unwrap()
            })
            .collect();
        summarize_ground_track(1, 2, &observations, &GroundTrackParams::default()).unwrap()
    }

    #[test]
    fn a_parked_car_driven_past_stands_still() {
        let track = drive_past(0.0);
        // The loose box puts it near at range: a plain line fit would see it
        // approach.
        assert!(track.velocity.norm() > 0.5, "{track:?}");
        assert!(!track.moving, "{track:?}");
        assert_eq!(track.speed, 0.0);
    }

    #[test]
    fn a_car_crossing_ahead_moves() {
        let track = drive_past(5.0);
        assert!(track.moving, "{track:?}");
        assert!((track.speed - 5.0).abs() < 1.0, "{track:?}");
        assert!(track.speed_sigma < 1.0, "{track:?}");
    }

    #[test]
    fn a_track_that_jumps_to_a_neighbour_stays_still() {
        // A person stands 8 m ahead for 3 s; for 6 frames the 2D track takes
        // the person next to them, 2 m to the side.
        let position = Vector3::new(0.0, 0.0, 1.5);
        let pose = camera(position, Vector3::new(0.0, 1.0, 0.0), 0.0);
        let observations: Vec<GroundObservation> = (0..30)
            .map(|k| {
                let x = if (12..18).contains(&k) { 2.5 } else { 0.5 };
                let rays = outline(&pose, Point3::new(x, 8.0, 0.0), (0.5, 0.3, 1.75));
                place_on_ground(&Point3::from(position), &rays, 0.0, k * 100_000_000).unwrap()
            })
            .collect();
        let track =
            summarize_ground_track(1, 0, &observations, &GroundTrackParams::default()).unwrap();
        assert!(!track.moving, "{track:?}");
        let plain = GroundTrackParams {
            outlier_chi2: f64::INFINITY,
            ..GroundTrackParams::default()
        };
        let jumped = summarize_ground_track(1, 0, &observations, &plain).unwrap();
        assert!(jumped.velocity.norm() > track.velocity.norm(), "{jumped:?}");
    }

    #[test]
    fn a_wrong_scale_makes_the_person_a_giant() {
        let track = walk(0.0, 2.0);
        assert!((track.height / 1.75 - 2.0).abs() < 0.1, "{track:?}");
        assert!((track.speed / SPEED - 2.0).abs() < 0.1, "{track:?}");
        let (low, high) = plausible_height(0).unwrap();
        assert!((low..=high).contains(&1.75) && !(low..=high).contains(&track.height));
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

    /// Views of a person 1.75 m tall walking across on a level `level` m
    /// below the camera's ground (above it if negative), 6 m ahead of a
    /// camera 1.5 m above that ground; and scene points: the camera's ground
    /// up to 3 m ahead, a flight of steps to the other level and the other
    /// level beyond, up to 8.5 m ahead.
    fn below_the_steps(level: f64) -> (Vec<GroundView>, Vec<Point3<f64>>) {
        let views = (0..20)
            .map(|k| {
                let t = k as f64 / 10.0;
                let position = Vector3::new(0.0, 0.0, 1.5);
                let pose = camera(position, Vector3::new(0.0, 1.0, -0.2), 0.0);
                let foot = Point3::new(-1.0 + t, 6.0, -level);
                GroundView {
                    timestamp_ns: (t * 1e9) as i64,
                    camera_center: Point3::from(position),
                    rays: outline(&pose, foot, (0.5, 0.3, 1.75)),
                }
            })
            .collect();
        let mut points = Vec::new();
        for i in 0..40 {
            for j in 0..30 {
                let (x, y) = (-2.0 + i as f64 * 0.1, j as f64 * 0.1);
                points.push(Point3::new(x, y, 0.0));
            }
            for step in 0..5 {
                let y = 3.0 + step as f64 * 0.2;
                points.push(Point3::new(
                    -2.0 + i as f64 * 0.1,
                    y,
                    -level * (step as f64 + 1.0) / 5.0,
                ));
            }
            for j in 0..30 {
                let (x, y) = (-2.0 + i as f64 * 0.1, 4.0 + j as f64 * 0.15);
                points.push(Point3::new(x, y, -level));
            }
        }
        (views, points)
    }

    #[test]
    fn a_person_below_the_steps_stands_on_the_lower_level() {
        let (views, points) = below_the_steps(1.0);
        let params = GroundParams::default();
        // On the camera's ground, the person comes out small and near.
        let flat: Vec<GroundObservation> = views
            .iter()
            .filter_map(|v| place_on_ground(&v.camera_center, &v.rays, 0.0, v.timestamp_ns))
            .collect();
        let flat = summarize_ground_track(1, 0, &flat, &GroundTrackParams::default()).unwrap();
        assert!(flat.height < 1.2, "{flat:?}");
        // On the ground under it, it stands 1 m lower and comes out right.
        let (ground, placed) = place_on_local_ground(&views, &points, 0.0, &params).unwrap();
        assert!((ground + 1.0).abs() < 0.05, "{ground}");
        let track = summarize_ground_track(1, 0, &placed, &GroundTrackParams::default()).unwrap();
        assert!((track.height - 1.75).abs() < 0.1, "{track:?}");
        assert!((track.distance - 6.0).abs() < 0.3, "{track:?}");
    }

    #[test]
    fn a_person_up_the_steps_stands_on_the_upper_level() {
        let (views, points) = below_the_steps(-0.5);
        let (ground, placed) =
            place_on_local_ground(&views, &points, 0.0, &GroundParams::default()).unwrap();
        assert!((ground - 0.5).abs() < 0.05, "{ground}");
        let track = summarize_ground_track(1, 0, &placed, &GroundTrackParams::default()).unwrap();
        assert!((track.height - 1.75).abs() < 0.1, "{track:?}");
        assert!((track.distance - 6.0).abs() < 0.3, "{track:?}");
    }

    #[test]
    fn a_person_on_the_camera_ground_stays_there() {
        let (views, points) = below_the_steps(0.0);
        let (ground, placed) =
            place_on_local_ground(&views, &points, 0.0, &GroundParams::default()).unwrap();
        assert_eq!(ground, 0.0);
        let track = summarize_ground_track(1, 0, &placed, &GroundTrackParams::default()).unwrap();
        assert!((track.height - 1.75).abs() < 0.1, "{track:?}");
    }

    #[test]
    fn the_path_ground_follows_a_falling_road() {
        // A camera 1.6 m above a road that falls 5 % along x; the road's
        // points 0-30 m ahead and 2 m to the sides, and car bodies 0.8 m up.
        let road = |x: f64| -0.05 * x;
        let path: Vec<Point3<f64>> = (0..100)
            .map(|i| Point3::new(i as f64, 0.0, road(i as f64) + 1.6))
            .collect();
        let mut points = Vec::new();
        for i in 0..400 {
            let x = i as f64 * 0.25;
            points.push(Point3::new(x, -2.0 + (i % 5) as f64, road(x)));
            if i % 3 == 0 {
                points.push(Point3::new(
                    x,
                    3.0,
                    road(x) + 0.8 + 0.1 * ((i / 3) % 4) as f64,
                ));
            }
        }
        let ground = PathGround::estimate(&path, &points, &GroundParams::default()).unwrap();
        let height = ground.camera_height_at(&Vector2::new(50.0, 0.0));
        assert!((height - 1.6).abs() < 0.05, "{height}");
        assert!((ground.height_at(&Vector2::new(60.0, 2.5)) - road(60.0)).abs() < 0.1);
        // A flat ground smears the 5 m drop over many layers.
        let flat = estimate_ground_height(&points, path[0].z, &GroundParams::default());
        assert!(
            flat.is_none_or(|z| (z - road(60.0)).abs() > 0.5),
            "{flat:?}"
        );
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
