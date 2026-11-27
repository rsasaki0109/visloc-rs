//! A small, deterministic simulation of a camera-carrying ground robot, for
//! testing map-based localization and navigation end to end without a robot.
//!
//! - [`World`]: a floor plan of vertical wall segments (walls, furniture),
//!   whose faces carry point *texture*: [`SimLandmark`]s with synthetic
//!   descriptors, some deliberately alike (repetitive texture).
//! - [`DiffDrive`]: a differential-drive robot with acceleration limits and
//!   actuation noise, stopped by walls.
//! - [`WheelOdometry`]: odometry integrated from wheel encoders with
//!   mis-calibrated wheel radii and track width, so it drifts as real wheel
//!   odometry does.
//! - [`CameraRig::observe`]: what the robot's forward camera detects: the
//!   landmarks in view and not occluded, as pixel keypoints with noise and
//!   noisy descriptors, plus clutter, as a [`QueryImage`] for visloc's
//!   localization pipeline.
//! - [`World::scan`]: a planar range scan (a depth camera's horizontal
//!   slice) for obstacle avoidance.
//! - [`build_map`]: a mapping run along a route, producing the
//!   [`VisualMap`] a SLAM session would save (landmarks with observations,
//!   keyframes with poses, both with small errors).
//!
//! The map frame is the world frame, gravity-aligned with `z` up. There is
//! no image rendering: the camera model is exact geometry on the landmarks,
//! which tests everything downstream of feature extraction.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, Frame, Keyframe, Landmark, Observation, QueryImage, VisualMap};

/// A small deterministic random generator (SplitMix64) with Gaussian draws.
#[derive(Debug, Clone)]
pub struct SimRng {
    state: u64,
    spare: Option<f64>,
}

impl SimRng {
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Standard normal.
    pub fn gaussian(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        self.spare = Some(radius * (2.0 * PI * u2).sin());
        radius * (2.0 * PI * u2).cos()
    }
}

/// A planar pose: position and heading (yaw about `z`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pose2 {
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
}

impl Pose2 {
    pub const fn new(x: f64, y: f64, yaw: f64) -> Self {
        Self { x, y, yaw }
    }

    /// The 3D pose (`parent_from_this`) on the floor plane.
    pub fn to_se3(&self) -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), self.yaw),
            Vector3::new(self.x, self.y, 0.0),
        )
    }

    /// The planar part of a 3D pose (its yaw about `z`).
    pub fn from_se3(pose: &SE3) -> Self {
        let heading = pose.rotation * Vector3::x();
        Self::new(
            pose.translation.x,
            pose.translation.y,
            heading.y.atan2(heading.x),
        )
    }

    /// `self * other`.
    pub fn compose(&self, other: &Self) -> Self {
        let (s, c) = self.yaw.sin_cos();
        Self::new(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            wrap_angle(self.yaw + other.yaw),
        )
    }

    pub fn inverse(&self) -> Self {
        let (s, c) = self.yaw.sin_cos();
        Self::new(
            -(c * self.x + s * self.y),
            -(-s * self.x + c * self.y),
            wrap_angle(-self.yaw),
        )
    }
}

/// Wraps an angle to `(-pi, pi]`.
pub fn wrap_angle(angle: f64) -> f64 {
    let wrapped = (angle + PI).rem_euclid(2.0 * PI) - PI;
    if wrapped <= -PI {
        wrapped + 2.0 * PI
    } else {
        wrapped
    }
}

/// A vertical wall segment from the floor to `height`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wall {
    pub a: Point2<f64>,
    pub b: Point2<f64>,
    pub height: f64,
}

/// A point of wall texture with its descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct SimLandmark {
    pub id: u64,
    pub position: Point3<f64>,
    pub descriptor: Vec<f32>,
}

/// Texture parameters of [`World::add_wall`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Texture {
    /// Mean spacing of texture points along a wall face, in metres.
    pub spacing: f64,
    /// Texture points lie up to this height (and the wall's).
    pub max_height: f64,
    /// Fraction of points whose descriptor copies an earlier point's (plus
    /// `alias_noise`): repetitive texture that matches the wrong landmark.
    pub alias_fraction: f64,
    pub alias_noise: f64,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            spacing: 0.08,
            max_height: 2.2,
            alias_fraction: 0.15,
            alias_noise: 0.08,
        }
    }
}

/// Descriptor length of [`SimLandmark`]s.
pub const DESCRIPTOR_DIM: usize = 32;

/// A floor plan with textured walls.
#[derive(Debug, Clone)]
pub struct World {
    pub walls: Vec<Wall>,
    pub landmarks: Vec<SimLandmark>,
    rng: SimRng,
}

impl World {
    pub fn empty(seed: u64) -> Self {
        Self {
            walls: Vec::new(),
            landmarks: Vec::new(),
            rng: SimRng::new(seed),
        }
    }

    fn random_descriptor(&mut self) -> Vec<f32> {
        let mut d: Vec<f64> = (0..DESCRIPTOR_DIM).map(|_| self.rng.gaussian()).collect();
        let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        d.iter_mut().for_each(|v| *v /= norm);
        d.into_iter().map(|v| v as f32).collect()
    }

    /// Adds a wall from `a` to `b`, textured on both faces (`texture: None`
    /// leaves it bare, as a textureless wall).
    pub fn add_wall(
        &mut self,
        a: (f64, f64),
        b: (f64, f64),
        height: f64,
        texture: Option<Texture>,
    ) {
        let wall = Wall {
            a: Point2::new(a.0, a.1),
            b: Point2::new(b.0, b.1),
            height,
        };
        self.walls.push(wall);
        let Some(texture) = texture else {
            return;
        };
        let along = wall.b - wall.a;
        let length = along.norm();
        if length < 1e-9 {
            return;
        }
        let dir = along / length;
        let normal = nalgebra::Vector2::new(-dir.y, dir.x);
        let top = height.min(texture.max_height);
        for side in [1.0, -1.0] {
            let count = (length / texture.spacing).round().max(1.0) as usize;
            for i in 0..count {
                let s = (i as f64 + self.rng.range(0.1, 0.9)) / count as f64 * length;
                let z = self.rng.range(0.05, top);
                let p = wall.a + dir * s + normal * (side * 0.01);
                let descriptor =
                    if !self.landmarks.is_empty() && self.rng.uniform() < texture.alias_fraction {
                        let source = (self.rng.next_u64() % self.landmarks.len() as u64) as usize;
                        let mut d = self.landmarks[source].descriptor.clone();
                        for v in &mut d {
                            *v += (self.rng.gaussian() * texture.alias_noise) as f32;
                        }
                        normalize(&mut d);
                        d
                    } else {
                        self.random_descriptor()
                    };
                let id = self.landmarks.len() as u64 + 1;
                self.landmarks.push(SimLandmark {
                    id,
                    position: Point3::new(p.x, p.y, z),
                    descriptor,
                });
            }
        }
    }

    /// Adds an axis-aligned box (furniture) of four textured walls.
    pub fn add_box(
        &mut self,
        min: (f64, f64),
        max: (f64, f64),
        height: f64,
        texture: Option<Texture>,
    ) {
        let corners = [
            (min.0, min.1),
            (max.0, min.1),
            (max.0, max.1),
            (min.0, max.1),
        ];
        for i in 0..4 {
            self.add_wall(corners[i], corners[(i + 1) % 4], height, texture);
        }
    }

    /// A 10 m x 7 m apartment: a living room, a kitchen, two bedrooms and a
    /// hallway, joined by 1 m doorways, with a sofa, a table and a shelf.
    /// One hallway wall is bare (textureless).
    pub fn apartment(seed: u64) -> Self {
        let mut w = Self::empty(seed);
        let t = Some(Texture::default());
        let h = 2.5;
        // Outer walls.
        w.add_wall((0.0, 0.0), (10.0, 0.0), h, t);
        w.add_wall((10.0, 0.0), (10.0, 7.0), h, t);
        w.add_wall((10.0, 7.0), (0.0, 7.0), h, t);
        w.add_wall((0.0, 7.0), (0.0, 0.0), h, t);
        // Living room (x < 6, y < 4) | kitchen (x > 6, y < 4): doorway y 1.5..2.5.
        w.add_wall((6.0, 0.0), (6.0, 1.5), h, t);
        w.add_wall((6.0, 2.5), (6.0, 4.0), h, t);
        // Hallway y 4..5: south wall with doorways x 2..3 (living) and 7..8 (kitchen).
        w.add_wall((0.0, 4.0), (2.0, 4.0), h, t);
        w.add_wall((3.0, 4.0), (7.0, 4.0), h, t);
        w.add_wall((8.0, 4.0), (10.0, 4.0), h, t);
        // North wall of the hallway, bare, with doorways x 1..2 and 8..9.
        w.add_wall((0.0, 5.0), (1.0, 5.0), h, None);
        w.add_wall((2.0, 5.0), (8.0, 5.0), h, None);
        w.add_wall((9.0, 5.0), (10.0, 5.0), h, t);
        // Bedrooms (y > 5) split at x = 5.
        w.add_wall((5.0, 5.0), (5.0, 7.0), h, t);
        // Furniture.
        w.add_box((1.0, 0.4), (3.0, 1.2), 0.8, t); // sofa
        w.add_box((3.4, 2.2), (4.6, 3.0), 0.75, t); // table
        w.add_box((8.8, 0.3), (9.7, 1.6), 1.8, t); // kitchen shelf
        w.add_box((5.2, 6.0), (6.6, 7.0), 0.5, t); // bed
        w
    }

    /// Whether the straight segment between two points is blocked by a wall
    /// (a wall crossing it below the wall's height).
    pub fn occluded(&self, from: &Point3<f64>, to: &Point3<f64>) -> bool {
        self.walls.iter().any(|wall| {
            segment_intersection(
                Point2::new(from.x, from.y),
                Point2::new(to.x, to.y),
                wall.a,
                wall.b,
            )
            .is_some_and(|t| {
                let z = from.z + (to.z - from.z) * t;
                t > 1e-6 && t < 1.0 - 1e-6 && z < wall.height
            })
        })
    }

    /// Range to the nearest wall from `origin` along `angle` in the plane
    /// at height `z`, up to `max_range`.
    pub fn ray_range(
        &self,
        origin: Point2<f64>,
        angle: f64,
        z: f64,
        max_range: f64,
    ) -> Option<f64> {
        let end = origin + nalgebra::Vector2::new(angle.cos(), angle.sin()) * max_range;
        self.walls
            .iter()
            .filter(|wall| wall.height > z)
            .filter_map(|wall| segment_intersection(origin, end, wall.a, wall.b))
            .map(|t| t * max_range)
            .min_by(f64::total_cmp)
    }

    /// A planar range scan of a sensor at `sensor` (see [`ScanConfig`]):
    /// ranges evenly from `-fov/2` to `+fov/2` about its heading, `None`
    /// beyond `max_range`, with Gaussian range noise.
    pub fn scan(&self, sensor: &Pose2, config: &ScanConfig, rng: &mut SimRng) -> Vec<Option<f64>> {
        let beams = config.beams;
        (0..beams)
            .map(|i| {
                let a = if beams > 1 {
                    -config.fov / 2.0 + config.fov * i as f64 / (beams - 1) as f64
                } else {
                    0.0
                };
                self.ray_range(
                    Point2::new(sensor.x, sensor.y),
                    sensor.yaw + a,
                    config.height,
                    config.max_range,
                )
                .map(|r| (r + rng.gaussian() * config.noise).max(0.0))
            })
            .collect()
    }

    /// Whether a disc of `radius` at `(x, y)` touches a wall taller than 5 cm.
    pub fn collides(&self, x: f64, y: f64, radius: f64) -> bool {
        let p = Point2::new(x, y);
        self.walls
            .iter()
            .filter(|wall| wall.height > 0.05)
            .any(|wall| point_segment_distance(p, wall.a, wall.b) < radius)
    }
}

fn normalize(d: &mut [f32]) {
    let norm = d.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    d.iter_mut().for_each(|v| *v /= norm);
}

/// Parameter `t` along `p0 -> p1` where it crosses segment `q0 -> q1`.
fn segment_intersection(
    p0: Point2<f64>,
    p1: Point2<f64>,
    q0: Point2<f64>,
    q1: Point2<f64>,
) -> Option<f64> {
    let r = p1 - p0;
    let s = q1 - q0;
    let denom = r.x * s.y - r.y * s.x;
    if denom.abs() < 1e-12 {
        return None;
    }
    let qp = q0 - p0;
    let t = (qp.x * s.y - qp.y * s.x) / denom;
    let u = (qp.x * r.y - qp.y * r.x) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some(t)
}

fn point_segment_distance(p: Point2<f64>, a: Point2<f64>, b: Point2<f64>) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(&ab) / ab.norm_squared().max(1e-12)).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm()
}

/// A planar range sensor: the horizontal slice of a depth camera, or a 2D
/// laser scanner, for [`World::scan`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanConfig {
    /// Height of the scan plane above the floor; walls lower than this are
    /// not seen.
    pub height: f64,
    /// Field of view in radians.
    pub fov: f64,
    pub beams: usize,
    pub max_range: f64,
    /// Range noise (standard deviation, metres).
    pub noise: f64,
}

impl Default for ScanConfig {
    /// A depth camera's slice: 87 degrees, 0.2 m above the floor, 4 m.
    fn default() -> Self {
        Self {
            height: 0.2,
            fov: 87f64.to_radians(),
            beams: 88,
            max_range: 4.0,
            noise: 0.01,
        }
    }
}

/// Limits and noise of [`DiffDrive`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveConfig {
    pub radius: f64,
    pub max_linear: f64,
    pub max_angular: f64,
    pub linear_accel: f64,
    pub angular_accel: f64,
    /// Relative noise of the executed velocities.
    pub actuation_noise: f64,
}

impl Default for DriveConfig {
    fn default() -> Self {
        Self {
            radius: 0.2,
            max_linear: 0.5,
            max_angular: 1.5,
            linear_accel: 1.0,
            angular_accel: 3.0,
            actuation_noise: 0.02,
        }
    }
}

/// A differential-drive robot in a [`World`]. `pose` is the ground truth.
#[derive(Debug, Clone)]
pub struct DiffDrive {
    pub config: DriveConfig,
    pub pose: Pose2,
    /// Executed velocities of the last step.
    pub linear: f64,
    pub angular: f64,
    /// Steps on which a wall stopped the robot.
    pub collisions: usize,
}

impl DiffDrive {
    pub const fn new(config: DriveConfig, pose: Pose2) -> Self {
        Self {
            config,
            pose,
            linear: 0.0,
            angular: 0.0,
            collisions: 0,
        }
    }

    /// Moves for `dt` seconds toward the commanded velocities (limited in
    /// magnitude and acceleration, with actuation noise). A move that would
    /// put the robot into a wall is not made (the robot stops) and counts as
    /// a collision. Returns the executed `(linear, angular)` velocities.
    pub fn step(
        &mut self,
        world: &World,
        linear_cmd: f64,
        angular_cmd: f64,
        dt: f64,
        rng: &mut SimRng,
    ) -> (f64, f64) {
        let c = self.config;
        let target_v = linear_cmd.clamp(-c.max_linear, c.max_linear);
        let target_w = angular_cmd.clamp(-c.max_angular, c.max_angular);
        let v =
            self.linear + (target_v - self.linear).clamp(-c.linear_accel * dt, c.linear_accel * dt);
        let w = self.angular
            + (target_w - self.angular).clamp(-c.angular_accel * dt, c.angular_accel * dt);
        let v_exec = v * (1.0 + c.actuation_noise * rng.gaussian());
        let w_exec = w * (1.0 + c.actuation_noise * rng.gaussian());
        let next = integrate(self.pose, v_exec, w_exec, dt);
        if world.collides(next.x, next.y, c.radius)
            && !world.collides(self.pose.x, self.pose.y, c.radius)
        {
            self.collisions += 1;
            self.linear = 0.0;
            self.angular = 0.0;
            return (0.0, 0.0);
        }
        self.pose = next;
        self.linear = v;
        self.angular = w;
        (v_exec, w_exec)
    }
}

/// Exact unicycle integration over `dt`.
pub fn integrate(pose: Pose2, v: f64, w: f64, dt: f64) -> Pose2 {
    let theta = w * dt;
    let (dx, dy) = if theta.abs() < 1e-9 {
        (v * dt, 0.0)
    } else {
        let r = v / w;
        (r * theta.sin(), r * (1.0 - theta.cos()))
    };
    pose.compose(&Pose2::new(dx, dy, theta))
}

/// Calibration errors and noise of [`WheelOdometry`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdometryConfig {
    /// Relative error of the left / right wheel radius the odometry assumes.
    pub left_scale_error: f64,
    pub right_scale_error: f64,
    /// True and assumed distance between the wheels.
    pub track: f64,
    pub track_error: f64,
    /// Relative noise per wheel-speed reading.
    pub noise: f64,
}

impl Default for OdometryConfig {
    fn default() -> Self {
        Self {
            left_scale_error: 0.0025,
            right_scale_error: -0.0025,
            track: 0.35,
            track_error: 0.02,
            noise: 0.02,
        }
    }
}

/// Wheel odometry in an `odom` frame that starts at the robot's first pose.
#[derive(Debug, Clone)]
pub struct WheelOdometry {
    pub config: OdometryConfig,
    pub pose: Pose2,
    pub linear: f64,
    pub angular: f64,
}

impl WheelOdometry {
    pub const fn new(config: OdometryConfig) -> Self {
        Self {
            config,
            pose: Pose2::new(0.0, 0.0, 0.0),
            linear: 0.0,
            angular: 0.0,
        }
    }

    /// Integrates the wheel speeds of an executed motion `(v, w)` over `dt`.
    pub fn step(&mut self, v: f64, w: f64, dt: f64, rng: &mut SimRng) {
        let c = self.config;
        let left = (v - w * c.track / 2.0) * (1.0 + c.left_scale_error + c.noise * rng.gaussian());
        let right =
            (v + w * c.track / 2.0) * (1.0 + c.right_scale_error + c.noise * rng.gaussian());
        let track = c.track * (1.0 + c.track_error);
        self.linear = (left + right) / 2.0;
        self.angular = (right - left) / track;
        self.pose = integrate(self.pose, self.linear, self.angular, dt);
    }
}

/// Detection noise of [`CameraRig::observe`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraNoise {
    /// Pixel noise of keypoints (standard deviation).
    pub pixel_sigma: f64,
    /// Per-dimension noise of descriptors.
    pub descriptor_sigma: f64,
    /// Probability that a landmark in view is detected.
    pub detection_probability: f64,
    /// Keypoints that are no landmark (random positions and descriptors).
    pub clutter: usize,
    /// Landmarks are seen from this distance at most.
    pub max_range: f64,
}

impl Default for CameraNoise {
    fn default() -> Self {
        Self {
            pixel_sigma: 0.7,
            descriptor_sigma: 0.04,
            detection_probability: 0.8,
            clutter: 40,
            max_range: 8.0,
        }
    }
}

/// The robot's forward camera.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraRig {
    pub camera: Camera,
    /// `base_from_camera`.
    pub base_from_camera: SE3,
}

impl CameraRig {
    /// A 640 x 480 pinhole camera (about 90 degrees across) looking forward
    /// from `height` above the floor at the robot's centre.
    pub fn forward(height: f64) -> Self {
        let camera = Camera::pinhole(1, 640, 480, 320.0, 320.0, 320.0, 240.0);
        // Camera z (optical axis) = base x, camera x = -base y, camera y = -base z.
        let rotation = Rotation3::from_matrix_unchecked(Matrix3::new(
            0.0, 0.0, 1.0, //
            -1.0, 0.0, 0.0, //
            0.0, -1.0, 0.0,
        ));
        Self {
            camera,
            base_from_camera: SE3::new(
                UnitQuaternion::from_rotation_matrix(&rotation),
                Vector3::new(0.1, 0.0, height),
            ),
        }
    }

    /// The camera pose (`Pose`, world-to-camera) of a robot at `base`.
    pub fn camera_pose(&self, base: &Pose2) -> Pose {
        Pose {
            world_to_camera: base.to_se3().compose(&self.base_from_camera).inverse(),
        }
    }

    /// The base pose a camera pose implies.
    pub fn base_from_camera_pose(&self, pose: &Pose) -> SE3 {
        pose.camera_to_world()
            .compose(&self.base_from_camera.inverse())
    }

    /// Landmarks in view of a robot at `base`: `(index into world.landmarks,
    /// exact pixel)`.
    pub fn visible(
        &self,
        world: &World,
        base: &Pose2,
        max_range: f64,
    ) -> Vec<(usize, Point2<f64>)> {
        let pose = self.camera_pose(base);
        let center = pose.camera_center_world();
        let (w, h) = (f64::from(self.camera.width), f64::from(self.camera.height));
        world
            .landmarks
            .iter()
            .enumerate()
            .filter_map(|(i, landmark)| {
                let pc = pose.transform_world_point(&landmark.position);
                if pc.z < 0.2 || (landmark.position - center).norm() > max_range {
                    return None;
                }
                let px = self.camera.project(&pc)?;
                if px.x < 0.0 || px.y < 0.0 || px.x >= w || px.y >= h {
                    return None;
                }
                (!world.occluded(&center, &landmark.position)).then_some((i, px))
            })
            .collect()
    }

    /// What the camera of a robot at `base` detects, as a query image:
    /// noisy keypoints and descriptors of the landmarks in view, plus
    /// clutter, shuffled. Also returns, per keypoint, the landmark index it
    /// came from (`None` for clutter).
    pub fn observe(
        &self,
        world: &World,
        base: &Pose2,
        noise: &CameraNoise,
        rng: &mut SimRng,
    ) -> (QueryImage, Vec<Option<usize>>) {
        let (w, h) = (f64::from(self.camera.width), f64::from(self.camera.height));
        let mut features: Vec<(Point2<f64>, Vec<f32>, Option<usize>)> = Vec::new();
        for (index, px) in self.visible(world, base, noise.max_range) {
            if rng.uniform() >= noise.detection_probability {
                continue;
            }
            let noisy = Point2::new(
                px.x + rng.gaussian() * noise.pixel_sigma,
                px.y + rng.gaussian() * noise.pixel_sigma,
            );
            let mut d = world.landmarks[index].descriptor.clone();
            for v in &mut d {
                *v += (rng.gaussian() * noise.descriptor_sigma) as f32;
            }
            normalize(&mut d);
            features.push((noisy, d, Some(index)));
        }
        for _ in 0..noise.clutter {
            let mut d: Vec<f32> = (0..DESCRIPTOR_DIM).map(|_| rng.gaussian() as f32).collect();
            normalize(&mut d);
            features.push((Point2::new(rng.range(0.0, w), rng.range(0.0, h)), d, None));
        }
        // Shuffle (Fisher-Yates) so keypoint order carries no information.
        for i in (1..features.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            features.swap(i, j);
        }
        let sources = features.iter().map(|f| f.2).collect();
        let query = QueryImage {
            camera: self.camera.clone(),
            keypoints: features.iter().map(|f| f.0).collect(),
            descriptors: features.into_iter().map(|f| f.1).collect(),
        };
        (query, sources)
    }
}

/// Drives along `waypoints` with a simple heading controller on the true
/// pose: turn toward the next waypoint, drive while roughly facing it.
/// Returns the `(linear, angular)` command and whether the route is done
/// (`next` advances in place).
pub fn waypoint_command(
    pose: &Pose2,
    waypoints: &[Pose2],
    next: &mut usize,
    config: &DriveConfig,
) -> (f64, f64, bool) {
    while *next < waypoints.len() {
        let target = waypoints[*next];
        if (target.x - pose.x).hypot(target.y - pose.y) < 0.15 {
            *next += 1;
        } else {
            break;
        }
    }
    let Some(target) = waypoints.get(*next) else {
        return (0.0, 0.0, true);
    };
    let distance = (target.x - pose.x).hypot(target.y - pose.y);
    let bearing = wrap_angle((target.y - pose.y).atan2(target.x - pose.x) - pose.yaw);
    let angular = (2.0 * bearing).clamp(-config.max_angular, config.max_angular);
    let linear = if bearing.abs() > 0.5 {
        0.0
    } else {
        (config.max_linear * (1.0 - bearing.abs() / 0.5)).min(distance)
    };
    (linear, angular, false)
}

/// Landmarks seen from a keyframe: index into the world's landmarks, pixel.
type Sightings = Vec<(usize, Point2<f64>)>;

/// Settings of [`build_map`].
#[derive(Debug, Clone, PartialEq)]
pub struct MappingConfig {
    pub drive: DriveConfig,
    pub camera_noise: CameraNoise,
    /// Simulation step.
    pub dt: f64,
    /// A keyframe is taken after this much travel or turn.
    pub keyframe_distance: f64,
    pub keyframe_angle: f64,
    /// Error of the mapped landmark positions and keyframe positions
    /// (standard deviation, metres), standing in for SLAM error.
    pub landmark_sigma: f64,
    pub keyframe_sigma: f64,
    /// Landmarks seen in fewer keyframes are left out of the map.
    pub min_observations: usize,
    pub seed: u64,
}

impl Default for MappingConfig {
    fn default() -> Self {
        Self {
            drive: DriveConfig::default(),
            camera_noise: CameraNoise {
                clutter: 0,
                ..CameraNoise::default()
            },
            dt: 0.05,
            keyframe_distance: 0.3,
            keyframe_angle: 0.3,
            landmark_sigma: 0.02,
            keyframe_sigma: 0.01,
            min_observations: 2,
            seed: 7,
        }
    }
}

/// A route through every room of [`World::apartment`], starting at
/// `(1.5, 2.5)` facing `+x`, and ending back there.
pub fn apartment_mapping_route() -> Vec<Pose2> {
    [
        (1.5, 2.5),
        (2.8, 1.8),
        (5.0, 1.6),
        (5.3, 3.5),
        (3.0, 3.5),
        (0.7, 2.6),
        (0.7, 1.7),
        (2.5, 3.5),
        (2.5, 4.5),
        (1.5, 4.5),
        (1.5, 6.0),
        (3.8, 6.1),
        (1.5, 6.0),
        (1.5, 4.5),
        (8.5, 4.5),
        (8.5, 5.6),
        (6.0, 5.5),
        (9.3, 6.4),
        (8.5, 5.6),
        (8.5, 4.5),
        (7.5, 4.5),
        (7.5, 3.0),
        (8.3, 2.2),
        (7.0, 0.9),
        (6.6, 2.0),
        (5.0, 1.8),
        (3.0, 1.7),
        (1.5, 2.5),
    ]
    .iter()
    .map(|&(x, y)| Pose2::new(x, y, 0.0))
    .collect()
}

/// A mapping run's output.
#[derive(Debug, Clone)]
pub struct MappingRun {
    /// The map: landmarks (at their mapped, slightly wrong positions, with
    /// their descriptor and observations) and keyframes (camera poses with
    /// errors, keypoints and descriptors).
    pub map: VisualMap,
    /// True robot poses at the keyframes.
    pub keyframe_ground_truth: Vec<Pose2>,
    /// Robot ground positions of the run, every step.
    pub trajectory: Vec<Pose2>,
}

/// Drives a robot along `route` (with [`waypoint_command`] on the true
/// pose) and records what a visual SLAM session would map: a keyframe every
/// `keyframe_distance` / `keyframe_angle`, and every landmark detected in at
/// least `min_observations` keyframes.
pub fn build_map(
    world: &World,
    rig: &CameraRig,
    route: &[Pose2],
    config: &MappingConfig,
) -> MappingRun {
    let mut rng = SimRng::new(config.seed);
    let start = route.first().copied().unwrap_or_default();
    let mut robot = DiffDrive::new(
        DriveConfig {
            actuation_noise: 0.0,
            ..config.drive
        },
        start,
    );
    robot.pose.yaw = route
        .get(1)
        .map_or(0.0, |next| (next.y - start.y).atan2(next.x - start.x));
    let mut next = 1;
    let mut last_keyframe: Option<Pose2> = None;
    let mut keyframes: Vec<(Pose2, Sightings)> = Vec::new();
    let mut trajectory = vec![robot.pose];
    for _ in 0..200_000 {
        let take = last_keyframe.is_none_or(|last| {
            (robot.pose.x - last.x).hypot(robot.pose.y - last.y) > config.keyframe_distance
                || wrap_angle(robot.pose.yaw - last.yaw).abs() > config.keyframe_angle
        });
        if take {
            let mut seen = Vec::new();
            for (i, px) in rig.visible(world, &robot.pose, config.camera_noise.max_range) {
                if rng.uniform() >= config.camera_noise.detection_probability {
                    continue;
                }
                let sigma = config.camera_noise.pixel_sigma;
                let noisy =
                    Point2::new(px.x + rng.gaussian() * sigma, px.y + rng.gaussian() * sigma);
                seen.push((i, noisy));
            }
            keyframes.push((robot.pose, seen));
            last_keyframe = Some(robot.pose);
        }
        let (v, w, done) = waypoint_command(&robot.pose, route, &mut next, &robot.config);
        if done {
            break;
        }
        robot.step(world, v, w, config.dt, &mut rng);
        trajectory.push(robot.pose);
    }

    // Ordered: each landmark below draws its noise from `rng` in this order,
    // so a hash map's per-process order made the map differ between runs.
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for (_, seen) in &keyframes {
        for (i, _) in seen {
            *counts.entry(*i).or_default() += 1;
        }
    }
    let mut map = VisualMap::new();
    map.cameras.insert(rig.camera.id, rig.camera.clone());
    let mut landmarks: BTreeMap<usize, Landmark> = BTreeMap::new();
    for (&index, &count) in &counts {
        if count < config.min_observations {
            continue;
        }
        let source = &world.landmarks[index];
        let mut position = source.position;
        for k in 0..3 {
            position[k] += rng.gaussian() * config.landmark_sigma;
        }
        let mut landmark = Landmark::new(source.id, position);
        landmark.descriptor = Some(source.descriptor.clone());
        landmarks.insert(index, landmark);
    }
    let mut keyframe_ground_truth = Vec::with_capacity(keyframes.len());
    for (k, (pose, seen)) in keyframes.iter().enumerate() {
        let frame_id = k as u64 + 1;
        let mut noisy = *pose;
        noisy.x += rng.gaussian() * config.keyframe_sigma;
        noisy.y += rng.gaussian() * config.keyframe_sigma;
        noisy.yaw += rng.gaussian() * config.keyframe_sigma;
        let mut frame = Frame::new(frame_id, rig.camera.id);
        frame.pose = Some(rig.camera_pose(&noisy));
        let mut observations = Vec::new();
        for (index, px) in seen {
            let Some(landmark) = landmarks.get_mut(index) else {
                continue;
            };
            let observation = Observation {
                frame_id,
                landmark_id: landmark.id,
                keypoint_index: frame.keypoints.len(),
                xy: *px,
            };
            frame.keypoints.push(*px);
            frame
                .descriptors
                .push(landmark.descriptor.clone().unwrap_or_default());
            landmark.observations.push(observation.clone());
            observations.push(observation);
        }
        map.keyframes.insert(
            frame_id,
            Keyframe {
                frame,
                observations,
            },
        );
        keyframe_ground_truth.push(*pose);
    }
    map.landmarks = landmarks.into_values().map(|l| (l.id, l)).collect();
    MappingRun {
        map,
        keyframe_ground_truth,
        trajectory,
    }
}
