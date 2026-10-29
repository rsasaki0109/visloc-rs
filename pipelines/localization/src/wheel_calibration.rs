//! Where a wheeled robot's odometry is measured, from a run with a VIO or
//! against a map.
//!
//! Wheel odometry reports a point near the middle of the drive axle, about
//! which the robot turns in place, and keeps that frame level. A robot's
//! description (`T_imu_body`, the body being e.g. `base_link`) may put the
//! body a few centimetres from that point and a degree or two off level.
//! With the camera high above the wheels, either moves the camera pose the
//! wheel odometry implies by centimetres in every turn. And the VIO, with
//! the map built from it, may be a few percent off in scale from the
//! wheels.
//!
//! [`WheelCalibration`] pairs wheel odometry poses with reference poses and
//! fits, in the plane,
//!
//! ```text
//! p_body(t) = s * R(a) * (p_wheel(t) + R(yaw_wheel(t)) * r) + t0
//! ```
//!
//! where `p_body` is the body origin in the reference, `r` the offset (body
//! frame) of that origin from the wheel odometry's point, and `s`, `a`, `t0`
//! a similarity between the two trajectories. Turns in place are what reveal
//! `r`. It also measures the body frame's tilt against the reference's
//! gravity, which on a level floor is an error of `T_imu_body`.
//! `scripts/calibrate_wheel_offset.py` fits the same offline.
//!
//! Two references feed it. A VIO ([`WheelCalibration::add_vio`]) follows
//! every turn, which shows the offset. Localizations against a map
//! ([`WheelCalibration::add_fix`]) come only where the camera sees the map,
//! rarely through a turn in place, but they carry the map's own scale and
//! gravity, which is what a consumer correcting the odometry into that map
//! needs. A VIO session's scale differs from the map's by a few percent
//! (by up to 3.6% on OpenLORIS-Scene), so it is reported but not used. So
//! the offset comes from the VIO, else from the map; the level from the
//! map, else from the VIO; the scale from the map only; and each from a
//! prior ([`WheelCalibration::with_prior`]) until the pairs show it.

use std::collections::VecDeque;

use nalgebra::{Matrix2, Matrix2x4, Matrix4, UnitQuaternion, Vector2, Vector3, Vector4};
use visloc_core::geometry::SE3;

/// Longest gap between the two wheel poses a map localization is
/// interpolated between.
const MAX_FIX_WHEEL_GAP_NS: i64 = 200_000_000;
/// Wheel poses kept for map localizations, which arrive after their image.
const FIX_WHEEL_HISTORY_NS: i64 = 10_000_000_000;
/// Map localizations waiting for a later wheel pose, at most.
const MAX_WAITING_FIXES: usize = 100;
/// Typical residuals of the fit (m): against the VIO, and against map
/// localizations. With [`WheelCalibrationConfig::offset_stddev_m`] they
/// weigh the offset's prior.
const VIO_STDDEV_M: f64 = 0.005;
const FIX_STDDEV_M: f64 = 0.015;

#[derive(Debug, Clone, PartialEq)]
pub struct WheelCalibrationConfig {
    /// A wheel pose pairs with the nearest VIO pose at most this far away.
    pub max_gap_ns: i64,
    /// The VIO gives the estimate once the wheel odometry has turned this
    /// many degrees in total over its pairs (which shows the offset) ...
    pub min_turn_deg: f64,
    /// ... and travelled this many metres (which shows the scale; turning
    /// in place alone cannot tell a larger offset from a larger scale) ...
    pub min_travel_m: f64,
    /// ... over at least this many pairs.
    pub min_pairs: usize,
    /// Map localizations give the level once there are this many, and the
    /// scale (and without a VIO the offset) once they have also travelled
    /// `min_travel_m`.
    pub min_fixes: usize,
    /// A map localization further than this (m) from the fit through the
    /// others is left out of it.
    pub max_fix_residual_m: f64,
    /// Only pairs from the last `window_s` seconds count (0: all), so that
    /// a long run's estimate follows the odometry's slow changes.
    pub window_s: f64,
    /// How far (m) the offset may be from the prior's (or from zero) before
    /// the pairs show it: the fit holds the offset there where the turns
    /// seen so far do not show it, as map localizations rarely do.
    pub offset_stddev_m: f64,
    /// Level the body frame against the reference's gravity (a level floor).
    pub level: bool,
}

impl Default for WheelCalibrationConfig {
    fn default() -> Self {
        Self {
            max_gap_ns: 20_000_000,
            min_turn_deg: 180.0,
            min_travel_m: 1.0,
            min_pairs: 200,
            min_fixes: 20,
            max_fix_residual_m: 0.10,
            window_s: 0.0,
            offset_stddev_m: 0.03,
            level: true,
        }
    }
}

/// Where a value of a [`WheelCalibrationEstimate`] comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationSource {
    /// The pairs with the VIO.
    Vio,
    /// The pairs with map localizations.
    Map,
    /// [`WheelCalibration::with_prior`].
    Prior,
    /// Nothing: no offset, scale 1, the body frame as given.
    Default,
}

impl CalibrationSource {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vio => "vio",
            Self::Map => "map",
            Self::Prior => "prior",
            Self::Default => "default",
        }
    }
}

/// What [`WheelCalibration::estimate`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct WheelCalibrationEstimate {
    /// Offset, in the body frame and the wheel odometry's units, of the
    /// body origin from the wheel odometry's point.
    pub offset: Vector2<f64>,
    /// The map's scale against the wheel odometry (1 until the map
    /// localizations show it, or the prior's).
    pub scale: f64,
    /// The VIO's scale against the wheel odometry, when it gave the offset
    /// (not used: a VIO session's scale is not the map's).
    pub vio_scale: Option<f64>,
    /// Rotation from the levelled body frame to the given one (identity
    /// without levelling).
    pub body_from_level: UnitQuaternion<f64>,
    /// Tilt of the given body frame against the reference's gravity, in
    /// degrees.
    pub tilt_deg: f64,
    /// RMS residual of the fit that gave the offset, in metres of its
    /// reference, and that fit without the offset.
    pub rms_m: f64,
    pub rms_without_offset_m: f64,
    /// RMS residual of the map localizations' fit, when they gave a value.
    pub map_rms_m: Option<f64>,
    /// Total turn of the wheel odometry over the VIO pairs, in degrees, and
    /// the number of pairs and of map localizations used.
    pub turned_deg: f64,
    pub pairs: usize,
    pub fixes: usize,
    pub offset_from: CalibrationSource,
    pub scale_from: CalibrationSource,
    pub level_from: CalibrationSource,
}

impl WheelCalibrationEstimate {
    /// No calibration: the body where the description puts it, scale 1.
    pub fn identity() -> Self {
        Self {
            offset: Vector2::zeros(),
            scale: 1.0,
            vio_scale: None,
            body_from_level: UnitQuaternion::identity(),
            tilt_deg: 0.0,
            rms_m: f64::NAN,
            rms_without_offset_m: f64::NAN,
            map_rms_m: None,
            turned_deg: 0.0,
            pairs: 0,
            fixes: 0,
            offset_from: CalibrationSource::Default,
            scale_from: CalibrationSource::Default,
            level_from: CalibrationSource::Default,
        }
    }

    /// `T_imu_body` with the body moved to the wheel odometry's point and
    /// levelled (the wheel odometry's own frame).
    pub fn imu_from_wheel(&self, imu_from_body: &SE3) -> SE3 {
        let rotation = imu_from_body.rotation * self.body_from_level;
        let offset = Vector3::new(self.offset.x, self.offset.y, 0.0);
        SE3::new(rotation, imu_from_body.translation - rotation * offset)
    }

    /// The body's pose in the odometry frame, in the map's scale, from a
    /// wheel odometry pose (of the wheel odometry's point, level).
    pub fn odom_from_body(&self, odom_from_wheel: &SE3) -> SE3 {
        let offset = Vector3::new(self.offset.x, self.offset.y, 0.0);
        SE3::new(
            odom_from_wheel.rotation * self.body_from_level.inverse(),
            self.scale * (odom_from_wheel.translation + odom_from_wheel.rotation * offset),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct Pair {
    timestamp_ns: i64,
    body: Vector2<f64>,
    wheel: Vector2<f64>,
    wheel_yaw: f64,
    up_in_body: Vector3<f64>,
}

/// Pairs wheel odometry with VIO poses and map localizations as they arrive
/// and fits where the wheel odometry is measured.
#[derive(Debug, Clone)]
pub struct WheelCalibration {
    config: WheelCalibrationConfig,
    /// The body in the VIO's IMU frame; VIO poses wait for it.
    imu_from_body: Option<SE3>,
    prior: Option<WheelCalibrationEstimate>,
    /// Recent VIO poses (`world_from_imu`), oldest first.
    vio: VecDeque<(i64, SE3)>,
    /// Wheel poses waiting for a later VIO pose.
    waiting: VecDeque<(i64, SE3)>,
    /// Recent wheel poses, for map localizations.
    wheel_history: VecDeque<(i64, SE3)>,
    /// Map localizations waiting for a later wheel pose.
    waiting_fixes: VecDeque<(i64, SE3)>,
    pairs: VecDeque<Pair>,
    fix_pairs: VecDeque<Pair>,
}

impl WheelCalibration {
    /// Pairs with a VIO, `imu_from_body` being the robot description's
    /// transform from the odometry's body (e.g. `base_link`) to the VIO's
    /// IMU.
    pub fn new(config: WheelCalibrationConfig, imu_from_body: SE3) -> Self {
        let mut calibration = Self::without_vio(config);
        calibration.imu_from_body = Some(imu_from_body);
        calibration
    }

    /// Pairs with map localizations only, until
    /// [`Self::set_imu_from_body`].
    pub fn without_vio(config: WheelCalibrationConfig) -> Self {
        Self {
            config,
            imu_from_body: None,
            prior: None,
            vio: VecDeque::new(),
            waiting: VecDeque::new(),
            wheel_history: VecDeque::new(),
            waiting_fixes: VecDeque::new(),
            pairs: VecDeque::new(),
            fix_pairs: VecDeque::new(),
        }
    }

    /// Values for what the pairs do not show yet (e.g. a saved calibration).
    #[must_use]
    pub fn with_prior(mut self, prior: WheelCalibrationEstimate) -> Self {
        self.prior = Some(prior);
        self
    }

    /// Where the VIO's IMU sits on the body; VIO poses pair from now on.
    pub fn set_imu_from_body(&mut self, imu_from_body: SE3) {
        self.imu_from_body = Some(imu_from_body);
    }

    pub const fn has_imu_from_body(&self) -> bool {
        self.imu_from_body.is_some()
    }

    /// Pairs with the VIO.
    pub fn pairs(&self) -> usize {
        self.pairs.len()
    }

    /// Map localizations paired with the wheel odometry.
    pub fn fixes(&self) -> usize {
        self.fix_pairs.len()
    }

    /// Total turn of the wheel odometry over the VIO pairs, in degrees.
    pub fn turned_deg(&self) -> f64 {
        turned_rad(&self.pairs).to_degrees()
    }

    /// The wheel odometry's path length over the VIO pairs, in its own units.
    pub fn travel_m(&self) -> f64 {
        travel(&self.pairs)
    }

    /// Turn and path of the wheel odometry between map localizations.
    pub fn fix_turned_deg(&self) -> f64 {
        turned_rad(&self.fix_pairs).to_degrees()
    }

    pub fn fix_travel_m(&self) -> f64 {
        travel(&self.fix_pairs)
    }

    /// Whether the VIO pairs have seen enough turning and travel to give
    /// the offset.
    pub fn is_ready(&self) -> bool {
        self.pairs.len() >= self.config.min_pairs.max(3)
            && self.turned_deg() >= self.config.min_turn_deg
            && self.travel_m() >= self.config.min_travel_m
    }

    /// Whether there are enough map localizations to give the level.
    pub fn fixes_ready(&self) -> bool {
        self.fix_pairs.len() >= self.config.min_fixes.max(3)
    }

    /// Whether the map localizations have also travelled enough to give the
    /// scale.
    pub fn fixes_travelled(&self) -> bool {
        self.fixes_ready() && self.fix_travel_m() >= self.config.min_travel_m
    }

    /// Adds a VIO pose (the IMU in the VIO's gravity-aligned world), in time
    /// order.
    pub fn add_vio(&mut self, timestamp_ns: i64, world_from_imu: &SE3) {
        if self.imu_from_body.is_none()
            || self
                .vio
                .back()
                .is_some_and(|(last, _)| timestamp_ns <= *last)
        {
            return;
        }
        self.vio.push_back((timestamp_ns, world_from_imu.clone()));
        while let Some((t, _)) = self.waiting.front() {
            if *t > timestamp_ns {
                break;
            }
            let (t, wheel) = self.waiting.pop_front().expect("front exists");
            self.pair(t, &wheel);
        }
        // Keep what the waiting wheel poses may still need.
        let keep_from = self
            .waiting
            .front()
            .map_or(timestamp_ns, |(t, _)| (*t).min(timestamp_ns))
            - self.config.max_gap_ns;
        while self.vio.len() > 2 && self.vio[1].0 < keep_from {
            self.vio.pop_front();
        }
    }

    /// Adds a wheel odometry pose (its point, level), in time order.
    pub fn add_wheel(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) {
        if self
            .wheel_history
            .back()
            .is_some_and(|(last, _)| timestamp_ns <= *last)
        {
            return;
        }
        self.wheel_history
            .push_back((timestamp_ns, odom_from_wheel.clone()));
        while self
            .wheel_history
            .front()
            .is_some_and(|(t, _)| timestamp_ns - t > FIX_WHEEL_HISTORY_NS)
        {
            self.wheel_history.pop_front();
        }
        while self
            .waiting_fixes
            .front()
            .is_some_and(|(t, _)| *t <= timestamp_ns)
        {
            let (t, fix) = self.waiting_fixes.pop_front().expect("front exists");
            self.pair_fix(t, &fix);
        }
        if self.imu_from_body.is_none() {
            return;
        }
        if self
            .vio
            .back()
            .is_some_and(|(last, _)| timestamp_ns <= *last)
        {
            self.pair(timestamp_ns, odom_from_wheel);
        } else {
            self.waiting
                .push_back((timestamp_ns, odom_from_wheel.clone()));
        }
    }

    /// Adds a localization against a map: the body's pose in the map
    /// (gravity-aligned), at its image's stamp. It pairs with the wheel
    /// odometry interpolated to that stamp, now or once the wheel odometry
    /// gets there.
    pub fn add_fix(&mut self, timestamp_ns: i64, map_from_body: &SE3) {
        if self
            .wheel_history
            .back()
            .is_some_and(|(last, _)| timestamp_ns <= *last)
        {
            self.pair_fix(timestamp_ns, map_from_body);
        } else if self.waiting_fixes.len() < MAX_WAITING_FIXES {
            self.waiting_fixes
                .push_back((timestamp_ns, map_from_body.clone()));
        }
    }

    fn pair(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) {
        let Some(imu_from_body) = &self.imu_from_body else {
            return;
        };
        let Some((t, world_from_imu)) = self
            .vio
            .iter()
            .min_by_key(|(t, _)| (t - timestamp_ns).abs())
        else {
            return;
        };
        if (t - timestamp_ns).abs() > self.config.max_gap_ns {
            return;
        }
        let body = world_from_imu.translation + world_from_imu.rotation * imu_from_body.translation;
        let world_from_body = world_from_imu.rotation * imu_from_body.rotation;
        let pair = Pair {
            timestamp_ns,
            body: body.xy(),
            wheel: odom_from_wheel.translation.xy(),
            wheel_yaw: yaw_of(&odom_from_wheel.rotation),
            up_in_body: world_from_body.inverse() * Vector3::z(),
        };
        self.pairs.push_back(pair);
        drop_older(&mut self.pairs, timestamp_ns, self.config.window_s);
    }

    fn pair_fix(&mut self, timestamp_ns: i64, map_from_body: &SE3) {
        let index = self
            .wheel_history
            .partition_point(|(t, _)| *t < timestamp_ns);
        let Some((after_ns, after)) = self.wheel_history.get(index) else {
            return;
        };
        let wheel = if *after_ns == timestamp_ns {
            after.clone()
        } else {
            let Some((before_ns, before)) = index.checked_sub(1).map(|i| &self.wheel_history[i])
            else {
                return;
            };
            if after_ns - before_ns > MAX_FIX_WHEEL_GAP_NS {
                return;
            }
            let alpha = (timestamp_ns - before_ns) as f64 / (after_ns - before_ns) as f64;
            SE3::new(
                before.rotation.slerp(&after.rotation, alpha),
                before.translation.lerp(&after.translation, alpha),
            )
        };
        let pair = Pair {
            timestamp_ns,
            body: map_from_body.translation.xy(),
            wheel: wheel.translation.xy(),
            wheel_yaw: yaw_of(&wheel.rotation),
            up_in_body: map_from_body.rotation.inverse() * Vector3::z(),
        };
        if self
            .fix_pairs
            .back()
            .is_some_and(|last| timestamp_ns <= last.timestamp_ns)
        {
            return;
        }
        self.fix_pairs.push_back(pair);
        drop_older(&mut self.fix_pairs, timestamp_ns, self.config.window_s);
    }

    /// The fit over the pairs so far: the offset from the VIO (from the map
    /// localizations without one), the level from the map localizations
    /// (from the VIO without them), the scale from the map localizations,
    /// the rest from the prior. `None` until the VIO pairs are
    /// [ready](Self::is_ready) or the map localizations
    /// [ready](Self::fixes_ready).
    pub fn estimate(&self) -> Option<WheelCalibrationEstimate> {
        let vio_ready = self.is_ready();
        let fixes_ready = self.fixes_ready();
        let fixes_travelled = self.fixes_travelled();
        if !vio_ready && !fixes_ready {
            return None;
        }
        let prior = self.prior.as_ref();
        let prior_offset = prior.map_or_else(Vector2::zeros, |prior| prior.offset);
        let weight = |stddev: f64| {
            if self.config.offset_stddev_m > 0.0 {
                (stddev / self.config.offset_stddev_m).powi(2)
            } else {
                0.0
            }
        };
        let pairs: Vec<Pair> = self.pairs.iter().copied().collect();
        let vio_fit = vio_ready.then(|| {
            fit(
                &pairs,
                OffsetFit::Fitted(prior_offset, weight(VIO_STDDEV_M)),
                30,
            )
        });

        let mut fixes: Vec<Pair> = self.fix_pairs.iter().copied().collect();
        let (offset, offset_from, rms_m, rms_without_offset_m) = if let Some(vio_fit) = &vio_fit {
            let without = fit(&pairs, OffsetFit::Held(Vector2::zeros()), 0);
            (
                vio_fit.offset,
                CalibrationSource::Vio,
                vio_fit.rms,
                without.rms,
            )
        } else if fixes_travelled {
            let fitted = robust_fit(
                &mut fixes,
                OffsetFit::Fitted(prior_offset, weight(FIX_STDDEV_M)),
                self.config.max_fix_residual_m,
                self.config.min_fixes,
            );
            let without = fit(&fixes, OffsetFit::Held(Vector2::zeros()), 0);
            (
                fitted.offset,
                CalibrationSource::Map,
                fitted.rms,
                without.rms,
            )
        } else if let Some(prior) = prior {
            (prior.offset, CalibrationSource::Prior, f64::NAN, f64::NAN)
        } else {
            (
                Vector2::zeros(),
                CalibrationSource::Default,
                f64::NAN,
                f64::NAN,
            )
        };

        let map_fit = fixes_travelled.then(|| {
            robust_fit(
                &mut fixes,
                OffsetFit::Held(offset),
                self.config.max_fix_residual_m,
                self.config.min_fixes,
            )
        });
        let (scale, scale_from) = match (&map_fit, prior) {
            (Some(map_fit), _) => (map_fit.scale, CalibrationSource::Map),
            (None, Some(prior)) => (prior.scale, CalibrationSource::Prior),
            (None, None) => (1.0, CalibrationSource::Default),
        };

        let mean_up = |pairs: &[Pair]| {
            pairs
                .iter()
                .map(|pair| pair.up_in_body)
                .sum::<Vector3<f64>>()
                .normalize()
        };
        // The map localizations the scale's fit kept, or all of them.
        let up = if fixes_ready {
            Some((mean_up(&fixes), CalibrationSource::Map))
        } else if vio_ready {
            Some((mean_up(&pairs), CalibrationSource::Vio))
        } else {
            None
        };
        let (body_from_level, tilt_deg, level_from) = match (up, prior) {
            (Some((up, source)), _) => {
                let tilt_deg = up.z.clamp(-1.0, 1.0).acos().to_degrees();
                let body_from_level = if self.config.level {
                    UnitQuaternion::rotation_between(&Vector3::z(), &up)
                        .unwrap_or_else(UnitQuaternion::identity)
                } else {
                    UnitQuaternion::identity()
                };
                (body_from_level, tilt_deg, source)
            }
            (None, Some(prior)) => (
                prior.body_from_level,
                prior.tilt_deg,
                CalibrationSource::Prior,
            ),
            (None, None) => (UnitQuaternion::identity(), 0.0, CalibrationSource::Default),
        };

        Some(WheelCalibrationEstimate {
            offset,
            scale,
            vio_scale: vio_fit.as_ref().map(|fit| fit.scale),
            body_from_level,
            tilt_deg,
            rms_m,
            rms_without_offset_m,
            map_rms_m: map_fit.map(|fit| fit.rms),
            turned_deg: self.turned_deg(),
            pairs: self.pairs.len(),
            fixes: fixes.len(),
            offset_from,
            scale_from,
            level_from,
        })
    }
}

fn drop_older(pairs: &mut VecDeque<Pair>, newest_ns: i64, window_s: f64) {
    if window_s <= 0.0 {
        return;
    }
    let window_ns = (window_s * 1e9) as i64;
    while pairs
        .front()
        .is_some_and(|pair| newest_ns - pair.timestamp_ns > window_ns)
    {
        pairs.pop_front();
    }
}

fn turned_rad(pairs: &VecDeque<Pair>) -> f64 {
    pairs
        .iter()
        .zip(pairs.iter().skip(1))
        .map(|(a, b)| wrap_angle(b.wheel_yaw - a.wheel_yaw).abs())
        .sum()
}

fn travel(pairs: &VecDeque<Pair>) -> f64 {
    pairs
        .iter()
        .zip(pairs.iter().skip(1))
        .map(|(a, b)| (b.wheel - a.wheel).norm())
        .sum()
}

fn yaw_of(rotation: &UnitQuaternion<f64>) -> f64 {
    let matrix = rotation.to_rotation_matrix();
    matrix[(1, 0)].atan2(matrix[(0, 0)])
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

fn rotation2(angle: f64) -> Matrix2<f64> {
    let (s, c) = angle.sin_cos();
    Matrix2::new(c, -s, s, c)
}

struct Fit {
    offset: Vector2<f64>,
    scale: f64,
    rms: f64,
    residuals: Vec<f64>,
}

/// The offset in [`fit`]: held, or fitted with a prior and the prior's
/// weight against a pair's squared residual.
#[derive(Debug, Clone, Copy)]
enum OffsetFit {
    Held(Vector2<f64>),
    Fitted(Vector2<f64>, f64),
}

/// Offset `r`, scale and residuals of
/// `body ~ s R(a) (wheel + R(yaw) r) + t0`, alternating the similarity and
/// the (linear) offset with `t0` `iterations` times, from the held offset
/// or the prior. A fitted offset is drawn to its prior by the weight, which
/// decides it where the pairs' turns do not.
fn fit(pairs: &[Pair], offset_fit: OffsetFit, iterations: usize) -> Fit {
    let n = pairs.len() as f64;
    let (mut offset, prior) = match offset_fit {
        OffsetFit::Held(offset) => (offset, None),
        OffsetFit::Fitted(prior, weight) => (prior, Some((prior, weight))),
    };
    let similarity = |offset: &Vector2<f64>| {
        let q: Vec<Vector2<f64>> = pairs
            .iter()
            .map(|pair| pair.wheel + rotation2(pair.wheel_yaw) * offset)
            .collect();
        let mean_q = q.iter().sum::<Vector2<f64>>() / n;
        let mean_g = pairs.iter().map(|pair| pair.body).sum::<Vector2<f64>>() / n;
        let h = q
            .iter()
            .zip(pairs)
            .map(|(q, pair)| (q - mean_q) * (pair.body - mean_g).transpose())
            .sum::<Matrix2<f64>>();
        let rotation = rotation2((h[(0, 1)] - h[(1, 0)]).atan2(h[(0, 0)] + h[(1, 1)]));
        let spread: f64 = q.iter().map(|q| (q - mean_q).norm_squared()).sum();
        let scale = if spread > 0.0 {
            q.iter()
                .zip(pairs)
                .map(|(q, pair)| (pair.body - mean_g).dot(&(rotation * (q - mean_q))))
                .sum::<f64>()
                / spread
        } else {
            1.0
        };
        let translation = mean_g - scale * rotation * mean_q;
        (rotation, scale, translation, q)
    };
    if let Some((prior, weight)) = prior {
        for _ in 0..iterations {
            let (rotation, scale, _, _) = similarity(&offset);
            // body - s R wheel = s R R(yaw) r + t0, linear in (r, t0).
            let (mut normal, mut rhs) = (Matrix4::zeros(), Vector4::zeros());
            for pair in pairs {
                let a = scale * rotation * rotation2(pair.wheel_yaw);
                let m = Matrix2x4::new(
                    a[(0, 0)],
                    a[(0, 1)],
                    1.0,
                    0.0,
                    a[(1, 0)],
                    a[(1, 1)],
                    0.0,
                    1.0,
                );
                let b = pair.body - scale * rotation * pair.wheel;
                normal += m.transpose() * m;
                rhs += m.transpose() * b;
            }
            for axis in 0..2 {
                normal[(axis, axis)] += weight;
                rhs[axis] += weight * prior[axis];
            }
            match normal.try_inverse() {
                Some(inverse) => offset = (inverse * rhs).fixed_rows::<2>(0).into_owned(),
                None => break,
            }
        }
    }
    let (rotation, scale, translation, q) = similarity(&offset);
    let residuals: Vec<f64> = q
        .iter()
        .zip(pairs)
        .map(|(q, pair)| (pair.body - (scale * rotation * q + translation)).norm())
        .collect();
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    Fit {
        offset,
        scale,
        rms,
        residuals,
    }
}

/// [`fit`] over map localizations, leaving out those further than
/// `max_residual` from the fit through the rest (twice), as long as
/// `min_kept` remain. `pairs` keeps those used.
fn robust_fit(
    pairs: &mut Vec<Pair>,
    offset_fit: OffsetFit,
    max_residual: f64,
    min_kept: usize,
) -> Fit {
    let mut result = fit(pairs, offset_fit, 30);
    for _ in 0..2 {
        if max_residual <= 0.0 || result.residuals.iter().all(|r| *r <= max_residual) {
            break;
        }
        let kept: Vec<Pair> = pairs
            .iter()
            .zip(&result.residuals)
            .filter(|(_, r)| **r <= max_residual)
            .map(|(pair, _)| *pair)
            .collect();
        if kept.len() < min_kept.max(3) {
            break;
        }
        *pairs = kept;
        result = fit(pairs, offset_fit, 30);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    fn rz(angle: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
    }

    /// A drive: 2 m ahead, a half turn in place, 2 m back, a quarter turn.
    fn drive() -> Vec<(f64, f64, f64)> {
        let (mut x, mut y, mut yaw, mut poses) = (0.0, 0.0, 0.0, Vec::new());
        for (steps, speed, turn) in [
            (100, 0.02, 0.0),
            (90, 0.0, std::f64::consts::PI / 90.0),
            (100, 0.02, 0.0),
            (45, 0.0, std::f64::consts::PI / 90.0),
        ] {
            for _ in 0..steps {
                x += speed * f64::cos(yaw);
                y += speed * f64::sin(yaw);
                yaw += turn;
                poses.push((x, y, yaw));
            }
        }
        poses
    }

    /// The true `T_imu_body`: the IMU 0.9 m above the body, rotated.
    fn imu_from_body() -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(-1.55, 0.02, -1.6),
            Vector3::new(0.03, -0.89, -0.25),
        )
    }

    /// The true body pose in a reference `scale` times the wheels', with
    /// the body origin `offset` from the wheel point.
    fn body_in(world: &SE3, wheel: &SE3, offset: Vector2<f64>, scale: f64) -> SE3 {
        let body_position =
            scale * (wheel.translation + wheel.rotation * Vector3::new(offset.x, offset.y, 0.0));
        world.compose(&SE3::new(wheel.rotation, body_position))
    }

    fn wheel_pose((x, y, yaw): (f64, f64, f64)) -> SE3 {
        SE3::new(rz(yaw), Vector3::new(x, y, 0.0))
    }

    /// Feeds a run whose body origin is `offset` from the wheel point and
    /// whose VIO is `scale` times the wheels, and the map localizations
    /// `fixes` once the wheel pose after their image is in.
    fn run_with(
        offset: Vector2<f64>,
        scale: f64,
        given: &SE3,
        fixes: &[(i64, SE3)],
    ) -> WheelCalibration {
        let truth = imu_from_body();
        let world = SE3::new(rz(0.4), Vector3::new(0.3, -1.2, 0.5));
        let mut calibration =
            WheelCalibration::new(WheelCalibrationConfig::default(), given.clone());
        let mut fixes = fixes.iter().peekable();
        for (index, pose) in drive().into_iter().enumerate() {
            let wheel = wheel_pose(pose);
            let world_from_imu = body_in(&world, &wheel, offset, scale).compose(&truth.inverse());
            let t = wheel_stamp(index);
            // The VIO runs 2 ms behind the wheels' stamps and arrives first
            // half the time.
            if index % 2 == 0 {
                calibration.add_vio(t + 2 * MS, &world_from_imu);
                calibration.add_wheel(t, &wheel);
            } else {
                calibration.add_wheel(t, &wheel);
                calibration.add_vio(t + 2 * MS, &world_from_imu);
            }
            while let Some((fix_t, fix)) = fixes.next_if(|(fix_t, _)| *fix_t <= t) {
                calibration.add_fix(*fix_t, fix);
            }
        }
        calibration
    }

    fn run(offset: Vector2<f64>, scale: f64, given: &SE3) -> WheelCalibration {
        run_with(offset, scale, given, &[])
    }

    fn wheel_stamp(index: usize) -> i64 {
        1_000 * MS + index as i64 * 50 * MS
    }

    /// Map localizations of [`drive`] halfway between every `every`-th
    /// wheel pose and the one before: the body in a map `scale` times the
    /// wheels', with the given body frame tilted by `tilt` against the
    /// map's gravity.
    fn fixes(
        offset: Vector2<f64>,
        scale: f64,
        tilt: &UnitQuaternion<f64>,
        every: usize,
    ) -> Vec<(i64, SE3)> {
        let map = SE3::new(rz(-1.1), Vector3::new(4.0, 2.0, 0.0));
        let poses = drive();
        (1..poses.len())
            .step_by(every)
            .map(|index| {
                let (a, b) = (wheel_pose(poses[index - 1]), wheel_pose(poses[index]));
                let wheel = SE3::new(
                    a.rotation.slerp(&b.rotation, 0.5),
                    a.translation.lerp(&b.translation, 0.5),
                );
                let body = body_in(&map, &wheel, offset, scale);
                (
                    wheel_stamp(index) - 25 * MS,
                    SE3::new(body.rotation * tilt.inverse(), body.translation),
                )
            })
            .collect()
    }

    #[test]
    fn recovers_the_offset_scale_and_level() {
        let offset = Vector2::new(0.041, 0.064);
        let calibration = run(offset, 1.03, &imu_from_body());
        assert!(calibration.is_ready());
        assert!(!calibration.fixes_ready());
        let estimate = calibration.estimate().expect("estimate");
        // (The prior, zero, draws the offset by a hundredth of a mm.)
        assert!((estimate.offset - offset).norm() < 1e-5, "{estimate:?}");
        // The VIO's scale is reported, not used: no map localizations, so 1.
        assert!((estimate.vio_scale.expect("vio") - 1.03).abs() < 1e-5);
        assert_eq!(estimate.scale, 1.0);
        assert!(estimate.rms_m < 1e-4);
        assert!(estimate.rms_without_offset_m > 0.02);
        assert!(estimate.tilt_deg < 1e-6);
        assert!(
            (estimate.turned_deg - 270.0).abs() < 2.5,
            "{}",
            estimate.turned_deg
        );
        assert_eq!(estimate.pairs, drive().len());
        assert_eq!(
            (
                estimate.offset_from,
                estimate.scale_from,
                estimate.level_from
            ),
            (
                CalibrationSource::Vio,
                CalibrationSource::Default,
                CalibrationSource::Vio
            )
        );
        assert_eq!(estimate.map_rms_m, None);
    }

    #[test]
    fn a_tilted_extrinsic_is_levelled_and_the_body_follows_the_wheels() {
        let truth = imu_from_body();
        let tilt = UnitQuaternion::from_euler_angles(0.0, 2f64.to_radians(), 0.0);
        let given = SE3::new(truth.rotation * tilt, truth.translation);
        let offset = Vector2::new(0.013, 0.062);
        let estimate = run(offset, 1.0, &given).estimate().expect("estimate");
        assert!((estimate.tilt_deg - 2.0).abs() < 1e-6);
        // Levelled, the given extrinsic becomes the true one (no turn about z).
        let imu_from_wheel = estimate.imu_from_wheel(&given);
        assert!(imu_from_wheel.rotation.angle_to(&truth.rotation) < 1e-9);
        let expected = truth.translation - truth.rotation * Vector3::new(offset.x, offset.y, 0.0);
        assert!((imu_from_wheel.translation - expected).norm() < 1e-5);
        // The body pose from a wheel pose puts the IMU where the given
        // extrinsic says, as the VIO saw it.
        let wheel = SE3::new(rz(0.7), Vector3::new(1.0, 2.0, 0.0));
        let odom_from_body = estimate.odom_from_body(&wheel);
        let imu_from_wheel_pose = wheel.compose(&imu_from_wheel.inverse());
        let imu_from_body_pose = odom_from_body.compose(&given.inverse());
        assert!((imu_from_wheel_pose.translation - imu_from_body_pose.translation).norm() < 1e-5);
        assert!(
            imu_from_wheel_pose
                .rotation
                .angle_to(&imu_from_body_pose.rotation)
                < 1e-9
        );
    }

    #[test]
    fn not_ready_before_enough_turning() {
        let mut calibration =
            WheelCalibration::new(WheelCalibrationConfig::default(), imu_from_body());
        for index in 0..300 {
            let t = index as i64 * 50 * MS;
            let pose = SE3::new(rz(0.0), Vector3::new(index as f64 * 0.02, 0.0, 0.0));
            calibration.add_vio(t, &pose.compose(&imu_from_body().inverse()));
            calibration.add_wheel(t, &pose);
        }
        assert_eq!(calibration.pairs(), 300);
        assert!(calibration.travel_m() > 5.9);
        assert!(!calibration.is_ready());
        assert!(calibration.estimate().is_none());
        // Wheel poses without a VIO pose close enough are not paired.
        calibration.add_wheel(400 * 50 * MS, &SE3::identity());
        calibration.add_vio(401 * 50 * MS, &SE3::identity());
        assert_eq!(calibration.pairs(), 300);
    }

    #[test]
    fn map_localizations_give_the_scale_and_the_level_and_the_vio_the_offset() {
        let offset = Vector2::new(0.041, 0.064);
        // The VIO is 3 % off the wheels' scale, the map 1.2 %; the map's
        // gravity is 1.5 deg off the VIO's for the given body frame.
        let tilt = UnitQuaternion::from_euler_angles(1.5f64.to_radians(), 0.0, 0.0);
        let fixes = fixes(offset, 1.012, &tilt, 7);
        let calibration = run_with(offset, 1.03, &imu_from_body(), &fixes);
        assert!(calibration.fixes_ready());
        let estimate = calibration.estimate().expect("estimate");
        assert_eq!(estimate.offset_from, CalibrationSource::Vio);
        assert_eq!(estimate.scale_from, CalibrationSource::Map);
        assert_eq!(estimate.level_from, CalibrationSource::Map);
        assert!((estimate.offset - offset).norm() < 1e-5);
        assert!((estimate.scale - 1.012).abs() < 1e-5, "{}", estimate.scale);
        assert!(
            (estimate.tilt_deg - 1.5).abs() < 1e-6,
            "{}",
            estimate.tilt_deg
        );
        assert!(estimate.map_rms_m.expect("map fit") < 1e-4);
        assert_eq!(estimate.fixes, fixes.len());
    }

    #[test]
    fn map_localizations_alone_hold_the_prior_offset_until_they_turn_and_drop_outliers() {
        let offset = Vector2::new(0.03, -0.05);
        let prior = WheelCalibrationEstimate {
            offset: Vector2::new(0.02, -0.04),
            scale: 1.05,
            ..WheelCalibrationEstimate::identity()
        };
        let mut fixes = fixes(offset, 0.99, &UnitQuaternion::identity(), 5);
        // One localization (in the turn) is 1 m off.
        fixes[30].1.translation.x += 1.0;
        let mut calibration =
            WheelCalibration::without_vio(WheelCalibrationConfig::default()).with_prior(prior);
        let mut pending = fixes.iter().peekable();
        let mut checked_before_turn = false;
        for (index, pose) in drive().into_iter().enumerate() {
            let t = wheel_stamp(index);
            // The localizations come in before the wheel pose after them.
            while let Some((fix_t, fix)) = pending.next_if(|(fix_t, _)| *fix_t <= t) {
                calibration.add_fix(*fix_t, fix);
            }
            calibration.add_wheel(t, &wheel_pose(pose));
            // After 2 m straight on, the scale and the level are known; the
            // offset, which no turn has shown yet, stays the prior's.
            if !checked_before_turn && calibration.fix_turned_deg() > 0.5 {
                checked_before_turn = true;
                let estimate = calibration.estimate().expect("2 m of fixes");
                assert_eq!(estimate.offset_from, CalibrationSource::Map);
                assert!((estimate.offset - Vector2::new(0.02, -0.04)).norm() < 1e-3);
                assert_eq!(estimate.scale_from, CalibrationSource::Map);
                assert!((estimate.scale - 0.99).abs() < 1e-3, "{}", estimate.scale);
            }
        }
        assert!(checked_before_turn);
        assert!(calibration.fix_turned_deg() > 180.0);
        let estimate = calibration.estimate().expect("estimate");
        assert_eq!(estimate.offset_from, CalibrationSource::Map);
        // The turns show the offset; the prior's pull is a fraction of a mm.
        assert!((estimate.offset - offset).norm() < 3e-4, "{estimate:?}");
        assert!((estimate.scale - 0.99).abs() < 1e-4);
        assert!(estimate.tilt_deg < 1e-6);
        // The localization 1 m off is left out.
        assert_eq!(estimate.fixes, fixes.len() - 1);
    }

    #[test]
    fn the_window_keeps_only_recent_pairs() {
        let config = WheelCalibrationConfig {
            window_s: 5.0,
            ..WheelCalibrationConfig::default()
        };
        let mut calibration = WheelCalibration::new(config, imu_from_body());
        for index in 0..300 {
            let t = index as i64 * 50 * MS;
            let pose = SE3::new(rz(0.0), Vector3::new(index as f64 * 0.02, 0.0, 0.0));
            calibration.add_vio(t, &pose.compose(&imu_from_body().inverse()));
            calibration.add_wheel(t, &pose);
        }
        assert_eq!(calibration.pairs(), 101);
    }
}
