//! Wheel odometry with its heading from a gyro.
//!
//! A wheeled robot's wheels slip in turns in place, and their odometry's
//! heading drifts with every turn: on OpenLORIS-Scene's corridor by 15 deg
//! in one 350 deg turn, 44 deg over 4.5 minutes. The distance the wheels
//! measure stays good. [`GyroWheelOdometry`] keeps the wheels' travel and
//! turns it by the heading a gyro integrates instead: about the vertical,
//! which the accelerometer gives, so the IMU may sit anywhere on the robot
//! in any orientation. Where the wheels stand still, the heading holds and
//! the gyro's rate about the vertical is its bias. On that corridor run the
//! heading stays within 8.4 deg of the ground truth.

use std::collections::VecDeque;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;

#[derive(Debug, Clone, PartialEq)]
pub struct GyroWheelConfig {
    /// The wheels stand still where two consecutive poses differ by less
    /// than this (m) ...
    pub still_m: f64,
    /// ... and this (deg).
    pub still_deg: f64,
    /// Gyro samples count towards the bias and the vertical once the wheels
    /// have stood still this long (s), past a stop's own settling.
    pub settle_s: f64,
    /// The bias and the vertical average over this many such samples at
    /// most, forgetting older ones.
    pub average_samples: usize,
    /// Between wheel poses with a gyro gap longer than this (s), or before
    /// the first gyro sample, the wheels' own turn is used.
    pub max_gyro_gap_s: f64,
}

impl Default for GyroWheelConfig {
    fn default() -> Self {
        Self {
            still_m: 0.0005,
            still_deg: 0.01,
            settle_s: 0.5,
            average_samples: 4_000,
            max_gyro_gap_s: 0.1,
        }
    }
}

/// A running mean over at most `cap` samples (older ones decay).
#[derive(Debug, Clone, Copy)]
struct RunningMean<T> {
    mean: T,
    count: usize,
}

impl<T> RunningMean<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<f64, Output = T>,
{
    fn add(&mut self, value: T, cap: usize) {
        self.count = (self.count + 1).min(cap.max(1));
        self.mean = self.mean + (value - self.mean) * (1.0 / self.count as f64);
    }
}

#[derive(Debug, Clone, Copy)]
struct GyroSample {
    timestamp_ns: i64,
    /// The heading integrated so far (rad).
    heading: f64,
    /// The rate about the vertical (rad/s), bias included.
    rate: f64,
    acceleration: Vector3<f64>,
}

/// Fuses wheel odometry poses (its point, level) with a gyro and an
/// accelerometer (any frame fixed to the robot), in time order of each.
#[derive(Debug, Clone)]
pub struct GyroWheelOdometry {
    config: GyroWheelConfig,
    /// The gyro samples since the last wheel pose (and the one before it).
    gyro: VecDeque<GyroSample>,
    /// The accelerometer's mean at standstill, and over every sample until
    /// the first standstill.
    up_still: RunningMean<Vector3<f64>>,
    up_any: RunningMean<Vector3<f64>>,
    bias: RunningMean<f64>,
    /// The last wheel pose, its stamp, the integrated heading used there,
    /// and since when the wheels stand still.
    last: Option<(i64, SE3, Option<f64>, Option<i64>)>,
    /// The fused pose at the last wheel pose: position and heading.
    position: Vector3<f64>,
    heading: f64,
}

impl GyroWheelOdometry {
    pub fn new(config: GyroWheelConfig) -> Self {
        Self {
            config,
            gyro: VecDeque::new(),
            up_still: RunningMean {
                mean: Vector3::zeros(),
                count: 0,
            },
            up_any: RunningMean {
                mean: Vector3::zeros(),
                count: 0,
            },
            bias: RunningMean {
                mean: 0.0,
                count: 0,
            },
            last: None,
            position: Vector3::zeros(),
            heading: 0.0,
        }
    }

    /// The gyro's bias about the vertical (rad/s), once a standstill gave it.
    pub fn bias(&self) -> Option<f64> {
        (self.bias.count > 0).then_some(self.bias.mean)
    }

    /// The vertical (up) in the IMU's frame, from the accelerometer.
    pub fn up(&self) -> Option<Vector3<f64>> {
        let mean = if self.up_still.count > 0 {
            self.up_still.mean
        } else {
            self.up_any.mean
        };
        mean.try_normalize(1e-9)
    }

    /// Adds an IMU sample: angular velocity (rad/s) and specific force
    /// (m/s^2) in the IMU's frame.
    pub fn add_imu(
        &mut self,
        timestamp_ns: i64,
        angular_velocity: &Vector3<f64>,
        linear_acceleration: &Vector3<f64>,
    ) {
        if self
            .gyro
            .back()
            .is_some_and(|last| timestamp_ns <= last.timestamp_ns)
        {
            return;
        }
        if self.up_still.count == 0 {
            self.up_any
                .add(*linear_acceleration, self.config.average_samples);
        }
        let Some(up) = self.up() else {
            return;
        };
        let rate = angular_velocity.dot(&up);
        let heading = match self.gyro.back() {
            Some(last) => {
                let dt = (timestamp_ns - last.timestamp_ns) as f64 * 1e-9;
                if dt > self.config.max_gyro_gap_s {
                    last.heading
                } else {
                    last.heading + (0.5 * (last.rate + rate) - self.bias.mean) * dt
                }
            }
            None => 0.0,
        };
        self.gyro.push_back(GyroSample {
            timestamp_ns,
            heading,
            rate,
            acceleration: *linear_acceleration,
        });
        self.prune();
    }

    /// Drops the gyro samples the next wheel pose no longer needs: those
    /// before the last wheel pose, but the one just before it.
    fn prune(&mut self) {
        let Some((keep_from, ..)) = self.last.as_ref() else {
            return;
        };
        while self.gyro.len() > 2 && self.gyro[1].timestamp_ns <= *keep_from {
            self.gyro.pop_front();
        }
    }

    /// The integrated heading at `timestamp_ns`, if the gyro covers it
    /// (interpolated, or extrapolated by at most `max_gyro_gap_s`).
    fn heading_at(&self, timestamp_ns: i64) -> Option<f64> {
        let max_gap_ns = (self.config.max_gyro_gap_s * 1e9) as i64;
        let index = self
            .gyro
            .partition_point(|sample| sample.timestamp_ns < timestamp_ns);
        match (
            index.checked_sub(1).map(|i| self.gyro[i]),
            self.gyro.get(index),
        ) {
            (_, Some(after)) if after.timestamp_ns == timestamp_ns => Some(after.heading),
            (Some(before), Some(after))
                if after.timestamp_ns - before.timestamp_ns <= max_gap_ns =>
            {
                let alpha = (timestamp_ns - before.timestamp_ns) as f64
                    / (after.timestamp_ns - before.timestamp_ns) as f64;
                Some(before.heading + alpha * (after.heading - before.heading))
            }
            (Some(before), None) if timestamp_ns - before.timestamp_ns <= max_gap_ns => Some(
                before.heading
                    + (before.rate - self.bias.mean)
                        * (timestamp_ns - before.timestamp_ns) as f64
                        * 1e-9,
            ),
            _ => None,
        }
    }

    /// The fused pose for a wheel odometry pose (its point, level), in the
    /// wheel odometry's frame: the first one as given, then the wheels'
    /// travel turned by the gyro's heading.
    pub fn wheel(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) -> SE3 {
        let yaw = yaw_of(&odom_from_wheel.rotation);
        let gyro_heading = self.heading_at(timestamp_ns);
        match self.last.take() {
            None => {
                self.position = odom_from_wheel.translation;
                self.heading = yaw;
                self.last = Some((timestamp_ns, odom_from_wheel.clone(), gyro_heading, None));
            }
            Some((last_ns, last_pose, last_gyro, still_since)) => {
                if timestamp_ns <= last_ns {
                    self.last = Some((last_ns, last_pose, last_gyro, still_since));
                    return self.pose();
                }
                let step = last_pose.rotation.inverse()
                    * (odom_from_wheel.translation - last_pose.translation);
                let turn = wrap_angle(yaw - yaw_of(&last_pose.rotation));
                let still = step.norm() < self.config.still_m
                    && turn.abs() < self.config.still_deg.to_radians();
                // Standing still, the heading holds; else the gyro's turn,
                // or the wheels' where the gyro does not cover the step.
                let max_gap_ns = (self.config.max_gyro_gap_s * 1e9) as i64;
                let gapless = self
                    .gyro
                    .iter()
                    .zip(self.gyro.iter().skip(1))
                    .all(|(a, b)| b.timestamp_ns - a.timestamp_ns <= max_gap_ns);
                let delta = if still {
                    0.0
                } else {
                    match (last_gyro, gyro_heading) {
                        (Some(from), Some(to)) if gapless => to - from,
                        _ => turn,
                    }
                };
                self.position +=
                    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), self.heading) * step;
                self.heading = wrap_angle(self.heading + delta);
                let still_since = if still {
                    Some(still_since.unwrap_or(last_ns))
                } else {
                    None
                };
                // Settled at a standstill, the gyro's rate about the vertical
                // over this step is its bias, and the accelerometer the
                // vertical.
                if still_since
                    .is_some_and(|since| last_ns - since >= (self.config.settle_s * 1e9) as i64)
                {
                    let cap = self.config.average_samples;
                    for sample in self.gyro.iter().filter(|sample| {
                        sample.timestamp_ns > last_ns && sample.timestamp_ns <= timestamp_ns
                    }) {
                        self.bias.add(sample.rate, cap);
                        self.up_still.add(sample.acceleration, cap);
                    }
                }
                self.last = Some((
                    timestamp_ns,
                    odom_from_wheel.clone(),
                    gyro_heading,
                    still_since,
                ));
            }
        }
        self.prune();
        self.pose()
    }

    fn pose(&self) -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), self.heading),
            self.position,
        )
    }
}

fn yaw_of(rotation: &UnitQuaternion<f64>) -> f64 {
    let matrix = rotation.to_rotation_matrix();
    matrix[(1, 0)].atan2(matrix[(0, 0)])
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    /// Stamp, true pose, wheel odometry pose, gyro, accelerometer.
    type Sample = (i64, SE3, SE3, Vector3<f64>, Vector3<f64>);

    /// A run sampled at 200 Hz. The wheels count 5 % more of every turn than
    /// the robot makes; the IMU is mounted tilted and turned, its gyro
    /// `bias` rad/s off.
    fn run(bias: f64) -> Vec<Sample> {
        let mount = UnitQuaternion::from_euler_angles(1.4, -0.3, 2.0);
        let (mut truth, mut wheels) = ((0.0, 0.0, 0.3), (0.0, 0.0, 0.3));
        let mut samples = Vec::new();
        let mut index = 0;
        // (seconds, speed m/s, turn rad/s): stand, ahead, a half turn in
        // place, ahead, a curve, stand, ahead.
        for (seconds, speed, turn) in [
            (1.0, 0.0, 0.0),
            (4.0, 0.5, 0.0),
            (3.0, 0.0, std::f64::consts::PI / 3.0),
            (4.0, 0.5, 0.0),
            (3.0, 0.4, std::f64::consts::FRAC_PI_6),
            (1.5, 0.0, 0.0),
            (2.0, 0.5, 0.0),
        ] {
            for _ in 0..(seconds * 200.0) as usize {
                let dt = 0.005;
                for (pose, gain) in [(&mut truth, 1.0), (&mut wheels, 1.05)] {
                    pose.0 += speed * f64::cos(pose.2) * dt;
                    pose.1 += speed * f64::sin(pose.2) * dt;
                    pose.2 += turn * gain * dt;
                }
                let at = |(x, y, yaw): (f64, f64, f64)| {
                    SE3::new(
                        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
                        Vector3::new(x, y, 0.0),
                    )
                };
                let gyro = mount.inverse() * Vector3::new(0.0, 0.0, turn + bias);
                let accel = mount.inverse() * Vector3::new(0.0, 0.0, 9.81);
                index += 1;
                samples.push((index * 5 * MS, at(truth), at(wheels), gyro, accel));
            }
        }
        samples
    }

    /// Feeds the IMU at 200 Hz and the wheels at 20 Hz; the true, the
    /// wheels' and the fused pose at the last wheel pose.
    fn fuse(bias: f64, imu: bool) -> (SE3, SE3, SE3, GyroWheelOdometry) {
        let mut odometry = GyroWheelOdometry::new(GyroWheelConfig::default());
        let mut last = None;
        for (index, (t, truth, wheels, gyro, accel)) in run(bias).into_iter().enumerate() {
            if imu {
                odometry.add_imu(t, &gyro, &accel);
            }
            if index % 10 == 0 {
                let fused = odometry.wheel(t, &wheels);
                last = Some((truth, wheels, fused));
            }
        }
        let (truth, wheels, fused) = last.expect("wheel poses");
        (truth, wheels, fused, odometry)
    }

    fn heading_error_deg(a: &SE3, b: &SE3) -> f64 {
        wrap_angle(yaw_of(&a.rotation) - yaw_of(&b.rotation)).to_degrees()
    }

    #[test]
    fn the_gyro_keeps_the_heading_the_wheels_lose_in_turns() {
        let bias = 0.004; // 14 deg a minute
        let (truth, wheels, fused, odometry) = fuse(bias, true);
        // The wheels alone end 13.5 deg and 0.59 m off.
        assert!(heading_error_deg(&truth, &wheels).abs() > 13.0);
        assert!((truth.translation - wheels.translation).norm() > 0.5);
        // The gyro's bias comes from the first standstill, the vertical
        // from the accelerometer, and the heading and the position stay
        // with the truth.
        assert!((odometry.bias().expect("bias") - bias).abs() < 1e-9);
        let mount = UnitQuaternion::from_euler_angles(1.4, -0.3, 2.0);
        assert!((mount * odometry.up().expect("up") - Vector3::z()).norm() < 1e-9);
        // (0.8 mm: the wheels' step taken in their frame at its start.)
        assert!(heading_error_deg(&truth, &fused).abs() < 0.01, "{fused:?}");
        assert!(
            (truth.translation - fused.translation).norm() < 0.002,
            "{fused:?}"
        );
    }

    #[test]
    fn without_a_gyro_the_wheels_pass_through() {
        let (_, wheels, fused, odometry) = fuse(0.0, false);
        assert_eq!(odometry.bias(), None);
        assert!(heading_error_deg(&wheels, &fused).abs() < 1e-9);
        assert!((wheels.translation - fused.translation).norm() < 1e-9);
    }

    #[test]
    fn across_a_gyro_gap_the_wheels_turn() {
        let mut odometry = GyroWheelOdometry::new(GyroWheelConfig::default());
        let up = Vector3::new(0.0, 0.0, 9.81);
        let turn = |yaw: f64| {
            SE3::new(
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
                Vector3::zeros(),
            )
        };
        odometry.add_imu(0, &Vector3::new(0.0, 0.0, 1.0), &up);
        odometry.wheel(0, &turn(0.0));
        // The gyro stops for 0.5 s while the wheels turn 0.5 rad.
        odometry.add_imu(500 * MS, &Vector3::new(0.0, 0.0, 1.0), &up);
        let fused = odometry.wheel(500 * MS, &turn(0.5));
        assert!((yaw_of(&fused.rotation) - 0.5).abs() < 1e-9);
    }
}
