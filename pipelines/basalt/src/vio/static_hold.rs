//! Holding a VIO still from a static start until the IMU first shows motion.
//!
//! A monocular-inertial VIO barely observes its velocity while the platform
//! stands still. Map fixes then pulling its position by a few centimetres can
//! start it accelerating; on a robot without wheel odometry that run-away
//! is what makes the map prior diverge at the start (docs/odometry_fusion.md).
//! From a static start the platform cannot move before it accelerates or
//! turns, and both show in the IMU, so until then a zero velocity is a
//! measurement, not a guess. [`StaticHold`] watches the IMU and, while
//! nothing has moved yet, gives a zero-velocity factor for each frame. Once
//! it sees motion it stops for good: a platform moving straight at a constant
//! speed looks as quiet to the IMU as one standing still, so a hold later on
//! could pin a moving platform.

use std::collections::VecDeque;

use nalgebra::Vector3;
use visloc_core::geometry::SE3;

use super::{ExternalFactor, ExternalMeasurement};
use crate::{ImuSample, TimestampNs};

/// Watches the IMU from the first sample until the first motion.
#[derive(Debug, Clone)]
pub struct StaticHold {
    /// Motion: any gyro sample above this (rad/s) ...
    pub gyro_threshold_rad_s: f64,
    /// ... or the mean accelerometer sample over `window_ns` moving this far
    /// (m/s^2) from the first window's mean.
    pub accel_threshold_m_s2: f64,
    pub window_ns: TimestampNs,
    /// 1-sigma (m/s) of the zero velocity.
    pub velocity_sigma_m_s: f64,
    first_ns: Option<TimestampNs>,
    start_accel: Option<Vector3<f64>>,
    recent: VecDeque<(TimestampNs, Vector3<f64>)>,
    moved_at_ns: Option<TimestampNs>,
}

impl Default for StaticHold {
    /// 0.05 rad/s, 0.15 m/s^2 over 0.1 s, 0.01 m/s. A T265 standing still on
    /// OpenLORIS-Scene stays under 0.022 rad/s, and its accelerometer's
    /// 0.1 s mean wanders by a few hundredths of a m/s^2.
    fn default() -> Self {
        Self {
            gyro_threshold_rad_s: 0.05,
            accel_threshold_m_s2: 0.15,
            window_ns: 100_000_000,
            velocity_sigma_m_s: 0.01,
            first_ns: None,
            start_accel: None,
            recent: VecDeque::new(),
            moved_at_ns: None,
        }
    }
}

impl StaticHold {
    /// Adds a frame's IMU samples (in time order) and returns whether the
    /// frame is still held, i.e. no motion has been seen up to its samples.
    pub fn observe(&mut self, samples: &[ImuSample]) -> bool {
        if self.moved_at_ns.is_some() {
            return false;
        }
        for sample in samples {
            self.recent
                .push_back((sample.timestamp_ns, sample.accel_m_s2));
            while self
                .recent
                .front()
                .is_some_and(|(t, _)| sample.timestamp_ns - t > self.window_ns)
            {
                self.recent.pop_front();
            }
            let first_ns = *self.first_ns.get_or_insert(sample.timestamp_ns);
            // The accelerometer is compared over full windows only: a single
            // sample is too noisy to set the start or to show motion.
            let accelerated = if sample.timestamp_ns - first_ns >= self.window_ns {
                let mean = self.recent.iter().map(|(_, a)| a).sum::<Vector3<f64>>()
                    / self.recent.len() as f64;
                let start = *self.start_accel.get_or_insert(mean);
                (mean - start).norm() > self.accel_threshold_m_s2
            } else {
                false
            };
            if sample.gyro_rad_s.norm() > self.gyro_threshold_rad_s || accelerated {
                self.moved_at_ns = Some(sample.timestamp_ns);
                return false;
            }
        }
        true
    }

    /// When the first motion was seen, if it was.
    pub const fn moved_at_ns(&self) -> Option<TimestampNs> {
        self.moved_at_ns
    }

    /// The zero-velocity factor for a held frame.
    pub fn factor(&self, frame_id: u64) -> ExternalFactor {
        ExternalFactor {
            frame_id,
            measurement: ExternalMeasurement::BodyVelocity {
                imu_from_body: SE3::identity(),
                imu_angular_velocity: Vector3::zeros(),
                body_velocity: Vector3::zeros(),
                sqrt_information: Vector3::repeat(1.0 / self.velocity_sigma_m_s),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(t_ms: i64, gyro: f64, accel_x: f64) -> ImuSample {
        ImuSample::new(
            t_ms * 1_000_000,
            Vector3::new(0.0, 0.0, gyro),
            Vector3::new(accel_x, 0.0, 9.81),
        )
    }

    #[test]
    fn noise_at_rest_keeps_holding() {
        let mut hold = StaticHold::default();
        for frame in 0..60 {
            let samples: Vec<_> = (0..6)
                .map(|k| {
                    let t = frame * 30 + k * 5;
                    let wobble = if k % 2 == 0 { 0.05 } else { -0.05 };
                    sample(t, 0.01, wobble)
                })
                .collect();
            assert!(hold.observe(&samples), "frame {frame}");
        }
        assert_eq!(hold.moved_at_ns(), None);
    }

    #[test]
    fn a_noisy_first_sample_does_not_end_the_hold() {
        let mut hold = StaticHold::default();
        // One sample 0.4 m/s^2 off, then the rest at rest, as a T265 can give.
        assert!(hold.observe(&[sample(0, 0.0, 0.4)]));
        for t in (5..300).step_by(5) {
            assert!(hold.observe(&[sample(t, 0.0, 0.0)]), "{t} ms");
        }
    }

    #[test]
    fn a_turn_ends_the_hold_for_good() {
        let mut hold = StaticHold::default();
        assert!(hold.observe(&[sample(0, 0.0, 0.0), sample(5, 0.0, 0.0)]));
        assert!(!hold.observe(&[sample(10, 0.2, 0.0)]));
        assert_eq!(hold.moved_at_ns(), Some(10_000_000));
        // Quiet again, as when driving straight: still not held.
        assert!(!hold.observe(&[sample(15, 0.0, 0.0)]));
    }

    #[test]
    fn an_acceleration_ends_the_hold() {
        let mut hold = StaticHold::default();
        for t in (0..200).step_by(5) {
            assert!(hold.observe(&[sample(t, 0.0, 0.0)]));
        }
        // 0.3 m/s^2 forward: the 0.1 s mean passes 0.15 within the window.
        let mut held = true;
        for t in (200..400).step_by(5) {
            held &= hold.observe(&[sample(t, 0.0, 0.3)]);
        }
        assert!(!held);
        let moved = hold.moved_at_ns().unwrap();
        assert!((200_000_000..=300_000_000).contains(&moved), "{moved}");
    }

    #[test]
    fn the_factor_is_a_zero_velocity() {
        let hold = StaticHold::default();
        let factor = hold.factor(7);
        assert_eq!(factor.frame_id, 7);
        let ExternalMeasurement::BodyVelocity {
            body_velocity,
            sqrt_information,
            ..
        } = factor.measurement
        else {
            panic!("a body velocity");
        };
        assert_eq!(body_velocity, Vector3::zeros());
        assert_eq!(sqrt_information, Vector3::repeat(100.0));
    }
}
