//! Measurements from outside the VIO, for `visloc_vio_node`: map fixes
//! (`map_prior`, from `visloc_map_localize_node`'s `odom_fix`) and a wheel
//! odometry's body velocity (`wheel_velocity`), turned into factors of the
//! Basalt estimator's window (`visloc_basalt::vio::ExternalFactor`).
//!
//! On OpenLORIS-Scene's office sequences the two together lower the error of
//! the VIO corrected into a map by 8-22% (docs/odometry_fusion.md).

use std::collections::VecDeque;

use nalgebra::{Vector3, Vector6};
use visloc_basalt::vio::{ExternalFactor, ExternalMeasurement};
use visloc_core::geometry::SE3;

/// Noise of the external measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalNoise {
    /// 1-sigma position (m) and yaw (rad) of a map fix.
    pub fix_position_m: f64,
    pub fix_orientation_rad: f64,
    /// 1-sigma roll/pitch (rad) of a map fix; 0 leaves them to the IMU.
    /// A monocular VIO without wheel velocity needs 0: a roll/pitch prior
    /// a degree off the IMU's gravity can make its velocity run away.
    pub fix_tilt_rad: f64,
    /// Huber threshold on a fix's whitened residual.
    pub fix_huber: f64,
    /// 1-sigma body velocity (m/s), plus this fraction of the speed.
    pub velocity_m_s: f64,
    pub velocity_per_speed: f64,
}

impl Default for ExternalNoise {
    fn default() -> Self {
        Self {
            fix_position_m: 0.03,
            fix_orientation_rad: 1f64.to_radians(),
            fix_tilt_rad: 1f64.to_radians(),
            fix_huber: 2.0,
            velocity_m_s: 0.02,
            velocity_per_speed: 0.02,
        }
    }
}

/// The recent estimates' header stamps and frame ids, to find the VIO frame
/// a stamped measurement belongs to.
#[derive(Clone, Debug)]
pub struct FrameStamps {
    frames: VecDeque<(i64, u64)>,
    capacity: usize,
}

impl FrameStamps {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Records an estimate: its header stamp and frame id.
    pub fn record(&mut self, header_stamp_ns: i64, frame_id: u64) {
        if self.frames.len() == self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back((header_stamp_ns, frame_id));
    }

    /// Forgets every frame (after an estimator reset).
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// The frame id of the estimate stamped `header_stamp_ns`, within
    /// `tolerance_ns`.
    pub fn frame_at(&self, header_stamp_ns: i64, tolerance_ns: i64) -> Option<u64> {
        self.frames
            .iter()
            .rev()
            .map(|(stamp, frame_id)| ((stamp - header_stamp_ns).abs(), *frame_id))
            .filter(|(gap, _)| *gap <= tolerance_ns)
            .min_by_key(|(gap, _)| *gap)
            .map(|(_, frame_id)| frame_id)
    }
}

/// A map fix as a pose prior on frame `frame_id`: `odom_from_out` is the
/// fix's pose of the output frame (the IMU, or `base_frame`) in the
/// published odometry frame; `out_from_imu` places the IMU in that output
/// frame (identity without `base_frame`), and `odom_from_vio` is the
/// wheel bridge's offset of the published odometry from the VIO's own frame
/// (identity without a bridge).
pub fn map_prior_factor(
    frame_id: u64,
    odom_from_out: &SE3,
    out_from_imu: &SE3,
    odom_from_vio: &SE3,
    noise: &ExternalNoise,
) -> ExternalFactor {
    let vio_from_imu = odom_from_vio
        .inverse()
        .compose(odom_from_out)
        .compose(out_from_imu);
    let p = 1.0 / noise.fix_position_m;
    let yaw = 1.0 / noise.fix_orientation_rad;
    let tilt = if noise.fix_tilt_rad > 0.0 {
        1.0 / noise.fix_tilt_rad
    } else {
        0.0
    };
    ExternalFactor {
        frame_id,
        measurement: ExternalMeasurement::PosePrior {
            world_from_imu: vio_from_imu,
            // The rotation rows are world axes (a left increment): x and y
            // tilt, z yaw.
            sqrt_information: Vector6::new(p, p, p, tilt, tilt, yaw),
            huber: noise.fix_huber,
        },
    }
}

/// A wheel odometry's twist as a body-velocity factor on frame `frame_id`:
/// `velocity_body` is the linear velocity of the wheel odometry's child
/// frame in that frame (`nav_msgs/Odometry` twist), `imu_from_body` places
/// that frame in the IMU's, `imu_angular_velocity` is the IMU's
/// bias-corrected angular velocity (for the lever arm).
pub fn wheel_velocity_factor(
    frame_id: u64,
    imu_from_body: SE3,
    imu_angular_velocity: Vector3<f64>,
    velocity_body: Vector3<f64>,
    noise: &ExternalNoise,
) -> ExternalFactor {
    let sigma = noise.velocity_m_s + noise.velocity_per_speed * velocity_body.norm();
    ExternalFactor {
        frame_id,
        measurement: ExternalMeasurement::BodyVelocity {
            imu_from_body,
            imu_angular_velocity,
            body_velocity: velocity_body,
            sqrt_information: Vector3::repeat(1.0 / sigma),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    fn pose(x: f64, y: f64, z: f64, yaw: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
            Vector3::new(x, y, z),
        )
    }

    #[test]
    fn frame_stamps_find_the_nearest_recent_frame() {
        let mut stamps = FrameStamps::new(3);
        for (stamp, frame) in [(100, 1), (200, 2), (300, 3), (400, 4)] {
            stamps.record(stamp, frame);
        }
        assert_eq!(stamps.frame_at(300, 0), Some(3));
        assert_eq!(stamps.frame_at(305, 10), Some(3));
        assert_eq!(stamps.frame_at(305, 1), None);
        // Evicted.
        assert_eq!(stamps.frame_at(100, 0), None);
        stamps.clear();
        assert_eq!(stamps.frame_at(400, 0), None);
    }

    #[test]
    fn a_fix_is_moved_into_the_vio_frame_and_onto_the_imu() {
        let vio_from_imu = pose(1.0, 2.0, 0.5, 0.3);
        let imu_from_base = pose(0.1, 0.0, -0.2, 0.0);
        let odom_from_vio = pose(-3.0, 1.0, 0.0, 1.0);
        // What the node publishes: the base pose in the bridged odometry.
        let odom_from_base = odom_from_vio.compose(&vio_from_imu).compose(&imu_from_base);
        let factor = map_prior_factor(
            7,
            &odom_from_base,
            &imu_from_base.inverse(),
            &odom_from_vio,
            &ExternalNoise::default(),
        );
        assert_eq!(factor.frame_id, 7);
        let ExternalMeasurement::PosePrior { world_from_imu, .. } = factor.measurement else {
            panic!("a pose prior");
        };
        assert!((world_from_imu.translation - vio_from_imu.translation).norm() < 1e-12);
        assert!(world_from_imu.rotation.angle_to(&vio_from_imu.rotation) < 1e-12);
    }
}
