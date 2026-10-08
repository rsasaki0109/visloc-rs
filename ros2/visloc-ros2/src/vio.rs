//! Transport-agnostic Basalt VIO core of `visloc_vio_node`.
//!
//! Wraps [`BasaltVioEstimatorAdapter`] exactly as
//! `examples/basalt_euroc_online_slam_demo.rs` builds it (same calibration /
//! config JSON contracts, same urgent-keyframe defaults), but is fed from
//! ROS messages instead of the EuRoC reader. No mapper thread is attached:
//! the node runs the lean VIO path (`process_without_marg_data_no_trace`),
//! which the adapter documents as numerically identical to the
//! MargData-retaining path for the trajectory.

use std::collections::VecDeque;

use nalgebra::{UnitQuaternion, Vector3};
use serde_json::json;
use visloc_basalt::{
    config::BasaltConfig, BasaltCalibration, BasaltVioEstimatorAdapter, EurocSensorFrame,
    ImuSample, RawU16Image,
};
use visloc_core::geometry::SE3;

use crate::msgs::{
    Covariance6, Header, Imu, Odometry, Path, Point, Pose, PoseStamped, PoseWithCovariance,
    Quaternion, TfMessage, Time, Transform, TransformStamped, Twist, TwistWithCovariance,
    Vector3 as RosVector3,
};
use crate::sync::SensorPacket;

/// Options layered on top of the Basalt JSON config (mirrors the demo).
#[derive(Clone, Debug)]
pub struct VioOptions {
    /// Keep the demo's default urgent-keyframe spacing (config values win).
    pub urgent_keyframes: bool,
    /// Seed KLT with the IMU rotation (`config.optical_flow_imu_seed_rotation`).
    pub imu_seed_klt: bool,
}

impl Default for VioOptions {
    fn default() -> Self {
        Self {
            urgent_keyframes: true,
            imu_seed_klt: false,
        }
    }
}

/// One VIO estimate, in Basalt's conventions: world frame gravity-aligned
/// (z up), body frame = IMU.
#[derive(Clone, Debug, PartialEq)]
pub struct VioEstimate {
    pub frame_id: u64,
    /// Frame time in the IMU clock.
    pub stamp_ns: i64,
    /// Original left-image header stamp.
    pub header_stamp_ns: i64,
    pub imu_to_world: SE3,
    pub velocity_world: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
    pub accel_bias: Vector3<f64>,
    /// Latest bias-corrected gyro reading of the frame's interval (body).
    pub angular_velocity_body: Option<Vector3<f64>>,
    pub is_keyframe: bool,
    pub observation_count: usize,
}

/// Error from the VIO core. `recoverable` errors leave the estimator
/// usable; others require [`VioCore::reset`].
#[derive(Debug)]
pub struct VioError {
    pub message: String,
    pub needs_reset: bool,
}

impl std::fmt::Display for VioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VioError {}

/// Parses the two Basalt JSON inputs and builds the estimator.
pub struct VioCore {
    calibration: BasaltCalibration,
    config: BasaltConfig,
    adapter: BasaltVioEstimatorAdapter,
    next_frame_id: u64,
}

impl VioCore {
    /// Builds the core from the contents of Basalt's calibration JSON
    /// (`{"value0": {...}}`) and VIO config JSON.
    pub fn from_json(
        calibration_json: &str,
        config_json: &str,
        options: &VioOptions,
    ) -> Result<Self, String> {
        let calibration = BasaltCalibration::from_json_str(calibration_json)
            .map_err(|error| format!("calibration JSON: {error}"))?;
        if calibration.cameras.len() < 2 || calibration.resolutions.len() < 2 {
            return Err(format!(
                "stereo VIO needs two cameras in the calibration, found {}",
                calibration.cameras.len()
            ));
        }
        let mut config = BasaltConfig::from_json(config_json)
            .map_err(|error| format!("config JSON: {error:?}"))?;
        if options.imu_seed_klt {
            config.values.insert(
                "config.optical_flow_imu_seed_rotation".to_string(),
                json!(true),
            );
        }
        if options.urgent_keyframes {
            // Same defaults as `basalt_euroc_online_slam_demo` (config wins).
            for (key, value) in [
                ("config.vio_urgent_kf_keypoints_thresh", json!(0.5)),
                ("config.vio_urgent_min_frames_after_kf", json!(2)),
            ] {
                config.values.entry(key.to_string()).or_insert(value);
            }
        }
        let adapter = BasaltVioEstimatorAdapter::from_config(&calibration, &config)
            .map_err(|error| format!("estimator setup: {error}"))?;
        Ok(Self {
            calibration,
            config,
            adapter,
            next_frame_id: 0,
        })
    }

    pub const fn calibration(&self) -> &BasaltCalibration {
        &self.calibration
    }

    /// `(width, height)` the calibration expects for camera `index`.
    pub fn expected_resolution(&self, index: usize) -> Option<(usize, usize)> {
        self.calibration
            .resolutions
            .get(index)
            .map(|(w, h)| (*w as usize, *h as usize))
    }

    /// `cam_time_offset_ns` from the calibration (camera -> IMU clock).
    pub const fn cam_time_offset_ns(&self) -> i64 {
        self.calibration.cam_time_offset_ns
    }

    /// Drops all estimator state; the next frame re-initializes from IMU.
    pub fn reset(&mut self) -> Result<(), String> {
        self.adapter = BasaltVioEstimatorAdapter::from_config(&self.calibration, &self.config)
            .map_err(|error| format!("estimator reset: {error}"))?;
        Ok(())
    }

    /// Runs one stereo frame through frontend + estimator.
    pub fn process(&mut self, packet: SensorPacket<RawU16Image>) -> Result<VioEstimate, VioError> {
        for (index, image) in [&packet.left, &packet.right].into_iter().enumerate() {
            if let Some(expected) = self.expected_resolution(index) {
                let actual = (image.width(), image.height());
                if actual != expected {
                    return Err(VioError {
                        message: format!(
                            "cam{index} image is {}x{}, calibration expects {}x{}",
                            actual.0, actual.1, expected.0, expected.1
                        ),
                        needs_reset: false,
                    });
                }
            }
        }
        let angular_velocity = packet.imu.last().map(|sample| sample.gyro_rad_s);
        let frame_id = self.next_frame_id;
        let frame = EurocSensorFrame {
            frame_id,
            timestamp_ns: packet.stamp_ns,
            cam0: packet.left,
            cam1: Some(packet.right),
            cam0_path: Default::default(),
            cam1_path: None,
            imu: packet.imu,
            initialization_imu: packet.initialization_imu,
        };
        let output = self
            .adapter
            .process_without_marg_data_no_trace(frame)
            .map_err(|error| VioError {
                message: format!("Basalt VIO: {error}"),
                needs_reset: true,
            })?;
        self.next_frame_id += 1;
        let state = &output.estimator.state;
        Ok(VioEstimate {
            frame_id,
            stamp_ns: packet.stamp_ns,
            header_stamp_ns: packet.header_stamp_ns,
            imu_to_world: state.imu_to_world.clone(),
            velocity_world: state.velocity_world_m_s,
            gyro_bias: state.gyro_bias_rad_s,
            accel_bias: state.accel_bias_m_s2,
            angular_velocity_body: angular_velocity.map(|gyro| gyro - state.gyro_bias_rad_s),
            is_keyframe: output.estimator.is_keyframe,
            observation_count: output.tracks.observations.len(),
        })
    }
}

/// Converts a `sensor_msgs/Imu` into Basalt's sample (stamp in ns).
pub fn imu_sample_from_msg(msg: &Imu) -> Option<ImuSample> {
    let gyro = Vector3::new(
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
    );
    let accel = Vector3::new(
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z,
    );
    if !gyro
        .iter()
        .chain(accel.iter())
        .all(|value| value.is_finite())
    {
        return None;
    }
    Some(ImuSample::new(msg.header.stamp.to_nanos(), gyro, accel))
}

/// ROS frame names used on the VIO outputs.
#[derive(Clone, Debug)]
pub struct VioFrames {
    /// Fixed odometry frame (`header.frame_id`), e.g. `odom`.
    pub odom_frame: String,
    /// Body frame (`child_frame_id`), Basalt's IMU frame, e.g. `imu_link`.
    pub body_frame: String,
}

pub fn ros_pose(pose: &SE3) -> Pose {
    let q: &UnitQuaternion<f64> = &pose.rotation;
    Pose {
        position: Point {
            x: pose.translation.x,
            y: pose.translation.y,
            z: pose.translation.z,
        },
        orientation: Quaternion {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        },
    }
}

fn ros_vector(v: &Vector3<f64>) -> RosVector3 {
    RosVector3 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}

/// Builds the node's outputs for one estimate.
pub fn odometry_msg(estimate: &VioEstimate, frames: &VioFrames) -> Odometry {
    let header = Header::new(
        Time::from_nanos(estimate.header_stamp_ns),
        frames.odom_frame.clone(),
    );
    // nav_msgs/Odometry's twist is expressed in the child (body) frame.
    let rotation_world_from_body = estimate.imu_to_world.rotation;
    let velocity_body = rotation_world_from_body.inverse() * estimate.velocity_world;
    Odometry {
        header,
        child_frame_id: frames.body_frame.clone(),
        pose: PoseWithCovariance {
            pose: ros_pose(&estimate.imu_to_world),
            // Basalt's marginalized window does not expose a calibrated
            // per-frame pose covariance; publish zeros ("unknown") rather
            // than a made-up value.
            covariance: Covariance6::default(),
        },
        twist: TwistWithCovariance {
            twist: Twist {
                linear: ros_vector(&velocity_body),
                angular: estimate
                    .angular_velocity_body
                    .as_ref()
                    .map(ros_vector)
                    .unwrap_or_default(),
            },
            covariance: Covariance6::default(),
        },
    }
}

pub fn pose_stamped_msg(estimate: &VioEstimate, frames: &VioFrames) -> PoseStamped {
    PoseStamped {
        header: Header::new(
            Time::from_nanos(estimate.header_stamp_ns),
            frames.odom_frame.clone(),
        ),
        pose: ros_pose(&estimate.imu_to_world),
    }
}

pub fn tf_msg(estimate: &VioEstimate, frames: &VioFrames) -> TfMessage {
    let pose = ros_pose(&estimate.imu_to_world);
    TfMessage {
        transforms: vec![TransformStamped {
            header: Header::new(
                Time::from_nanos(estimate.header_stamp_ns),
                frames.odom_frame.clone(),
            ),
            child_frame_id: frames.body_frame.clone(),
            transform: Transform {
                translation: RosVector3 {
                    x: pose.position.x,
                    y: pose.position.y,
                    z: pose.position.z,
                },
                rotation: pose.orientation,
            },
        }],
    }
}

/// A bounded trajectory for `nav_msgs/Path` (drop-oldest beyond `max_len`).
#[derive(Debug)]
pub struct PathAccumulator {
    poses: VecDeque<PoseStamped>,
    max_len: usize,
}

impl PathAccumulator {
    pub fn new(max_len: usize) -> Self {
        Self {
            poses: VecDeque::new(),
            max_len: max_len.max(1),
        }
    }

    pub fn push(&mut self, pose: PoseStamped) {
        self.poses.push_back(pose);
        while self.poses.len() > self.max_len {
            self.poses.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.poses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    pub fn clear(&mut self) {
        self.poses.clear();
    }

    pub fn msg(&self, frame_id: &str) -> Path {
        let stamp = self
            .poses
            .back()
            .map(|pose| pose.header.stamp)
            .unwrap_or_default();
        Path {
            header: Header::new(stamp, frame_id),
            poses: self.poses.iter().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate() -> VioEstimate {
        VioEstimate {
            frame_id: 3,
            stamp_ns: 2_000_000_007,
            header_stamp_ns: 2_000_000_000,
            imu_to_world: SE3 {
                rotation: UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
                translation: Vector3::new(1.0, 2.0, 3.0),
            },
            velocity_world: Vector3::new(0.0, 1.0, 0.0),
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
            angular_velocity_body: Some(Vector3::new(0.0, 0.0, 0.5)),
            is_keyframe: true,
            observation_count: 10,
        }
    }

    fn frames() -> VioFrames {
        VioFrames {
            odom_frame: "odom".into(),
            body_frame: "imu_link".into(),
        }
    }

    #[test]
    fn odometry_twist_is_in_body_frame() {
        let odom = odometry_msg(&estimate(), &frames());
        assert_eq!(odom.header.frame_id, "odom");
        assert_eq!(odom.child_frame_id, "imu_link");
        assert_eq!(odom.header.stamp, Time { sec: 2, nanosec: 0 });
        assert!((odom.pose.pose.position.y - 2.0).abs() < 1e-12);
        // Body is yawed +90 deg: world +y velocity is body +x.
        assert!((odom.twist.twist.linear.x - 1.0).abs() < 1e-12);
        assert!(odom.twist.twist.linear.y.abs() < 1e-12);
        assert!((odom.twist.twist.angular.z - 0.5).abs() < 1e-12);
        let q = odom.pose.pose.orientation;
        assert!((q.z - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((q.w - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn path_is_bounded() {
        let mut path = PathAccumulator::new(2);
        for _ in 0..5 {
            path.push(pose_stamped_msg(&estimate(), &frames()));
        }
        assert_eq!(path.len(), 2);
        assert_eq!(path.msg("odom").poses.len(), 2);
        let tf = tf_msg(&estimate(), &frames());
        assert_eq!(tf.transforms[0].child_frame_id, "imu_link");
    }

    #[test]
    fn imu_conversion_rejects_non_finite() {
        let mut msg = Imu::default();
        msg.header.stamp = Time { sec: 1, nanosec: 5 };
        msg.linear_acceleration.z = 9.81;
        let sample = imu_sample_from_msg(&msg).unwrap();
        assert_eq!(sample.timestamp_ns, 1_000_000_005);
        msg.angular_velocity.x = f64::NAN;
        assert!(imu_sample_from_msg(&msg).is_none());
    }
}
