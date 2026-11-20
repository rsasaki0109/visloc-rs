//! Message builders for `visloc_nav_sim_node`: a simulated robot
//! ([`visloc_navigation::sim_robot::SimRobot`]) published the way a robot
//! driver plus a localization node present a robot to Nav2.
//!
//! - `odom` (`nav_msgs/Odometry`) and `/tf` `odom -> base_link`: the wheel
//!   odometry, as a robot base driver publishes it;
//! - `/tf` `map -> odom`: the localization (REP-105), as AMCL would, once
//!   known;
//! - `scan` (`sensor_msgs/LaserScan`): the depth camera's horizontal slice,
//!   for Nav2's obstacle layers;
//! - `ground_truth` (`nav_msgs/Odometry` in `map`) and `map_pose`
//!   (`geometry_msgs/PoseStamped`, the believed pose): for evaluation.

use std::fmt::Write as _;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_navigation::sim::{Pose2, ScanConfig};
use visloc_navigation::sim_robot::StepOutput;

use crate::msgs::{
    Covariance6, Header, LaserScan, Odometry, Point, Pose, PoseStamped, PoseWithCovariance,
    Quaternion, TfMessage, Time, Transform, TransformStamped, Twist, TwistWithCovariance,
    Vector3 as RosVector3,
};

/// Frame ids of the published messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavSimFrames {
    pub map: String,
    pub odom: String,
    pub base: String,
}

impl Default for NavSimFrames {
    fn default() -> Self {
        Self {
            map: "map".into(),
            odom: "odom".into(),
            base: "base_link".into(),
        }
    }
}

fn quaternion(yaw: f64) -> Quaternion {
    let q = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw);
    Quaternion {
        x: q.i,
        y: q.j,
        z: q.k,
        w: q.w,
    }
}

/// A planar pose as a `geometry_msgs/Pose`.
pub fn ros_pose2(pose: &Pose2) -> Pose {
    Pose {
        position: Point {
            x: pose.x,
            y: pose.y,
            z: 0.0,
        },
        orientation: quaternion(pose.yaw),
    }
}

/// The yaw of a `geometry_msgs/Quaternion`.
pub fn yaw_of(q: &Quaternion) -> f64 {
    (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z))
}

fn transform(stamp: Time, parent: &str, child: &str, pose: &Pose2) -> TransformStamped {
    TransformStamped {
        header: Header::new(stamp, parent),
        child_frame_id: child.to_owned(),
        transform: Transform {
            translation: RosVector3 {
                x: pose.x,
                y: pose.y,
                z: 0.0,
            },
            rotation: quaternion(pose.yaw),
        },
    }
}

fn odometry(stamp: Time, frame: &str, child: &str, pose: &Pose2, v: f64, w: f64) -> Odometry {
    Odometry {
        header: Header::new(stamp, frame),
        child_frame_id: child.to_owned(),
        pose: PoseWithCovariance {
            pose: ros_pose2(pose),
            covariance: Covariance6::diagonal(1e-3, 1e-3),
        },
        twist: TwistWithCovariance {
            twist: Twist {
                linear: RosVector3 { x: v, y: 0.0, z: 0.0 },
                angular: RosVector3 { x: 0.0, y: 0.0, z: w },
            },
            covariance: Covariance6::diagonal(1e-3, 1e-3),
        },
    }
}

/// The wheel odometry message (`odom -> base_link`, velocities in the base).
pub fn odom_msg(stamp: Time, frames: &NavSimFrames, step: &StepOutput) -> Odometry {
    odometry(stamp, &frames.odom, &frames.base, &step.odom, step.linear, step.angular)
}

/// The true pose in the map frame, for evaluation.
pub fn ground_truth_msg(stamp: Time, frames: &NavSimFrames, step: &StepOutput) -> Odometry {
    odometry(stamp, &frames.map, &frames.base, &step.ground_truth, 0.0, 0.0)
}

/// The pose the robot believes (`map -> odom -> base_link`), once known.
pub fn map_pose_msg(stamp: Time, frames: &NavSimFrames, step: &StepOutput) -> Option<PoseStamped> {
    step.estimate().map(|pose| PoseStamped {
        header: Header::new(stamp, &frames.map),
        pose: ros_pose2(&pose),
    })
}

/// `/tf`: `odom -> base_link`, and `map -> odom` once known.
pub fn tf_msg(stamp: Time, frames: &NavSimFrames, step: &StepOutput) -> TfMessage {
    let mut transforms = vec![transform(stamp, &frames.odom, &frames.base, &step.odom)];
    if let Some(map_from_odom) = &step.map_from_odom {
        transforms.push(transform(stamp, &frames.map, &frames.odom, map_from_odom));
    }
    TfMessage { transforms }
}

/// A scan from [`visloc_navigation::sim::World::scan`]: beams beyond range
/// are `+inf` (REP-117), for an obstacle layer with `inf_is_valid` to clear.
pub fn scan_msg(stamp: Time, frame: &str, config: &ScanConfig, ranges: &[Option<f64>]) -> LaserScan {
    let increment = if ranges.len() > 1 {
        config.fov / (ranges.len() - 1) as f64
    } else {
        0.0
    };
    LaserScan {
        header: Header::new(stamp, frame),
        angle_min: (-config.fov / 2.0) as f32,
        angle_max: (config.fov / 2.0) as f32,
        angle_increment: increment as f32,
        time_increment: 0.0,
        scan_time: 0.0,
        range_min: 0.05,
        range_max: config.max_range as f32,
        ranges: ranges
            .iter()
            .map(|range| range.map_or(f32::INFINITY, |r| r as f32))
            .collect(),
        intensities: Vec::new(),
    }
}

/// One CSV row of the node's `record` file: time, true pose, believed pose
/// (empty until localized), collisions so far.
pub fn record_row(step: &StepOutput, collisions: usize) -> String {
    let t = &step.ground_truth;
    let mut row = format!("{:.3},{:.4},{:.4},{:.4},", step.time_s, t.x, t.y, t.yaw);
    match step.estimate() {
        Some(e) => write!(row, "{:.4},{:.4},{:.4}", e.x, e.y, e.yaw),
        None => write!(row, ",,"),
    }
    .expect("string write");
    write!(row, ",{collisions}").expect("string write");
    row
}

/// Header of [`record_row`]'s CSV.
pub const RECORD_HEADER: &str = "time_s,true_x,true_y,true_yaw,est_x,est_y,est_yaw,collisions";

/// An axis-aligned box `x0,y0,x1,y1,height` (an obstacle the map lacks).
pub fn parse_box(text: &str) -> Result<((f64, f64), (f64, f64), f64), String> {
    let values = text
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("bad box `{text}`: want x0,y0,x1,y1,height"))?;
    match values[..] {
        [x0, y0, x1, y1, h] if x1 > x0 && y1 > y0 && h > 0.0 => Ok(((x0, y0), (x1, y1), h)),
        _ => Err(format!("bad box `{text}`: want x0,y0,x1,y1,height with x1 > x0, y1 > y0, height > 0")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step() -> StepOutput {
        StepOutput {
            time_s: 1.5,
            ground_truth: Pose2::new(1.0, 2.0, 0.5),
            odom: Pose2::new(0.5, 0.1, 0.4),
            linear: 0.3,
            angular: -0.2,
            map_from_odom: None,
            localization: None,
        }
    }

    #[test]
    fn tf_carries_map_to_odom_only_once_known() {
        let frames = NavSimFrames::default();
        let mut s = step();
        let stamp = Time::from_nanos(5);
        assert_eq!(tf_msg(stamp, &frames, &s).transforms.len(), 1);
        assert!(map_pose_msg(stamp, &frames, &s).is_none());
        s.map_from_odom = Some(Pose2::new(1.0, 0.0, 0.1));
        let tf = tf_msg(stamp, &frames, &s);
        assert_eq!(tf.transforms[1].header.frame_id, "map");
        assert_eq!(tf.transforms[1].child_frame_id, "odom");
        let pose = map_pose_msg(stamp, &frames, &s).unwrap().pose;
        let expected = Pose2::new(1.0, 0.0, 0.1).compose(&Pose2::new(0.5, 0.1, 0.4));
        assert!((pose.position.x - expected.x).abs() < 1e-12);
        assert!((yaw_of(&pose.orientation) - expected.yaw).abs() < 1e-12);
        assert!(record_row(&s, 2).ends_with(",2"));
    }

    #[test]
    fn scan_marks_missing_beams_infinite() {
        let config = ScanConfig::default();
        let scan = scan_msg(Time::default(), "base_link", &config, &[Some(1.0), None, Some(2.5)]);
        assert_eq!(scan.ranges[1], f32::INFINITY);
        assert!((scan.angle_min + scan.angle_increment * 2.0 - scan.angle_max).abs() < 1e-6);
        assert!(parse_box("4.6,4.0,5.2,5.0,0.8").is_ok());
        assert!(parse_box("1,2,0,3,1").is_err());
    }
}
