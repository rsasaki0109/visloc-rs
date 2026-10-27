//! Static transforms from `/tf_static`, for the extrinsics of a robot's
//! sensors.
//!
//! On a robot, where the camera and the IMU sit relative to the robot's base
//! comes from its description (URDF via `robot_state_publisher`, or
//! `static_transform_publisher`), published once on `/tf_static` with
//! transient-local durability. [`StaticTransforms`] keeps the latest
//! transform per child frame and composes chains of them like `tf2` does,
//! so a node can ask for, e.g., `base_link <- cam0` however many links lie
//! between them. Only static transforms are kept: there is no time
//! interpolation.

use std::collections::HashMap;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;

use crate::msgs::{TfMessage, Transform};

/// `tf2` ignores a leading `/` in frame names.
fn frame_name(name: &str) -> &str {
    name.strip_prefix('/').unwrap_or(name)
}

/// `T_parent_child` of a `geometry_msgs/Transform`.
pub fn se3_from_transform(transform: &Transform) -> SE3 {
    let (t, q) = (&transform.translation, &transform.rotation);
    SE3::new(
        UnitQuaternion::from_quaternion(Quaternion::new(q.w, q.x, q.y, q.z)),
        Vector3::new(t.x, t.y, t.z),
    )
}

/// A forest of static transforms, keyed by child frame.
#[derive(Debug, Clone, Default)]
pub struct StaticTransforms {
    /// `child -> (parent, T_parent_child)`.
    parents: HashMap<String, (String, SE3)>,
}

impl StaticTransforms {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces the transforms of one `/tf_static` message.
    pub fn insert(&mut self, message: &TfMessage) {
        for transform in &message.transforms {
            let child = frame_name(&transform.child_frame_id);
            let parent = frame_name(&transform.header.frame_id);
            if child.is_empty() || parent.is_empty() || child == parent {
                continue;
            }
            self.parents.insert(
                child.to_owned(),
                (parent.to_owned(), se3_from_transform(&transform.transform)),
            );
        }
    }

    pub fn len(&self) -> usize {
        self.parents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parents.is_empty()
    }

    /// `frame` and each of its ancestors with `T_ancestor_frame`, nearest
    /// first; stops at a loop.
    fn ancestors(&self, frame: &str) -> Vec<(String, SE3)> {
        let mut chain = vec![(frame.to_owned(), SE3::identity())];
        while let Some((parent, parent_from_child)) = self.parents.get(&chain[chain.len() - 1].0) {
            if chain.iter().any(|(name, _)| name == parent) {
                break;
            }
            let ancestor_from_frame = parent_from_child.compose(&chain[chain.len() - 1].1);
            chain.push((parent.clone(), ancestor_from_frame));
        }
        chain
    }

    /// `T_target_source`, which takes points in `source` to `target`; `None`
    /// when the two frames are not connected.
    pub fn lookup(&self, target: &str, source: &str) -> Option<SE3> {
        let (target, source) = (frame_name(target), frame_name(source));
        let target_chain = self.ancestors(target);
        for (ancestor, ancestor_from_source) in self.ancestors(source) {
            if let Some((_, ancestor_from_target)) =
                target_chain.iter().find(|(name, _)| *name == ancestor)
            {
                return Some(
                    ancestor_from_target
                        .inverse()
                        .compose(&ancestor_from_source),
                );
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msgs::{Header, Quaternion as RosQuaternion, Time, TransformStamped, Vector3 as V3};

    fn stamped(parent: &str, child: &str, pose: &SE3) -> TransformStamped {
        let q = pose.rotation.quaternion();
        TransformStamped {
            header: Header::new(Time::from_nanos(0), parent.to_owned()),
            child_frame_id: child.to_owned(),
            transform: Transform {
                translation: V3 {
                    x: pose.translation.x,
                    y: pose.translation.y,
                    z: pose.translation.z,
                },
                rotation: RosQuaternion {
                    x: q.i,
                    y: q.j,
                    z: q.k,
                    w: q.w,
                },
            },
        }
    }

    fn pose(roll: f64, pitch: f64, yaw: f64, x: f64, y: f64, z: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(roll, pitch, yaw),
            Vector3::new(x, y, z),
        )
    }

    fn assert_close(actual: &SE3, expected: &SE3) {
        let error = expected.inverse().compose(actual);
        assert!(
            error.translation.norm() < 1e-12,
            "{actual:?} vs {expected:?}"
        );
        assert!(error.rotation.angle() < 1e-12, "{actual:?} vs {expected:?}");
    }

    /// `base_link -> imu_link`, `base_link -> camera_link -> cam0`, and an
    /// unrelated `world -> marker`.
    fn tree() -> (StaticTransforms, SE3, SE3, SE3) {
        let base_from_imu = pose(0.0, 0.0, 1.2, 0.1, 0.0, 0.3);
        let base_from_link = pose(0.0, 0.1, 0.0, 0.25, 0.05, 0.4);
        let link_from_cam = pose(
            -std::f64::consts::FRAC_PI_2,
            0.0,
            -std::f64::consts::FRAC_PI_2,
            0.0,
            0.0,
            0.0,
        );
        let mut transforms = StaticTransforms::new();
        transforms.insert(&TfMessage {
            transforms: vec![
                stamped("base_link", "imu_link", &base_from_imu),
                stamped("/base_link", "camera_link", &base_from_link),
            ],
        });
        transforms.insert(&TfMessage {
            transforms: vec![
                stamped("camera_link", "cam0", &link_from_cam),
                stamped("world", "marker", &pose(0.0, 0.0, 0.0, 1.0, 2.0, 3.0)),
            ],
        });
        (transforms, base_from_imu, base_from_link, link_from_cam)
    }

    #[test]
    fn chains_compose_up_down_and_across() {
        let (transforms, base_from_imu, base_from_link, link_from_cam) = tree();
        assert_eq!(transforms.len(), 4);
        let base_from_cam = base_from_link.compose(&link_from_cam);
        assert_close(
            &transforms.lookup("base_link", "cam0").expect("up"),
            &base_from_cam,
        );
        assert_close(
            &transforms.lookup("cam0", "base_link").expect("down"),
            &base_from_cam.inverse(),
        );
        assert_close(
            &transforms.lookup("imu_link", "cam0").expect("across"),
            &base_from_imu.inverse().compose(&base_from_cam),
        );
        assert_close(
            &transforms.lookup("/cam0", "cam0").expect("self"),
            &SE3::identity(),
        );
    }

    #[test]
    fn unconnected_or_unknown_frames_have_no_transform() {
        let (transforms, ..) = tree();
        assert!(transforms.lookup("base_link", "marker").is_none());
        assert!(transforms.lookup("base_link", "lidar").is_none());
    }

    #[test]
    fn a_republished_transform_replaces_the_old_one() {
        let (mut transforms, ..) = tree();
        let moved = pose(0.0, 0.0, 0.0, 0.5, 0.0, 0.0);
        transforms.insert(&TfMessage {
            transforms: vec![stamped("base_link", "imu_link", &moved)],
        });
        assert_close(
            &transforms.lookup("base_link", "imu_link").expect("tf"),
            &moved,
        );
    }

    #[test]
    fn a_loop_does_not_hang() {
        let mut transforms = StaticTransforms::new();
        let step = pose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        transforms.insert(&TfMessage {
            transforms: vec![stamped("a", "b", &step), stamped("b", "a", &step)],
        });
        assert!(transforms.lookup("a", "c").is_none());
        assert!(transforms.lookup("a", "b").is_some());
    }
}
