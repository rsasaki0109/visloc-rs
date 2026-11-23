//! Keeps `visloc_vio_node`'s odometry going through lost tracking on wheel
//! odometry.
//!
//! Without it, a node that loses tracking resets its estimator and publishes
//! nothing until tracking holds again, in a new odometry frame
//! ([`crate::tracking`]): the robot stops, and map localization starts over.
//! A wheeled robot can do better. Its wheels still measure how it moves
//! while the cameras see nothing useful. [`OdomBridge`] carries the
//! published pose on from the last good VIO pose by the wheel odometry's
//! motion since then. When the new estimator's poses are published again,
//! they are moved by a yaw and a translation so that they continue from
//! where the wheels took the robot. The output frame never jumps, so
//! `map -> odom` stays valid, and no `odom_reset` is needed.
//!
//! All poses are of the same body (the node's `base_frame`, which is also
//! the wheel odometry's child frame) in their odometry frames. Both VIO and
//! the output are gravity-aligned, which keeps the yaw-and-translation
//! offset exact.

use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::SE3;

/// The yaw of a gravity-aligned pose's x axis (rad).
fn yaw(pose: &SE3) -> f64 {
    let x = pose.rotation * Vector3::x();
    x.y.atan2(x.x)
}

/// Where the outputs come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeState {
    /// VIO poses, moved by the current offset.
    Vio,
    /// The wheel odometry's motion since tracking was lost.
    Wheels,
}

#[derive(Clone, Debug)]
struct Bridge {
    out_at_loss: SE3,
    wheel_at_loss: SE3,
}

/// See the module docs.
#[derive(Clone, Debug)]
pub struct OdomBridge {
    /// output <- VIO frame: a rotation about z and a translation.
    offset: SE3,
    bridge: Option<Bridge>,
    last_out: Option<SE3>,
    /// Distance driven on the wheels during the current (or last) outage (m).
    bridged_m: f64,
    last_bridged: Option<SE3>,
}

impl Default for OdomBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl OdomBridge {
    pub fn new() -> Self {
        Self {
            offset: SE3::identity(),
            bridge: None,
            last_out: None,
            bridged_m: 0.0,
            last_bridged: None,
        }
    }

    pub fn state(&self) -> BridgeState {
        if self.bridge.is_some() {
            BridgeState::Wheels
        } else {
            BridgeState::Vio
        }
    }

    /// The offset (output <- VIO frame) VIO poses are moved by.
    pub fn offset(&self) -> &SE3 {
        &self.offset
    }

    /// Distance driven on the wheels in the current (or last) outage (m).
    pub fn bridged_m(&self) -> f64 {
        self.bridged_m
    }

    /// The output pose for a VIO pose. Right after an outage, this sets the
    /// offset that continues the outputs from the last bridged pose.
    pub fn vio(&mut self, vio_pose: &SE3) -> SE3 {
        if self.bridge.take().is_some() {
            if let Some(out) = &self.last_out {
                let rotation =
                    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw(out) - yaw(vio_pose));
                let translation = out.translation - rotation * vio_pose.translation;
                self.offset = SE3::new(rotation, translation);
            }
        }
        let out = self.offset.compose(vio_pose);
        self.last_out = Some(out.clone());
        out
    }

    /// Tracking was lost: from now on [`Self::wheels`] gives the outputs,
    /// from the last output and this wheel pose. Returns false (nothing to
    /// bridge from) before the first output.
    pub fn lost(&mut self, wheel_pose: &SE3) -> bool {
        let Some(out) = self.last_out.clone() else {
            return false;
        };
        if self.bridge.is_none() {
            self.bridge = Some(Bridge {
                out_at_loss: out,
                wheel_at_loss: wheel_pose.clone(),
            });
            self.bridged_m = 0.0;
            self.last_bridged = None;
        }
        true
    }

    /// The output pose for a wheel pose while tracking is lost (None while
    /// VIO gives the outputs).
    pub fn wheels(&mut self, wheel_pose: &SE3) -> Option<SE3> {
        let bridge = self.bridge.as_ref()?;
        let out = bridge
            .out_at_loss
            .compose(&bridge.wheel_at_loss.inverse().compose(wheel_pose));
        if let Some(previous) = &self.last_bridged {
            self.bridged_m += (out.translation - previous.translation).norm();
        }
        self.last_bridged = Some(out.clone());
        self.last_out = Some(out.clone());
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planar(x: f64, y: f64, yaw: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw),
            Vector3::new(x, y, 0.0),
        )
    }

    fn close(a: &SE3, b: &SE3) -> bool {
        (a.translation - b.translation).norm() < 1e-9 && a.rotation.angle_to(&b.rotation) < 1e-9
    }

    #[test]
    fn passes_vio_through_until_tracking_is_lost() {
        let mut bridge = OdomBridge::new();
        let p = planar(1.0, 2.0, 0.3);
        assert!(close(&bridge.vio(&p), &p));
        assert_eq!(bridge.state(), BridgeState::Vio);
        assert!(bridge.wheels(&planar(0.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn nothing_to_bridge_before_the_first_pose() {
        let mut bridge = OdomBridge::new();
        assert!(!bridge.lost(&planar(0.0, 0.0, 0.0)));
        assert_eq!(bridge.state(), BridgeState::Vio);
    }

    #[test]
    fn wheels_carry_the_pose_on_and_vio_continues_from_there() {
        let mut bridge = OdomBridge::new();
        // VIO at (2, 0) facing +y; the wheel odometry has its own frame.
        bridge.vio(&planar(2.0, 0.0, std::f64::consts::FRAC_PI_2));
        let wheel0 = planar(10.0, 5.0, 0.0);
        assert!(bridge.lost(&wheel0));
        // The wheels drive 1 m forward and turn 90 degrees left.
        let wheel1 = wheel0.compose(&planar(1.0, 0.0, std::f64::consts::FRAC_PI_2));
        let out = bridge.wheels(&wheel1).unwrap();
        assert!(close(&out, &planar(2.0, 1.0, std::f64::consts::PI)));
        assert!((bridge.bridged_m() - 0.0).abs() < 1e-12); // first bridged pose: no distance yet
        let wheel2 = wheel1.compose(&planar(0.5, 0.0, 0.0));
        let out2 = bridge.wheels(&wheel2).unwrap();
        assert!(close(&out2, &planar(1.5, 1.0, std::f64::consts::PI)));
        assert!((bridge.bridged_m() - 0.5).abs() < 1e-9);
        // A reset estimator restarts at its origin, in a frame of its own.
        let vio_new = planar(0.0, 0.0, 0.4);
        let resumed = bridge.vio(&vio_new);
        assert!(close(&resumed, &out2));
        assert_eq!(bridge.state(), BridgeState::Vio);
        // Its later motion follows on from there.
        let vio_later = vio_new.compose(&planar(1.0, 0.0, 0.0));
        assert!(close(
            &bridge.vio(&vio_later),
            &out2.compose(&planar(1.0, 0.0, 0.0))
        ));
    }

    #[test]
    fn a_second_loss_while_bridging_keeps_the_first_start() {
        let mut bridge = OdomBridge::new();
        bridge.vio(&planar(0.0, 0.0, 0.0));
        bridge.lost(&planar(0.0, 0.0, 0.0));
        bridge.wheels(&planar(1.0, 0.0, 0.0));
        bridge.lost(&planar(1.0, 0.0, 0.0));
        let out = bridge.wheels(&planar(2.0, 0.0, 0.0)).unwrap();
        assert!(close(&out, &planar(2.0, 0.0, 0.0)));
    }
}
