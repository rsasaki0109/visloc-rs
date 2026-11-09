//! The runtime-dispatched motion model (`DemoMotionModel`) and its CLI kind.

use super::*;

/// Runtime-dispatched motion model. Both inner models implement
/// [`MotionModel`], but `Tracker<P, M>` is generic in `M` and Rust's
/// monomorphisation forces a single concrete `M` for the lifetime of
/// the `Tracker`. Wrap the two stock models in an enum so the demo
/// can pick between them via a CLI flag without duplicating the
/// downstream pipeline construction.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(super) enum DemoMotionModel {
    Pose(ConstantPoseMotionModel),
    Velocity(ConstantVelocityMotionModel),
    ImuPredictive(ImuPredictiveMotionModel),
    AdaptiveImuPose(AdaptiveImuPoseMotionModel),
}

#[cfg(feature = "image-io")]
impl DemoMotionModel {
    /// Forward a raw body-frame IMU sample into the wrapped motion
    /// model. No-op for `Pose` / `Velocity`; the IMU-predictive variant
    /// buffers the sample for the next `predict_pose` call.
    pub(super) fn push_imu_measurement(
        &mut self,
        gyro: Vector3<f64>,
        accel: Vector3<f64>,
        dt: f64,
    ) {
        match self {
            DemoMotionModel::ImuPredictive(inner) => inner.push_imu_measurement(gyro, accel, dt),
            DemoMotionModel::AdaptiveImuPose(inner) => {
                inner.imu_mut().push_imu_measurement(gyro, accel, dt);
            }
            _ => {}
        }
    }

    /// Refresh `velocity_world` on the IMU-predictive variant from two
    /// successive observed camera poses. The IMU strapdown
    /// integrator's initial-velocity slot is otherwise only updated by
    /// a downstream VI-BA pass (not exercised in this demo); without
    /// this hook `velocity_world` stays at zero forever and the
    /// per-frame position prediction only captures the quadratic accel
    /// term, systematically under-predicting the body's true motion.
    /// No-op for the `Pose` / `Velocity` variants — they maintain
    /// their own state via the standard `observe` path.
    pub(super) fn update_velocity_from_pose_diff(
        &mut self,
        prev: &Pose,
        curr: &Pose,
        dt_seconds: f64,
    ) {
        match self {
            DemoMotionModel::ImuPredictive(inner) => {
                inner.update_velocity_from_camera_pose_difference(prev, curr, dt_seconds);
            }
            DemoMotionModel::AdaptiveImuPose(inner) => {
                inner
                    .imu_mut()
                    .update_velocity_from_camera_pose_difference(prev, curr, dt_seconds);
            }
            _ => {}
        }
    }

    /// Push the local-VI-BA-refined per-keyframe `(velocity_world,
    /// bias_gyro, bias_acc)` into the IMU-predictive variant's
    /// integration state, matching the documented contract on
    /// `ImuPredictiveMotionModel::set_velocity_world` /
    /// `set_biases`. No-op for the `Pose` / `Velocity` variants.
    pub(super) fn mirror_vi_ba_state(
        &mut self,
        velocity_world: Vector3<f64>,
        bias_gyro: Vector3<f64>,
        bias_acc: Vector3<f64>,
    ) {
        match self {
            DemoMotionModel::ImuPredictive(inner) => {
                inner.set_velocity_world(velocity_world);
                inner.set_biases(bias_gyro, bias_acc);
            }
            DemoMotionModel::AdaptiveImuPose(inner) => {
                let imu = inner.imu_mut();
                imu.set_velocity_world(velocity_world);
                imu.set_biases(bias_gyro, bias_acc);
            }
            _ => {}
        }
    }

    /// `Some((switches_to_pose, switches_to_imu,
    /// velocity_refreshes_on_switch_to_imu, current_mode))` when the
    /// wrapped model is the Phase-23 #4 / Phase-24 adaptive variant.
    /// `None` otherwise — the summary line then reports `n/a`.
    pub(super) fn adaptive_stats(&self) -> Option<(u64, u64, u64, AdaptiveMotionMode)> {
        match self {
            DemoMotionModel::AdaptiveImuPose(inner) => Some((
                inner.switches_to_pose(),
                inner.switches_to_imu(),
                inner.velocity_refreshes_on_switch_to_imu(),
                inner.mode(),
            )),
            _ => None,
        }
    }
}

#[cfg(feature = "image-io")]
impl MotionModel for DemoMotionModel {
    fn predict_pose(
        &self,
        frame: &Frame,
        last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        match self {
            DemoMotionModel::Pose(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
            DemoMotionModel::Velocity(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
            DemoMotionModel::ImuPredictive(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
            DemoMotionModel::AdaptiveImuPose(inner) => {
                inner.predict_pose(frame, last_result, last_successful_pose)
            }
        }
    }

    fn observe(&mut self, result: &TrackingResult) {
        match self {
            DemoMotionModel::Pose(inner) => inner.observe(result),
            DemoMotionModel::Velocity(inner) => inner.observe(result),
            DemoMotionModel::ImuPredictive(inner) => inner.observe(result),
            DemoMotionModel::AdaptiveImuPose(inner) => inner.observe(result),
        }
    }

    fn allows_pnp_pose_prior_warm_start(&self) -> bool {
        match self {
            DemoMotionModel::Pose(inner) => inner.allows_pnp_pose_prior_warm_start(),
            DemoMotionModel::Velocity(inner) => inner.allows_pnp_pose_prior_warm_start(),
            DemoMotionModel::ImuPredictive(inner) => inner.allows_pnp_pose_prior_warm_start(),
            DemoMotionModel::AdaptiveImuPose(inner) => inner.allows_pnp_pose_prior_warm_start(),
        }
    }

    fn reset(&mut self) {
        match self {
            DemoMotionModel::Pose(inner) => inner.reset(),
            DemoMotionModel::Velocity(inner) => inner.reset(),
            DemoMotionModel::ImuPredictive(inner) => inner.reset(),
            DemoMotionModel::AdaptiveImuPose(inner) => inner.reset(),
        }
    }
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MotionModelKind {
    Pose,
    Velocity,
    ImuPredictive,
    AdaptiveImuPose,
}
