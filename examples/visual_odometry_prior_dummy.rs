use std::convert::Infallible;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::Frame;
use visloc_rs::{VisualOdometryEstimate, VisualOdometryFrontend, VisualOdometryPriorProvider};

#[derive(Debug, Clone)]
struct FixedVisualOdometryFrontend {
    previous_to_current: SE3,
}

impl VisualOdometryFrontend for FixedVisualOdometryFrontend {
    type Error = Infallible;

    fn estimate_relative_pose(
        &self,
        previous_frame: &Frame,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error> {
        let mut estimate = VisualOdometryEstimate::new(
            previous_frame.id,
            current_frame.id,
            self.previous_to_current.clone(),
        );
        estimate.match_count = 128;
        estimate.inlier_count = 91;
        estimate.mean_reprojection_error = Some(0.82);
        Ok(Some(estimate))
    }
}

fn main() {
    let previous_frame = Frame::new(10, 1);
    let current_frame = Frame::new(11, 1);
    let previous_pose =
        Pose::from_world_to_camera(UnitQuaternion::identity(), -Vector3::new(2.0, 0.0, 0.0));
    let provider = VisualOdometryPriorProvider::new(FixedVisualOdometryFrontend {
        previous_to_current: SE3::new(UnitQuaternion::identity(), Vector3::new(-1.5, 0.0, 0.0)),
    });

    let prior = provider
        .predict_pose_prior(&previous_frame, &previous_pose, &current_frame)
        .expect("fixed frontend cannot fail")
        .expect("fixed frontend returns an estimate");
    let center = prior.pose.camera_center_world();

    println!(
        "vo_prior previous={} current={} matches={} inliers={} mean_error={:?}",
        prior.estimate.previous_frame_id,
        prior.estimate.current_frame_id,
        prior.estimate.match_count,
        prior.estimate.inlier_count,
        prior.estimate.mean_reprojection_error
    );
    println!(
        "predicted_camera_center_world=[{:.3}, {:.3}, {:.3}]",
        center.x, center.y, center.z
    );
}
