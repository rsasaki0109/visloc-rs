//! PnP loop-closure verification demo.
//!
//! Builds a small synthetic scene with two camera poses and a single loop
//! candidate between them, then runs:
//!
//! 1. The classical [`EssentialMatrixLoopClosureVerifier`] on 2D-2D
//!    correspondences (needs an externally supplied translation scale).
//! 2. The new [`PnPLoopClosureVerifier`] on 2D-3D correspondences (returns a
//!    metric relative pose using the keyframe's stored world pose, no scale
//!    parameter).
//!
//! The demo prints the verifier output for both paths so the
//! 2D-2D-with-scale vs 2D-3D-direct trade-off is visible side-by-side.

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    relative_world_to_camera, verify_loop_closure_candidates, verify_loop_closure_candidates_pnp,
    EssentialMatrixLoopClosureVerifier, LocalMappingPipeline, LocalizationPipeline,
    LoopClosureConfig, LoopClosureVerifierConfig, OnlineSlamConfig, OnlineSlamPipeline,
    PnPLoopClosureVerifier, PnPLoopClosureVerifierConfig, Tracker, TrackingConfig,
};

fn map_and_frame(frame_id: u64, camera_center: Vector3<f64>) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
    let points = [
        nalgebra::Point3::new(-1.0, -1.0, 5.0),
        nalgebra::Point3::new(1.0, -1.0, 5.1),
        nalgebra::Point3::new(-1.0, 1.0, 4.9),
        nalgebra::Point3::new(1.0, 1.0, 5.0),
        nalgebra::Point3::new(0.0, 0.0, 5.05),
        nalgebra::Point3::new(0.5, -0.25, 4.95),
        nalgebra::Point3::new(-0.6, 0.4, 4.8),
        nalgebra::Point3::new(0.4, 0.7, 5.2),
        nalgebra::Point3::new(-0.3, -0.6, 4.85),
        nalgebra::Point3::new(0.7, -0.5, 5.3),
        nalgebra::Point3::new(0.0, 0.5, 5.4),
        nalgebra::Point3::new(-0.7, -0.2, 4.7),
    ];
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut frame = Frame::new(frame_id, camera.id);
    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 1.0];
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark_id, landmark);
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }
    (map, frame)
}

fn main() {
    let (map, first_frame) = map_and_frame(10, Vector3::zeros());
    let (_, second_frame) = map_and_frame(30, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();

    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 8,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            ..OnlineSlamConfig::default()
        },
    );
    let _ = slam.process_frame(&first_frame, []);
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(
        second.loop_closure_candidates.len(),
        1,
        "expected exactly one loop candidate"
    );

    println!("# Loop candidate: query=30 matched_keyframe=10 shared landmarks=12");

    println!("\n## Essential-matrix verifier (2D-2D, needs scale)");
    let truth_first = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let truth_second =
        Pose::from_world_to_camera(UnitQuaternion::identity(), -Vector3::new(0.2, 0.0, 0.1));
    let scale = relative_world_to_camera(&truth_first, &truth_second)
        .translation
        .norm();
    let mut essential_candidates = second.loop_closure_candidates.clone();
    let essential_verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: scale,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates(
        &mut essential_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &essential_verifier,
    );
    print_verification("essential", &essential_candidates[0]);

    println!("\n## PnP verifier (2D-3D, metric directly)");
    let pnp_verifier: PnPLoopClosureVerifier = PnPLoopClosureVerifier {
        config: PnPLoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_reprojection_error_px: 4.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates_pnp(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &pnp_verifier,
    );
    print_verification("pnp", &second.loop_closure_candidates[0]);

    let truth_relative = relative_world_to_camera(&truth_first, &truth_second);
    println!(
        "\nTruth relative_t = [{:.3}, {:.3}, {:.3}]",
        truth_relative.translation.x, truth_relative.translation.y, truth_relative.translation.z,
    );
}

fn print_verification(label: &str, candidate: &visloc_rs::LoopClosureCandidate) {
    let verification = candidate.verification.as_ref().expect("verifier was run");
    let relative = verification
        .relative_pose
        .as_ref()
        .map(|p| {
            format!(
                "[{:.3}, {:.3}, {:.3}]",
                p.translation.x, p.translation.y, p.translation.z
            )
        })
        .unwrap_or_else(|| "-".to_string());
    let mean_label = if let Some(px) = verification.mean_reprojection_error_px {
        format!("mean_reproj={px:.4} px")
    } else {
        format!("mean_sampson={:.4}", verification.mean_sampson_error)
    };
    println!(
        "{label}: verified={} inliers={} ratio={:.3} {mean_label} score={:.3} relative_t={relative}",
        verification.verified,
        verification.inlier_count,
        verification.inlier_ratio,
        verification.score,
    );
}
