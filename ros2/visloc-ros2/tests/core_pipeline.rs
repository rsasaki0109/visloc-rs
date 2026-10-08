//! End-to-end tests of the transport-agnostic node cores (no DDS): the same
//! message -> core -> message path the node binaries run, driven
//! synchronously.

mod common;

use common::*;
use visloc_ros2::image::decode_luma;
use visloc_ros2::localize::{diagnostics_msg, LocalizeCore, LocalizeOptions};
use visloc_ros2::msgs::{Header, Time};
use visloc_ros2::sync::VioInputs;
use visloc_ros2::vio::{imu_sample_from_msg, odometry_msg, VioCore, VioFrames, VioOptions};

fn vio_core() -> VioCore {
    let calibration = std::fs::read_to_string(euroc_calibration_path()).unwrap();
    let config = std::fs::read_to_string(euroc_config_path()).unwrap();
    VioCore::from_json(&calibration, &config, &VioOptions::default()).unwrap()
}

#[test]
fn vio_core_tracks_a_static_synthetic_stereo_rig() {
    let mut core = vio_core();
    let stereo = SyntheticStereo::euroc_sized();
    let mut inputs = VioInputs::new(1_000_000, 8, 4, 4000, core.cam_time_offset_ns());
    let frames = VioFrames {
        odom_frame: "odom".into(),
        body_frame: "imu_link".into(),
    };
    let frame_count = 30;
    let mut estimates = Vec::new();
    let mut imu_ns = START_NS - 100 * IMU_PERIOD_NS;
    for frame in 0..frame_count {
        let stamp = START_NS + frame * CAMERA_PERIOD_NS;
        // Sensor messages exactly as a driver would publish them.
        while imu_ns <= stamp {
            let sample = imu_sample_from_msg(&stationary_imu_msg(imu_ns)).unwrap();
            assert!(inputs.push_imu(sample));
            imu_ns += IMU_PERIOD_NS;
        }
        for right in [true, false] {
            let luma = decode_luma(&stereo.image_msg(right, stamp)).unwrap();
            let image = luma.to_basalt().unwrap();
            if right {
                inputs.push_right(stamp, image);
            } else {
                inputs.push_left(stamp, image);
            }
        }
        while let Some(packet) = inputs.pop_ready() {
            let estimate = core.process(packet).expect("VIO step");
            estimates.push(estimate);
        }
    }
    assert_eq!(estimates.len(), frame_count as usize);
    for (index, estimate) in estimates.iter().enumerate() {
        assert_eq!(estimate.frame_id, index as u64);
        let t = estimate.imu_to_world.translation;
        assert!(t.iter().all(|v| v.is_finite()));
        // A static rig must not drift far in 1.5 s.
        assert!(t.norm() < 0.5, "frame {index}: drift {t:?}");
        let odom = odometry_msg(estimate, &frames);
        assert_eq!(
            odom.header.stamp,
            Time::from_nanos(estimate.header_stamp_ns)
        );
    }
    assert!(estimates.iter().any(|e| e.observation_count > 20));
    // Gravity alignment from the IMU: body +x (measured specific force)
    // maps to world +z.
    let up = estimates.last().unwrap().imu_to_world.rotation * nalgebra::Vector3::x();
    assert!(up.z > 0.99, "body x should point up, got {up:?}");
}

#[test]
fn vio_core_rejects_wrong_resolution() {
    let mut core = vio_core();
    let mut inputs = VioInputs::new(0, 4, 4, 100, 0);
    for k in 0..4 {
        inputs.push_imu(stationary_imu_sample(START_NS + k * IMU_PERIOD_NS));
    }
    let small = visloc_basalt::RawU16Image::new(64, 48, vec![0; 64 * 48]).unwrap();
    inputs.push_left(START_NS, small.clone());
    inputs.push_right(START_NS, small);
    let error = core.process(inputs.pop_ready().unwrap()).unwrap_err();
    assert!(!error.needs_reset);
    assert!(error.message.contains("752x480"), "{}", error.message);
}

#[test]
fn localize_core_recovers_a_rendered_query_pose() {
    let core = synthetic_localize_core();
    let msg = visloc_ros2::image::mono8_image(
        Header::new(Time::from_nanos(START_NS), "camera"),
        LOC_WIDTH as u32,
        LOC_HEIGHT as u32,
        loc_query_image(),
    );
    let luma = decode_luma(&msg).unwrap();
    let outcome = core.localize(&luma, None);
    assert!(outcome.success, "{outcome:?}");
    assert!(outcome.inlier_count >= 30, "{outcome:?}");
    let pose = outcome.camera_to_map.as_ref().unwrap();
    let expected_x = expected_query_center_x();
    assert!(
        (pose.translation.x - expected_x).abs() < 0.02
            && pose.translation.y.abs() < 0.02
            && pose.translation.z.abs() < 0.02,
        "camera center {:?}, expected ({expected_x}, 0, 0)",
        pose.translation
    );
    assert!(pose.rotation.angle() < 0.01);
    let pose_msg = core.pose_msg(&outcome, msg.header.stamp, "map").unwrap();
    assert_eq!(pose_msg.header.frame_id, "map");
    assert!((pose_msg.pose.pose.position.x - expected_x).abs() < 0.02);
    let diagnostics = diagnostics_msg(&outcome, msg.header.stamp, "visloc_localize");
    let inliers = diagnostics.status[0]
        .values
        .iter()
        .find(|kv| kv.key == "inliers")
        .unwrap();
    assert_eq!(inliers.value, outcome.inlier_count.to_string());
}

#[test]
fn localize_core_loads_colmap_text_map_from_disk() {
    let options = LocalizeOptions::default();
    let (map, store) = synthetic_map(&options);
    let dir = std::env::temp_dir().join(format!("visloc_ros2_map_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    visloc_io::colmap::write_colmap_text_model(&map, &dir).unwrap();
    let descriptors = write_descriptor_store(&store, &dir.join("landmark_descriptors.txt"));
    let core = LocalizeCore::load(&dir, Some(&descriptors), None, options).unwrap();
    assert_eq!(core.map().landmarks.len(), map.landmarks.len());
    assert_eq!(core.descriptor_count(), store.len());
    let msg = visloc_ros2::image::mono8_image(
        Header::default(),
        LOC_WIDTH as u32,
        LOC_HEIGHT as u32,
        loc_query_image(),
    );
    let outcome = core.localize(&decode_luma(&msg).unwrap(), None);
    assert!(outcome.success, "{outcome:?}");
    // A wrong-size image is reported, not mislocalized.
    let small = visloc_ros2::image::mono8_image(Header::default(), 32, 32, vec![0; 32 * 32]);
    let outcome = core.localize(&decode_luma(&small).unwrap(), None);
    assert!(!outcome.success);
    assert!(outcome.failure.unwrap().contains("320x240"));
    let _ = std::fs::remove_dir_all(&dir);
}
