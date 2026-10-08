//! Over-the-wire tests: the real node binaries talking RTPS/DDS to a
//! `ros2-client` test node on a private ROS domain id.
//!
//! These need working UDP multicast (SPDP discovery) on loopback. They do
//! not need a ROS 2 installation: both sides are RustDDS.

mod common;

use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use common::*;
use visloc_ros2::dds::{output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode};
use visloc_ros2::image::mono8_image;
use visloc_ros2::msgs::{
    DiagnosticArray, Header, Image, Imu, Int32, Odometry, PoseStamped, PoseWithCovarianceStamped,
    Time,
};

/// A per-process, per-test domain id in RustDDS's usable range.
fn domain_id(offset: u16) -> u16 {
    100 + (std::process::id() % 40) as u16 * 3 + offset
}

/// Kills the node process when the test ends (pass or panic).
struct NodeProcess(Child);

impl Drop for NodeProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl NodeProcess {
    fn assert_running(&mut self) {
        if let Ok(Some(status)) = self.0.try_wait() {
            panic!("node exited early with {status}");
        }
    }
}

fn spawn_node(binary: &str, args: &[String]) -> NodeProcess {
    let child = Command::new(binary)
        .args(args)
        .env_remove("ROS_DOMAIN_ID")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn node binary");
    NodeProcess(child)
}

#[test]
fn dds_loopback_pub_sub_between_two_participants() {
    let domain = domain_id(0);
    let mut publisher_node = RosNode::new(domain, "/", "loopback_pub").unwrap();
    let mut subscriber_node = RosNode::new(domain, "/", "loopback_sub").unwrap();
    let publisher = publisher_node
        .publisher::<Int32>("loopback", output_qos(10))
        .unwrap();
    let subscription = subscriber_node
        .subscription::<Int32>("loopback", sensor_qos(10))
        .unwrap();
    let (tx, rx) = mpsc::channel();
    spawn_receiver("loopback-rx", subscription, move |msg: Int32| {
        let _ = tx.send(msg.data);
    })
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut value = 0;
    while Instant::now() < deadline {
        value += 1;
        publish_logged(&publisher, Int32 { data: value }, "loopback");
        if let Ok(received) = rx.recv_timeout(Duration::from_millis(200)) {
            assert!(received >= 1 && received <= value);
            return;
        }
    }
    panic!("no DDS sample received over loopback within 20 s (multicast discovery blocked?)");
}

#[test]
fn vio_node_publishes_odometry_for_synthetic_stereo_imu() {
    let domain = domain_id(1);
    let mut node = spawn_node(
        env!("CARGO_BIN_EXE_visloc_vio_node"),
        &[
            "--calibration".into(),
            euroc_calibration_path().display().to_string(),
            "--config".into(),
            euroc_config_path().display().to_string(),
            "--domain-id".into(),
            domain.to_string(),
            // Exercise remapping on one input.
            "--remap".into(),
            "imu=/imu0".into(),
        ],
    );

    let mut test_node = RosNode::new(domain, "/", "vio_test_driver").unwrap();
    let left_pub = test_node
        .publisher::<Image>("left/image_raw", output_qos(5))
        .unwrap();
    let right_pub = test_node
        .publisher::<Image>("right/image_raw", output_qos(5))
        .unwrap();
    let imu_pub = test_node.publisher::<Imu>("/imu0", output_qos(50)).unwrap();
    let odom_sub = test_node
        .subscription::<Odometry>("odom", sensor_qos(50))
        .unwrap();
    let pose_sub = test_node
        .subscription::<PoseStamped>("pose", sensor_qos(50))
        .unwrap();
    let (odom_tx, odom_rx) = mpsc::channel();
    spawn_receiver("odom-rx", odom_sub, move |msg: Odometry| {
        let _ = odom_tx.send(msg);
    })
    .unwrap();
    let (pose_tx, pose_rx) = mpsc::channel();
    spawn_receiver("pose-rx", pose_sub, move |msg: PoseStamped| {
        let _ = pose_tx.send(msg);
    })
    .unwrap();

    let stereo = SyntheticStereo::euroc_sized();
    let mut odometry = Vec::new();
    let mut poses = Vec::new();
    let mut imu_ns = START_NS - 20 * IMU_PERIOD_NS;
    let started = Instant::now();
    let mut frame = 0i64;
    while started.elapsed() < Duration::from_secs(60) && (odometry.len() < 10 || poses.len() < 10) {
        node.assert_running();
        let stamp = START_NS + frame * CAMERA_PERIOD_NS;
        // IMU slightly ahead of the camera, as real drivers deliver it.
        while imu_ns <= stamp + 2 * IMU_PERIOD_NS {
            publish_logged(&imu_pub, stationary_imu_msg(imu_ns), "imu");
            imu_ns += IMU_PERIOD_NS;
        }
        publish_logged(&left_pub, stereo.image_msg(false, stamp), "left");
        publish_logged(&right_pub, stereo.image_msg(true, stamp), "right");
        frame += 1;
        std::thread::sleep(Duration::from_millis(50));
        odometry.extend(odom_rx.try_iter());
        poses.extend(pose_rx.try_iter());
    }
    assert!(
        odometry.len() >= 10 && poses.len() >= 10,
        "received {} odometry / {} pose messages after {frame} frames",
        odometry.len(),
        poses.len()
    );
    let mut last_stamp = Time::default();
    for msg in &odometry {
        assert_eq!(msg.header.frame_id, "odom");
        assert_eq!(msg.child_frame_id, "imu_link");
        assert!(msg.header.stamp > last_stamp, "stamps must increase");
        last_stamp = msg.header.stamp;
        // Output stamps are the input image stamps.
        let offset = msg.header.stamp.to_nanos() - START_NS;
        assert_eq!(offset % CAMERA_PERIOD_NS, 0);
        // The synthetic "right = shifted left" pair is not geometrically
        // consistent with the real EuRoC extrinsics, so only boundedness is
        // checked here; the tight static-drift bound lives in the
        // deterministic `core_pipeline` test, which processes a fixed,
        // drop-free frame sequence. This test is about the transport.
        let p = msg.pose.pose.position;
        let drift = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        assert!(
            drift.is_finite() && drift < 5.0,
            "static rig drifted {drift} m"
        );
    }
    // PoseStamped and Odometry carry the same estimate for the same stamp.
    let pose = poses.last().unwrap();
    assert_eq!(pose.header.frame_id, "odom");
    if let Some(odom) = odometry
        .iter()
        .find(|o| o.header.stamp == pose.header.stamp)
    {
        assert_eq!(odom.pose.pose.position, pose.pose.position);
        assert_eq!(odom.pose.pose.orientation, pose.pose.orientation);
    }
}

#[test]
fn localize_node_publishes_pose_for_rendered_query() {
    let domain = domain_id(2);
    let dir = std::env::temp_dir().join(format!("visloc_ros2_dds_map_{}", std::process::id()));
    let descriptors = write_synthetic_map(&dir);
    let mut node = spawn_node(
        env!("CARGO_BIN_EXE_visloc_localize_node"),
        &[
            "--ros-args".into(),
            "-p".into(),
            format!("map:={}", dir.display()),
            "-p".into(),
            format!("descriptors:={}", descriptors.display()),
            "-p".into(),
            format!("domain_id:={domain}"),
            "-r".into(),
            "image:=/camera/image_raw".into(),
        ],
    );

    let mut test_node = RosNode::new(domain, "/", "localize_test_driver").unwrap();
    let image_pub = test_node
        .publisher::<Image>("/camera/image_raw", output_qos(2))
        .unwrap();
    let pose_sub = test_node
        .subscription::<PoseWithCovarianceStamped>("pose", sensor_qos(10))
        .unwrap();
    let inlier_sub = test_node
        .subscription::<Int32>("inlier_count", sensor_qos(10))
        .unwrap();
    let diagnostics_sub = test_node
        .subscription::<DiagnosticArray>("/diagnostics", sensor_qos(10))
        .unwrap();
    let (pose_tx, pose_rx) = mpsc::channel();
    spawn_receiver(
        "pose-rx",
        pose_sub,
        move |msg: PoseWithCovarianceStamped| {
            let _ = pose_tx.send(msg);
        },
    )
    .unwrap();
    let (inlier_tx, inlier_rx) = mpsc::channel();
    spawn_receiver("inlier-rx", inlier_sub, move |msg: Int32| {
        let _ = inlier_tx.send(msg.data);
    })
    .unwrap();
    let (diag_tx, diag_rx) = mpsc::channel();
    spawn_receiver("diag-rx", diagnostics_sub, move |msg: DiagnosticArray| {
        let _ = diag_tx.send(msg);
    })
    .unwrap();

    let query = loc_query_image();
    let started = Instant::now();
    let mut pose = None;
    let mut inliers = Vec::new();
    let mut diagnostics = Vec::new();
    let mut sent = 0i64;
    while started.elapsed() < Duration::from_secs(60)
        && (pose.is_none() || inliers.is_empty() || diagnostics.is_empty())
    {
        node.assert_running();
        let stamp = START_NS + sent * CAMERA_PERIOD_NS;
        let msg = mono8_image(
            Header::new(Time::from_nanos(stamp), "camera"),
            LOC_WIDTH as u32,
            LOC_HEIGHT as u32,
            query.clone(),
        );
        publish_logged(&image_pub, msg, "image");
        sent += 1;
        std::thread::sleep(Duration::from_millis(250));
        if let Some(msg) = pose_rx.try_iter().last() {
            pose = Some(msg);
        }
        inliers.extend(inlier_rx.try_iter());
        diagnostics.extend(diag_rx.try_iter());
    }
    let _ = std::fs::remove_dir_all(&dir);
    let pose = pose.unwrap_or_else(|| panic!("no pose after sending {sent} images"));
    assert_eq!(pose.header.frame_id, "map");
    let p = pose.pose.pose.position;
    assert!(
        (p.x - expected_query_center_x()).abs() < 0.02 && p.y.abs() < 0.02 && p.z.abs() < 0.02,
        "pose {p:?}"
    );
    assert!(pose.pose.covariance.0[0] > 0.0);
    assert!(inliers.iter().any(|&n| n >= 30), "inliers {inliers:?}");
    let status = &diagnostics.last().unwrap().status[0];
    assert!(status.values.iter().any(|kv| kv.key == "inliers"));
}
