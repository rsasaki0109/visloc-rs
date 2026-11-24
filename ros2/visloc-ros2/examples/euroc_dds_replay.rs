//! Replays a EuRoC-layout sequence over DDS and records the nodes' outputs,
//! for running `visloc_vio_node` and `visloc_map_localize_node` on real data
//! without a ROS 2 installation (both sides are RustDDS).
//!
//! ```text
//! cargo run --release --example euroc_dds_replay -- --euroc-dir data/office1-6 \
//!   --domain-id 42 --rate 0.25 \
//!   [--odometry data/office1-6/odom.tum --imu-from-body tx,ty,tz,qx,qy,qz,qw] \
//!   [--record-map-pose map_pose.tum] [--record-odom odom.tum]
//! ```
//!
//! Publishes `cam0/image_raw`, `cam1/image_raw` (mono8), `imu0`
//! (`sensor_msgs/Imu`) and, with `--odometry` (a TUM file of the wheel
//! odometry's body, e.g. `base_link`), `wheel/odom` (`nav_msgs/Odometry`,
//! frame `wheel_odom`, child `base_link`, twist from the poses around each
//! one) at the odometry's own stamps, plus the `/tf_static` transform
//! `imu_link -> base_link` from `--imu-from-body`. `--rate` slows playback
//! (0.25: a quarter of real time). Records `map_pose` and `odom` as TUM
//! files (IMU or body poses, as the nodes publish them).

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use nalgebra::{Quaternion as NQuaternion, UnitQuaternion, Vector3};
use visloc_basalt::euroc::{read_camera_image, read_camera_manifest};
use visloc_core::geometry::SE3;
use visloc_ros2::dds::{
    latched_qos, output_qos, publish_logged, sensor_qos, spawn_receiver, RosNode,
};
use visloc_ros2::image::mono8_image;
use visloc_ros2::msgs::{
    Header, Imu, Odometry, Pose, PoseWithCovarianceStamped, Quaternion, TfMessage, Time, Transform,
    TransformStamped, Vector3 as RosVector3,
};
use visloc_ros2::vio::ros_pose;

struct Args {
    euroc_dir: PathBuf,
    domain_id: u16,
    rate: f64,
    odometry: Option<PathBuf>,
    imu_from_body: SE3,
    record_map_pose: Option<PathBuf>,
    record_odom: Option<PathBuf>,
    tail_s: f64,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        euroc_dir: PathBuf::new(),
        domain_id: 42,
        rate: 0.25,
        odometry: None,
        imu_from_body: SE3::identity(),
        record_map_pose: None,
        record_odom: None,
        tail_s: 5.0,
    };
    let mut euroc = None;
    let mut it = std::env::args().skip(1);
    while let Some(option) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{option} needs a value"));
        match option.as_str() {
            "--euroc-dir" => euroc = Some(PathBuf::from(value()?)),
            "--domain-id" => args.domain_id = value()?.parse().map_err(|e| format!("{e}"))?,
            "--rate" => args.rate = value()?.parse().map_err(|e| format!("{e}"))?,
            "--odometry" => args.odometry = Some(PathBuf::from(value()?)),
            "--imu-from-body" => {
                let v: Vec<f64> = value()?
                    .split(',')
                    .map(|x| x.parse::<f64>().map_err(|e| format!("{e}")))
                    .collect::<Result<_, _>>()?;
                if v.len() != 7 {
                    return Err("--imu-from-body: tx,ty,tz,qx,qy,qz,qw".into());
                }
                args.imu_from_body = SE3::new(
                    UnitQuaternion::from_quaternion(NQuaternion::new(v[6], v[3], v[4], v[5])),
                    Vector3::new(v[0], v[1], v[2]),
                );
            }
            "--record-map-pose" => args.record_map_pose = Some(PathBuf::from(value()?)),
            "--record-odom" => args.record_odom = Some(PathBuf::from(value()?)),
            "--tail-s" => args.tail_s = value()?.parse().map_err(|e| format!("{e}"))?,
            other => return Err(format!("unknown option {other}")),
        }
    }
    args.euroc_dir = euroc.ok_or("--euroc-dir is required")?;
    Ok(args)
}

fn read_imu(path: &Path) -> Result<Vec<(i64, [f64; 6])>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        let t: i64 = f[0].parse().map_err(|e| format!("{e}"))?;
        let mut v = [0.0; 6];
        for (k, x) in v.iter_mut().enumerate() {
            *x = f[k + 1].parse().map_err(|e| format!("{e}"))?;
        }
        rows.push((t, v));
    }
    Ok(rows)
}

fn read_tum(path: &Path) -> Result<Vec<(i64, SE3)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let (seconds, fraction) = f[0].split_once('.').unwrap_or((f[0], ""));
        let fraction = format!("{fraction:0<9}");
        let t = seconds.parse::<i64>().map_err(|e| format!("{e}"))? * 1_000_000_000
            + fraction[..9].parse::<i64>().map_err(|e| format!("{e}"))?;
        let v: Vec<f64> = f[1..8]
            .iter()
            .map(|x| x.parse::<f64>().map_err(|e| format!("{e}")))
            .collect::<Result<_, _>>()?;
        rows.push((
            t,
            SE3::new(
                UnitQuaternion::from_quaternion(NQuaternion::new(v[6], v[3], v[4], v[5])),
                Vector3::new(v[0], v[1], v[2]),
            ),
        ));
    }
    rows.sort_by_key(|(t, _)| *t);
    Ok(rows)
}

fn tum_line(stamp_ns: i64, pose: &Pose) -> String {
    format!(
        "{}.{:09} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}\n",
        stamp_ns.div_euclid(1_000_000_000),
        stamp_ns.rem_euclid(1_000_000_000),
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    )
}

/// The wheel odometry message at `index`: pose, and the body twist from the
/// poses before and after it.
fn wheel_msg(poses: &[(i64, SE3)], index: usize) -> Odometry {
    let (t, pose) = &poses[index];
    let before = &poses[index.saturating_sub(1)];
    let after = &poses[(index + 1).min(poses.len() - 1)];
    let dt = (after.0 - before.0) as f64 * 1e-9;
    let (linear, angular) = if dt > 0.0 {
        let world = (after.1.translation - before.1.translation) / dt;
        let body = pose.rotation.inverse() * world;
        let turn = (before.1.rotation.inverse() * after.1.rotation).scaled_axis() / dt;
        (body, turn)
    } else {
        (Vector3::zeros(), Vector3::zeros())
    };
    let mut msg = Odometry {
        header: Header::new(Time::from_nanos(*t), "wheel_odom"),
        child_frame_id: "base_link".into(),
        ..Odometry::default()
    };
    msg.pose.pose = ros_pose(pose);
    msg.twist.twist.linear = RosVector3 {
        x: linear.x,
        y: linear.y,
        z: linear.z,
    };
    msg.twist.twist.angular = RosVector3 {
        x: angular.x,
        y: angular.y,
        z: angular.z,
    };
    msg
}

fn run() -> Result<(), String> {
    let args = parse()?;
    let mav0 = args.euroc_dir.join("mav0");
    let cam0 = read_camera_manifest(mav0.join("cam0")).map_err(|e| e.to_string())?;
    let cam1 = read_camera_manifest(mav0.join("cam1")).map_err(|e| e.to_string())?;
    let imu = read_imu(&mav0.join("imu0/data.csv"))?;
    let wheel = args.odometry.as_deref().map(read_tum).transpose()?;

    let mut node = RosNode::new(args.domain_id, "/", "euroc_dds_replay")?;
    let left_pub = node.publisher::<visloc_ros2::msgs::Image>("cam0/image_raw", output_qos(5))?;
    let right_pub = node.publisher::<visloc_ros2::msgs::Image>("cam1/image_raw", output_qos(5))?;
    let imu_pub = node.publisher::<Imu>("imu0", output_qos(400))?;
    let wheel_pub = node.publisher::<Odometry>("wheel/odom", output_qos(50))?;
    let tf_static_pub = node.publisher::<TfMessage>("/tf_static", latched_qos(1))?;

    let (pose_tx, pose_rx) = mpsc::channel::<String>();
    let map_pose_sub =
        node.subscription::<PoseWithCovarianceStamped>("map_pose", sensor_qos(50))?;
    spawn_receiver(
        "map-pose-rx",
        map_pose_sub,
        move |msg: PoseWithCovarianceStamped| {
            let _ = pose_tx.send(tum_line(msg.header.stamp.to_nanos(), &msg.pose.pose));
        },
    )?;
    let (odom_tx, odom_rx) = mpsc::channel::<String>();
    let odom_sub = node.subscription::<Odometry>("odom", sensor_qos(50))?;
    spawn_receiver("odom-rx", odom_sub, move |msg: Odometry| {
        let _ = odom_tx.send(tum_line(msg.header.stamp.to_nanos(), &msg.pose.pose));
    })?;

    let body = &args.imu_from_body;
    let q = body.rotation.quaternion();
    publish_logged(
        &tf_static_pub,
        TfMessage {
            transforms: vec![TransformStamped {
                header: Header::new(Time::from_nanos(cam0[0].timestamp_ns), "imu_link"),
                child_frame_id: "base_link".into(),
                transform: Transform {
                    translation: RosVector3 {
                        x: body.translation.x,
                        y: body.translation.y,
                        z: body.translation.z,
                    },
                    rotation: Quaternion {
                        x: q.i,
                        y: q.j,
                        z: q.k,
                        w: q.w,
                    },
                },
            }],
        },
        "tf_static",
    );
    // Let discovery settle before the first frame.
    std::thread::sleep(Duration::from_secs(3));

    // Merge the streams by stamp.
    let mut events: VecDeque<(i64, u8, usize)> = VecDeque::new();
    let mut all: Vec<(i64, u8, usize)> = Vec::new();
    all.extend(imu.iter().enumerate().map(|(i, (t, _))| (*t, 0u8, i)));
    all.extend(
        cam0.iter()
            .enumerate()
            .map(|(i, e)| (e.timestamp_ns, 1u8, i)),
    );
    if let Some(wheel) = &wheel {
        all.extend(wheel.iter().enumerate().map(|(i, (t, _))| (*t, 2u8, i)));
    }
    all.sort();
    events.extend(all);
    let start_stamp = events.front().map_or(0, |e| e.0);
    let started = Instant::now();
    let mut frames = 0usize;
    while let Some((stamp, kind, index)) = events.pop_front() {
        let due = Duration::from_secs_f64((stamp - start_stamp) as f64 * 1e-9 / args.rate);
        if let Some(wait) = due.checked_sub(started.elapsed()) {
            std::thread::sleep(wait);
        }
        match kind {
            0 => {
                let (_, v) = imu[index];
                let msg = Imu {
                    header: Header::new(Time::from_nanos(stamp), "imu_link"),
                    angular_velocity: RosVector3 {
                        x: v[0],
                        y: v[1],
                        z: v[2],
                    },
                    linear_acceleration: RosVector3 {
                        x: v[3],
                        y: v[4],
                        z: v[5],
                    },
                    ..Imu::default()
                };
                publish_logged(&imu_pub, msg, "imu");
            }
            1 => {
                for (manifest, publisher, frame) in
                    [(&cam0, &left_pub, "cam0"), (&cam1, &right_pub, "cam1")]
                {
                    let Some(entry) = manifest.iter().find(|e| e.timestamp_ns == stamp) else {
                        continue;
                    };
                    let image = read_camera_image(&entry.path).map_err(|e| e.to_string())?;
                    let data = image.pixels().iter().map(|p| (p >> 8) as u8).collect();
                    publish_logged(
                        publisher,
                        mono8_image(
                            Header::new(Time::from_nanos(stamp), frame),
                            image.width() as u32,
                            image.height() as u32,
                            data,
                        ),
                        frame,
                    );
                }
                frames += 1;
                if frames.is_multiple_of(100) {
                    eprintln!("euroc_dds_replay: {frames}/{} frames", cam0.len());
                }
            }
            _ => {
                let wheel = wheel.as_ref().expect("wheel events only with --odometry");
                publish_logged(&wheel_pub, wheel_msg(wheel, index), "wheel");
            }
        }
    }
    std::thread::sleep(Duration::from_secs_f64(args.tail_s));
    for (path, rx) in [
        (&args.record_map_pose, pose_rx),
        (&args.record_odom, odom_rx),
    ] {
        let mut text = String::from("# timestamp tx ty tz qx qy qz qw\n");
        let mut count = 0;
        for line in rx.try_iter() {
            text.push_str(&line);
            count += 1;
        }
        if let Some(path) = path {
            std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut note = String::new();
            let _ = write!(note, "{count} poses -> {}", path.display());
            eprintln!("euroc_dds_replay: {note}");
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("euroc_dds_replay: {error}");
        std::process::exit(1);
    }
}
