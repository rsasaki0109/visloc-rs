//! Replays a EuRoC recording through `visloc_vio_node`'s VIO stages and its
//! online-mapper session (`mapper:=true`) without DDS, and writes what the
//! node would publish and save:
//!
//! - `trajectory_map_pose.tum`: every frame's live `map_pose`
//!   (`T_map_odom` at that moment times the VIO pose);
//! - `trajectory_vio.tum`: the VIO (`odom`) poses;
//! - `trajectory_mapper.tum`: the full-frame trajectory written at shutdown
//!   (`mapper_trajectory_out`).
//!
//! ```text
//! cargo run --release --manifest-path ros2/visloc-ros2/Cargo.toml \
//!   --example euroc_mapper_replay -- \
//!   --euroc-dir /data/V1_01_easy \
//!   --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
//!   --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
//!   --out-dir target/ros2_mapper_v101
//! ```
//!
//! Frames are paced at the recording's rate (as a live node receives them)
//! unless `--as-fast-as-possible` is given; the node's mapper queue drops
//! packets when it is full, so the pace matters. Poses are the IMU (body)
//! frame, stamped with the left image's time in seconds.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{env, fs, process};

use nalgebra::Vector3;
use visloc_basalt::EurocSensorDataset;
use visloc_core::geometry::SE3;
use visloc_ros2::sync::SensorPacket;
use visloc_ros2::vio::{VioCore, VioOptions};
use visloc_ros2::vio_mapper::{new_online_mapper, MapperParams, MapperSession};

struct Args {
    euroc_dir: PathBuf,
    calibration: PathBuf,
    config: PathBuf,
    out_dir: PathBuf,
    max_frames: Option<usize>,
    realtime: bool,
    optimize_every_k: usize,
    queue: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut euroc_dir = None;
    let mut calibration = None;
    let mut config = None;
    let mut out_dir = PathBuf::from("target/ros2_mapper_replay");
    let mut max_frames = None;
    let mut realtime = true;
    let mut optimize_every_k = 100;
    let mut queue = 64;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--euroc-dir" => euroc_dir = Some(PathBuf::from(value()?)),
            "--calibration" => calibration = Some(PathBuf::from(value()?)),
            "--config" => config = Some(PathBuf::from(value()?)),
            "--out-dir" => out_dir = PathBuf::from(value()?),
            "--max-frames" => max_frames = Some(value()?.parse().map_err(|e| format!("{e}"))?),
            "--optimize-every-k" => {
                optimize_every_k = value()?.parse().map_err(|e| format!("{e}"))?
            }
            "--mapper-queue" => queue = value()?.parse().map_err(|e| format!("{e}"))?,
            "--as-fast-as-possible" => realtime = false,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        euroc_dir: euroc_dir.ok_or("--euroc-dir is required")?,
        calibration: calibration.ok_or("--calibration is required")?,
        config: config.ok_or("--config is required")?,
        out_dir,
        max_frames,
        realtime,
        optimize_every_k,
        queue,
    })
}

fn tum_line(out: &mut String, stamp_ns: i64, pose: &SE3) {
    let t = pose.translation;
    let q = pose.rotation.quaternion();
    let _ = writeln!(
        out,
        "{:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
        stamp_ns as f64 * 1e-9,
        t.x,
        t.y,
        t.z,
        q.i,
        q.j,
        q.k,
        q.w
    );
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out_dir).map_err(|e| e.to_string())?;
    let dataset = EurocSensorDataset::open(&args.euroc_dir, &args.calibration, &args.config)
        .map_err(|e| e.to_string())?;
    let calibration_json = fs::read_to_string(&args.calibration).map_err(|e| e.to_string())?;
    let config_json = fs::read_to_string(&args.config).map_err(|e| e.to_string())?;
    let options = VioOptions {
        retain_marg_data: true,
        ..VioOptions::default()
    };
    let core = VioCore::from_json(&calibration_json, &config_json, &options)?;
    let offset_ns = core.cam_time_offset_ns();
    let params = MapperParams {
        map_frame: "map".into(),
        queue_capacity: args.queue,
        optimize_every_k: args.optimize_every_k,
        trajectory_out: None,
    };
    let mapper = new_online_mapper(core.calibration(), core.config(), &params)?;
    let mut session = MapperSession::spawn(mapper, params.queue_capacity)?;
    let (mut frontend, mut backend) = core.into_stages();

    let frames = args
        .max_frames
        .unwrap_or(dataset.frame_count())
        .min(dataset.frame_count());
    let mut live = String::new();
    let mut vio = String::new();
    let mut gyro_bias = Vector3::zeros();
    let started = Instant::now();
    let mut first_stamp = None;
    for index in 0..frames {
        let frame = dataset.frame(index).map_err(|e| e.to_string())?;
        if args.realtime {
            let first = *first_stamp.get_or_insert(frame.timestamp_ns);
            let due = started + Duration::from_nanos((frame.timestamp_ns - first).max(0) as u64);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
        }
        let packet = SensorPacket {
            stamp_ns: frame.timestamp_ns + offset_ns,
            header_stamp_ns: frame.timestamp_ns,
            left: frame.cam0,
            right: frame.cam1.ok_or("the replay needs stereo frames")?,
            imu: frame.imu,
            initialization_imu: frame.initialization_imu,
        };
        let tracked = frontend
            .track(packet, gyro_bias)
            .map_err(|e| e.message.clone())?;
        let (estimate, marg_data) = backend
            .estimate_with_marg_data(tracked)
            .map_err(|e| e.message.clone())?;
        gyro_bias = estimate.gyro_bias;
        session.record_vio_pose(
            estimate.frame_id,
            estimate.header_stamp_ns,
            &estimate.imu_to_world,
        );
        if let Some(packet) = marg_data {
            session.offer(packet);
        }
        tum_line(&mut vio, estimate.header_stamp_ns, &estimate.imu_to_world);
        let map_pose = session.map_from_odom().compose(&estimate.imu_to_world);
        tum_line(&mut live, estimate.header_stamp_ns, &map_pose);
        if index % 200 == 0 {
            eprintln!("frame {index}/{frames} {}", session.status().summary());
        }
    }
    let vio_seconds = started.elapsed().as_secs_f64();
    let status = session.status();
    eprintln!("end of stream: {}", status.summary());
    let finished = session.finish()?;
    let mut saved = String::new();
    for (stamp, pose) in &finished.trajectory {
        tum_line(&mut saved, *stamp, pose);
    }
    for (name, text) in [
        ("trajectory_map_pose.tum", &live),
        ("trajectory_vio.tum", &vio),
        ("trajectory_mapper.tum", &saved),
    ] {
        fs::write(args.out_dir.join(name), text).map_err(|e| e.to_string())?;
    }
    eprintln!(
        "frames={frames} vio_wall={vio_seconds:.1}s total_wall={:.1}s keyframes={} \
         packets_sent={} packets_dropped={}",
        started.elapsed().as_secs_f64(),
        finished.keyframe_count,
        status.packets_sent,
        status.packets_dropped
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("euroc_mapper_replay: {error}");
        process::exit(1);
    }
}
