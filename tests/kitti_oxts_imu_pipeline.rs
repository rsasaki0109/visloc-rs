//! End-to-end smoke for the KITTI raw OXTS / IMU CLI integration:
//!
//! - `read_kitti_oxts_dir` round-trips a synthetic `oxts/` layout (data + timestamps).
//! - The body-frame gyro / accel triplets plus per-sample wall-clock nanoseconds are sliced
//!   into the per-keyframe `Vec<Vec<StereoVoBaImuSample>>` layout via
//!   `slice_imu_samples_for_keyframes`, using image-stream `timestamps.txt` rows as
//!   keyframe times.
//!
//! These checks mirror what `examples/stereo_vo_external_deep_files --kitti-oxts-dir`
//! does before handing the windows to `refine_stereo_vo_with_ba`.

use std::fs;
use std::path::PathBuf;

use nalgebra::Vector3;
use visloc_rs::{
    parse_kitti_oxts_timestamps_txt, read_kitti_oxts_dir, slice_imu_samples_for_keyframes,
};

fn make_tempdir(label: &str) -> PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "visloc_{}_{}_{}",
        label,
        std::process::id(),
        suffix,
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a row whose body-frame accel is `(ax, ay, az)` and body-frame gyro is `(wx, wy, wz)`.
/// All other OXTS fields are zero / nominal.
fn synthetic_oxts_row(ax: f64, ay: f64, az: f64, wx: f64, wy: f64, wz: f64) -> String {
    // lat lon alt roll pitch yaw vn ve vf vl vu  ax ay az  af al au  wx wy wz  wf wl wu  pos_acc vel_acc navstat numsats posmode velmode orimode
    format!(
        "0 0 0 0 0 0 0 0 0 0 0  {ax} {ay} {az}  0 0 9.81  {wx} {wy} {wz}  0 0 0  0.05 0.10 4 12 1 1 1\n",
    )
}

#[test]
fn kitti_oxts_dir_records_slice_into_keyframe_windows() {
    let dir = make_tempdir("kitti_oxts_pipeline");
    let oxts_root = dir.join("oxts");
    let data_dir = oxts_root.join("data");
    fs::create_dir_all(&data_dir).unwrap();

    // 4 OXTS samples at 0, 100, 200, 300 ms.
    fs::write(
        data_dir.join("0000000000.txt"),
        synthetic_oxts_row(0.10, 0.0, 9.81, 0.01, 0.0, 0.0),
    )
    .unwrap();
    fs::write(
        data_dir.join("0000000001.txt"),
        synthetic_oxts_row(0.20, 0.0, 9.81, 0.02, 0.0, 0.0),
    )
    .unwrap();
    fs::write(
        data_dir.join("0000000002.txt"),
        synthetic_oxts_row(0.30, 0.0, 9.81, 0.03, 0.0, 0.0),
    )
    .unwrap();
    fs::write(
        data_dir.join("0000000003.txt"),
        synthetic_oxts_row(0.40, 0.0, 9.81, 0.04, 0.0, 0.0),
    )
    .unwrap();

    fs::write(
        oxts_root.join("timestamps.txt"),
        "1970-01-01 00:00:00.000000000\n\
         1970-01-01 00:00:00.100000000\n\
         1970-01-01 00:00:00.200000000\n\
         1970-01-01 00:00:00.300000000\n",
    )
    .unwrap();

    // Image timestamps: 3 keyframes at 0, 100, 300 ms → two windows of 100 ms and 200 ms.
    let image_timestamps_path = dir.join("image_timestamps.txt");
    fs::write(
        &image_timestamps_path,
        "1970-01-01 00:00:00.000000000\n\
         1970-01-01 00:00:00.100000000\n\
         1970-01-01 00:00:00.300000000\n",
    )
    .unwrap();

    let records = read_kitti_oxts_dir(&oxts_root).expect("read OXTS");
    assert_eq!(records.len(), 4);
    assert_eq!(records[0].timestamp_nanoseconds, 0);
    assert_eq!(records[3].timestamp_nanoseconds, 300_000_000);

    let imu_t: Vec<i128> = records.iter().map(|r| r.timestamp_nanoseconds).collect();
    let imu_gyro: Vec<Vector3<f64>> = records
        .iter()
        .map(|r| r.sample.angular_rate_body_rps)
        .collect();
    let imu_accel: Vec<Vector3<f64>> = records
        .iter()
        .map(|r| r.sample.acceleration_body_mps2)
        .collect();
    assert_eq!(imu_accel[1], Vector3::new(0.20, 0.0, 9.81));
    assert_eq!(imu_gyro[2], Vector3::new(0.03, 0.0, 0.0));

    let kf_text = fs::read_to_string(&image_timestamps_path).unwrap();
    let kf_t = parse_kitti_oxts_timestamps_txt(&kf_text).expect("kf timestamps");
    assert_eq!(kf_t, vec![0i128, 100_000_000, 300_000_000]);

    let windows =
        slice_imu_samples_for_keyframes(&imu_t, &imu_gyro, &imu_accel, &kf_t).expect("slice");
    assert_eq!(windows.len(), 2);

    // Window 0 covers (0 ms, 100 ms]: contains the sample at 100 ms with dt=100 ms.
    assert_eq!(windows[0].len(), 1);
    assert!((windows[0][0].dt - 0.1).abs() < 1e-12);
    assert_eq!(windows[0][0].accel, Vector3::new(0.20, 0.0, 9.81));
    assert_eq!(windows[0][0].gyro, Vector3::new(0.02, 0.0, 0.0));

    // Window 1 covers (100 ms, 300 ms]: contains samples at 200 ms (dt=100 ms) and 300 ms (dt=100 ms).
    assert_eq!(windows[1].len(), 2);
    assert!((windows[1][0].dt - 0.1).abs() < 1e-12);
    assert_eq!(windows[1][0].accel, Vector3::new(0.30, 0.0, 9.81));
    assert!((windows[1][1].dt - 0.1).abs() < 1e-12);
    assert_eq!(windows[1][1].accel, Vector3::new(0.40, 0.0, 9.81));

    // Both windows' integrated Δt matches the keyframe span.
    let total_dt: f64 = windows.iter().flatten().map(|s| s.dt).sum();
    assert!((total_dt - 0.3).abs() < 1e-12);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn kitti_oxts_slice_emits_trailing_zoh_when_last_sample_short() {
    let dir = make_tempdir("kitti_oxts_zoh");
    let oxts_root = dir.join("oxts");
    let data_dir = oxts_root.join("data");
    fs::create_dir_all(&data_dir).unwrap();

    // Two OXTS samples at 50 ms and 150 ms.
    fs::write(
        data_dir.join("0000000000.txt"),
        synthetic_oxts_row(1.0, 0.0, 9.81, 0.0, 0.0, 0.05),
    )
    .unwrap();
    fs::write(
        data_dir.join("0000000001.txt"),
        synthetic_oxts_row(2.0, 0.0, 9.81, 0.0, 0.0, 0.10),
    )
    .unwrap();

    fs::write(
        oxts_root.join("timestamps.txt"),
        "1970-01-01 00:00:00.050000000\n\
         1970-01-01 00:00:00.150000000\n",
    )
    .unwrap();

    // One keyframe interval from 0 to 200 ms: last sample at 150 ms stops short, so the
    // slicer must append a trailing zero-order-hold step of dt = 50 ms.
    let image_timestamps_path = dir.join("image_timestamps.txt");
    fs::write(
        &image_timestamps_path,
        "1970-01-01 00:00:00.000000000\n\
         1970-01-01 00:00:00.200000000\n",
    )
    .unwrap();

    let records = read_kitti_oxts_dir(&oxts_root).expect("read OXTS");
    let imu_t: Vec<i128> = records.iter().map(|r| r.timestamp_nanoseconds).collect();
    let imu_gyro: Vec<Vector3<f64>> = records
        .iter()
        .map(|r| r.sample.angular_rate_body_rps)
        .collect();
    let imu_accel: Vec<Vector3<f64>> = records
        .iter()
        .map(|r| r.sample.acceleration_body_mps2)
        .collect();

    let kf_text = fs::read_to_string(&image_timestamps_path).unwrap();
    let kf_t = parse_kitti_oxts_timestamps_txt(&kf_text).expect("kf timestamps");
    let windows =
        slice_imu_samples_for_keyframes(&imu_t, &imu_gyro, &imu_accel, &kf_t).expect("slice");

    assert_eq!(windows.len(), 1);
    // Three entries: 50ms (sample 0), 100ms (sample 1), 50ms (trailing ZOH from sample 1).
    assert_eq!(windows[0].len(), 3);
    assert!((windows[0][0].dt - 0.05).abs() < 1e-12);
    assert_eq!(windows[0][0].accel, Vector3::new(1.0, 0.0, 9.81));
    assert!((windows[0][1].dt - 0.10).abs() < 1e-12);
    assert_eq!(windows[0][1].accel, Vector3::new(2.0, 0.0, 9.81));
    assert!((windows[0][2].dt - 0.05).abs() < 1e-12);
    // Trailing ZOH carries the last sample's gyro/accel.
    assert_eq!(windows[0][2].accel, Vector3::new(2.0, 0.0, 9.81));
    assert_eq!(windows[0][2].gyro, Vector3::new(0.0, 0.0, 0.10));

    let total_dt: f64 = windows[0].iter().map(|s| s.dt).sum();
    assert!((total_dt - 0.20).abs() < 1e-12);

    fs::remove_dir_all(&dir).ok();
}
