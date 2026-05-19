use std::fs;
use std::path::PathBuf;

use nalgebra::Vector3;
use visloc_io::kitti_imu::{
    parse_kitti_oxts_sample, parse_kitti_oxts_timestamp_line, parse_kitti_oxts_timestamps_txt,
    read_kitti_oxts_dir, KittiOxtsError,
};

#[test]
fn parses_single_oxts_sample_row() {
    // 30 fields: lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au wx wy wz wf wl wu pos_acc vel_acc nav numsats posmode velmode orimode
    let text = "49.0 8.4 113.0  0.01 -0.02 0.5  10.0 0.1 9.9 0.05 0.0  0.11 -0.22 9.81  0.10 -0.20 9.80  0.001 0.002 -0.003  0.001 0.002 -0.003  0.05 0.10  4 12 1 1 1\n";
    let sample = parse_kitti_oxts_sample(text).expect("parse");

    assert_eq!(sample.lat_deg, 49.0);
    assert_eq!(sample.lon_deg, 8.4);
    assert_eq!(sample.alt_m, 113.0);
    assert_eq!(sample.yaw_rad, 0.5);
    assert_eq!(sample.velocity_north_mps, 10.0);
    assert_eq!(
        sample.acceleration_body_mps2,
        Vector3::new(0.11, -0.22, 9.81)
    );
    assert_eq!(
        sample.angular_rate_body_rps,
        Vector3::new(0.001, 0.002, -0.003)
    );
    assert_eq!(sample.position_accuracy_m, 0.05);
    assert_eq!(sample.velocity_accuracy_mps, 0.10);
    assert_eq!(sample.number_of_satellites, 12);
}

#[test]
fn skips_comments_and_blank_lines_when_parsing_sample() {
    let text = "# header comment\n\n49.0 8.4 113.0 0 0 0 0 0 0 0 0  0 0 9.81  0 0 9.81  0 0 0  0 0 0  0.05 0.10 4 12 1 1 1\n";
    let sample = parse_kitti_oxts_sample(text).expect("parse");
    assert_eq!(sample.lat_deg, 49.0);
}

#[test]
fn parses_decimal_integer_status_flags_from_raw_kitti() {
    let text = "49.0 8.4 113.0 0 0 0 0 0 0 0 0  0 0 9.81  0 0 9.81  0 0 0  0 0 0  0.05 0.10 4.00000000000000 12.00000000000000 -1.00000000000000 -1.00000000000000 -1.00000000000000\n";
    let sample = parse_kitti_oxts_sample(text).expect("parse");
    assert_eq!(sample.navigation_status, 4);
    assert_eq!(sample.number_of_satellites, 12);
    assert_eq!(sample.position_mode, -1);
}

#[test]
fn rejects_oxts_sample_with_too_few_fields() {
    let err = parse_kitti_oxts_sample("1 2 3\n").expect_err("too few fields should fail");
    assert!(err.contains("30 fields"), "unexpected error: {err}");
}

#[test]
fn rejects_oxts_sample_with_non_numeric_field() {
    let text = format!("49.0 8.4 NOT_A_NUMBER {}\n", "0 ".repeat(27).trim());
    let err = parse_kitti_oxts_sample(&text).expect_err("non-numeric should fail");
    assert!(err.contains("invalid alt"), "unexpected error: {err}");
}

#[test]
fn parses_kitti_timestamp_line_with_nanosecond_fraction() {
    let nanos = parse_kitti_oxts_timestamp_line("2011-09-26 13:02:25.964389445").unwrap();
    // 2011-09-26 is day 15243 since 1970-01-01.
    let expected_seconds: i128 = 15243 * 86_400 + 13 * 3600 + 2 * 60 + 25;
    let expected_nanos: i128 = expected_seconds * 1_000_000_000 + 964_389_445;
    assert_eq!(nanos, expected_nanos);
}

#[test]
fn parses_kitti_timestamp_line_without_fraction() {
    let nanos = parse_kitti_oxts_timestamp_line("1970-01-01 00:00:01").unwrap();
    assert_eq!(nanos, 1_000_000_000);
}

#[test]
fn parses_kitti_timestamps_txt_skipping_blanks() {
    let text = "1970-01-01 00:00:00.000000000\n\n# comment\n1970-01-01 00:00:00.100000000\n";
    let stamps = parse_kitti_oxts_timestamps_txt(text).unwrap();
    assert_eq!(stamps, vec![0, 100_000_000]);
}

#[test]
fn reads_kitti_oxts_dir_round_trip() {
    let dir = make_tempdir("kitti_oxts_read");
    let oxts_root = dir.join("oxts");
    let data_dir = oxts_root.join("data");
    fs::create_dir_all(&data_dir).unwrap();

    let sample_row =
        "0 0 0 0 0 0 0 0 0 0 0  0.0 0.0 9.81  0 0 9.81  0.0 0.0 0.0  0 0 0  0.05 0.10 4 12 1 1 1\n";
    fs::write(data_dir.join("0000000000.txt"), sample_row).unwrap();
    let sample_row_2 =
        "0 0 0 0 0 0 0 0 0 0 0  1.0 0.0 9.81  0 0 9.81  0.0 0.0 0.1  0 0 0  0.05 0.10 4 12 1 1 1\n";
    fs::write(data_dir.join("0000000001.txt"), sample_row_2).unwrap();

    fs::write(
        oxts_root.join("timestamps.txt"),
        "1970-01-01 00:00:00.000000000\n1970-01-01 00:00:00.100000000\n",
    )
    .unwrap();

    let records = read_kitti_oxts_dir(&oxts_root).expect("read");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].timestamp_nanoseconds, 0);
    assert_eq!(records[1].timestamp_nanoseconds, 100_000_000);
    assert_eq!(
        records[1].sample.acceleration_body_mps2,
        Vector3::new(1.0, 0.0, 9.81)
    );
    assert_eq!(
        records[1].sample.angular_rate_body_rps,
        Vector3::new(0.0, 0.0, 0.1)
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn read_kitti_oxts_dir_detects_count_mismatch() {
    let dir = make_tempdir("kitti_oxts_mismatch");
    let oxts_root = dir.join("oxts");
    let data_dir = oxts_root.join("data");
    fs::create_dir_all(&data_dir).unwrap();
    let sample_row =
        "0 0 0 0 0 0 0 0 0 0 0  0.0 0.0 9.81  0 0 9.81  0.0 0.0 0.0  0 0 0  0.05 0.10 4 12 1 1 1\n";
    fs::write(data_dir.join("0000000000.txt"), sample_row).unwrap();

    // Two timestamps, but only one data file.
    fs::write(
        oxts_root.join("timestamps.txt"),
        "1970-01-01 00:00:00.000000000\n1970-01-01 00:00:00.100000000\n",
    )
    .unwrap();

    let err = read_kitti_oxts_dir(&oxts_root).expect_err("mismatch should fail");
    assert!(matches!(
        err,
        KittiOxtsError::DataTimestampCountMismatch {
            data_count: 1,
            timestamp_count: 2
        }
    ));

    fs::remove_dir_all(&dir).ok();
}

fn make_tempdir(label: &str) -> PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "visloc_{}_{}_{}",
        label,
        std::process::id(),
        suffix
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}
