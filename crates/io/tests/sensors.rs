use nalgebra::Point3;
use visloc_fusion::{TimedMeasurement, Timestamp};
use visloc_io::sensors::{parse_gnss_measurements_txt, read_gnss_measurements_txt, SensorLogError};

#[test]
fn parses_gnss_measurements_text() {
    let measurements = parse_gnss_measurements_txt(
        r#"
        # timestamp_ns x y z horizontal_accuracy vertical_accuracy
        timestamp_ns,x,y,z,hacc,vacc
        1000,1.0,2.0,3.0,0.5,1.5
        2000 4.0 5.0 6.0 2.5
        3000 7.0 8.0 9.0
        "#,
    )
    .unwrap();

    assert_eq!(measurements.len(), 3);
    assert_eq!(
        measurements[0].timestamp(),
        Timestamp::from_nanoseconds(1000)
    );
    assert_eq!(measurements[0].position_world, Point3::new(1.0, 2.0, 3.0));
    assert_eq!(measurements[0].horizontal_accuracy, Some(0.5));
    assert_eq!(measurements[0].vertical_accuracy, Some(1.5));
    assert_eq!(measurements[1].horizontal_accuracy, Some(2.5));
    assert_eq!(measurements[1].vertical_accuracy, None);
    assert_eq!(measurements[2].horizontal_accuracy, None);
}

#[test]
fn reads_gnss_measurements_text_file() {
    let dir = tempfile_dir();
    let path = dir.join("gnss.txt");
    std::fs::write(&path, "1000 1.0 2.0 3.0 4.0 5.0\n").unwrap();

    let measurements = read_gnss_measurements_txt(&path).unwrap();

    assert_eq!(measurements.len(), 1);
    assert_eq!(
        measurements[0].timestamp(),
        Timestamp::from_nanoseconds(1000)
    );
    assert_eq!(measurements[0].position_world, Point3::new(1.0, 2.0, 3.0));
    assert_eq!(measurements[0].horizontal_accuracy, Some(4.0));
    assert_eq!(measurements[0].vertical_accuracy, Some(5.0));
}

#[test]
fn rejects_gnss_measurement_line_without_position() {
    let error = parse_gnss_measurements_txt("1000 1.0 2.0")
        .expect_err("GNSS line without xyz position should fail");

    assert!(matches!(
        error,
        SensorLogError::InvalidGnssLine { line_number: 1, .. }
    ));
}

#[test]
fn rejects_invalid_gnss_measurement_number() {
    let error = parse_gnss_measurements_txt("1000 1.0 bad 3.0")
        .expect_err("invalid GNSS number should fail");

    assert!(matches!(
        error,
        SensorLogError::InvalidGnssLine { line_number: 1, .. }
    ));
    assert!(error.to_string().contains("invalid y"));
}

fn tempfile_dir() -> std::path::PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "visloc_sensor_log_test_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
