use std::path::PathBuf;

use nalgebra::{Point3, Vector3};
use visloc_basalt::{
    select_imu_interval, BasaltCalibration, ImuSample, TimeInterval, TimeIntervalError,
};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("euroc_ds_calib_minimal.json")
}

#[test]
fn reads_wrapped_euroc_ds_calibration_golden_contract() {
    let calibration = BasaltCalibration::from_path(fixture_path()).unwrap();

    assert_eq!(calibration.cameras.len(), 2);
    assert_eq!(calibration.resolutions, [(752, 480), (752, 480)]);
    assert_eq!(calibration.imu_update_rate_hz, 200.0);
    assert_eq!(calibration.mocap_to_imu_offset_ns, 140_763_258_159_875);
    assert_eq!(calibration.cam_time_offset_ns, 100);
    assert_eq!(
        calibration.t_imu_cam[0].translation,
        Vector3::new(0.1, -0.02, 0.03)
    );

    let camera = calibration.camera(0).unwrap();
    let pixel = camera.project(&Point3::new(0.1, -0.05, 1.0)).unwrap();
    assert!((pixel.x - 394.693_793_482_414_3).abs() < 1e-9);
    assert!((pixel.y - 234.676_283_381_008_18).abs() < 1e-9);

    let interval = TimeInterval::new(1_000, 2_000).unwrap();
    let corrected = calibration.camera_interval_to_imu(interval).unwrap();
    assert_eq!(corrected.start_ns(), 1_100);
    assert_eq!(corrected.end_ns(), 2_100);
}

#[test]
fn imu_selection_uses_calibration_clock_and_half_open_boundaries() {
    let calibration = BasaltCalibration::from_path(fixture_path()).unwrap();
    let camera_interval = TimeInterval::new(1_000, 2_000).unwrap();
    let imu_interval = calibration.camera_interval_to_imu(camera_interval).unwrap();
    let samples = [
        ImuSample::new(1_099, Vector3::zeros(), Vector3::zeros()),
        ImuSample::new(1_100, Vector3::new(1.0, 0.0, 0.0), Vector3::zeros()),
        ImuSample::new(2_099, Vector3::new(2.0, 0.0, 0.0), Vector3::zeros()),
        ImuSample::new(2_100, Vector3::new(3.0, 0.0, 0.0), Vector3::zeros()),
    ];
    let selected = select_imu_interval(&samples, imu_interval).unwrap();
    assert_eq!(selected.samples().len(), 2);
    assert_eq!(selected.samples()[0].timestamp_ns, 1_100);
    assert_eq!(selected.samples()[1].timestamp_ns, 2_099);
}

#[test]
fn unsupported_camera_model_is_rejected_before_generic_pipeline_use() {
    let json =
        include_str!("fixtures/euroc_ds_calib_minimal.json").replace("\"ds\"", "\"pinhole\"");
    let error = BasaltCalibration::from_json_str(&json).unwrap_err();
    assert!(matches!(
        error,
        visloc_basalt::CalibrationError::UnsupportedCameraType { .. }
    ));
}

#[test]
fn interval_overflow_is_reported() {
    let calibration = BasaltCalibration::from_path(fixture_path()).unwrap();
    let interval = TimeInterval::new(i64::MAX - 50, i64::MAX - 10).unwrap();
    assert!(matches!(
        calibration.camera_interval_to_imu(interval),
        Err(TimeIntervalError::TimestampOverflow)
    ));
}

#[test]
fn retain_cameras_produces_cam0_only_monocular_rig() {
    let stereo = BasaltCalibration::from_path(fixture_path()).unwrap();
    let mono = stereo.retain_cameras(1).unwrap();

    assert_eq!(mono.cameras.len(), 1);
    assert_eq!(mono.t_imu_cam.len(), 1);
    assert_eq!(mono.resolutions, [(752, 480)]);
    assert_eq!(mono.cameras[0], stereo.cameras[0]);
    assert_eq!(mono.t_imu_cam[0], stereo.t_imu_cam[0]);
    assert!(mono.camera(1).is_none());
    // IMU and clock contracts are untouched.
    assert_eq!(mono.accel_noise_std, stereo.accel_noise_std);
    assert_eq!(mono.gyro_bias_std, stereo.gyro_bias_std);
    assert_eq!(mono.cam_time_offset_ns, stereo.cam_time_offset_ns);
    // Retaining every camera is the identity.
    assert_eq!(stereo.retain_cameras(2).unwrap(), stereo);
    assert!(stereo.retain_cameras(0).is_err());
    assert!(stereo.retain_cameras(3).is_err());
}
