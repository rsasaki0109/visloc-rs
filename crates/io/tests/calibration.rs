use visloc_core::types::CameraModel;
use visloc_io::calibration::{
    kitti_projection_to_pinhole_camera, parse_kitti_calibration_txt, read_kitti_pinhole_camera,
    CalibrationError,
};

#[test]
fn parses_kitti_projection_matrices() {
    let projections = parse_kitti_calibration_txt(
        r#"
        # KITTI-style calibration
        P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
        R0_rect: 1 0 0 0 1 0 0 0 1
        P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.454157e-01 0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03
        "#,
    )
    .unwrap();

    assert_eq!(projections.len(), 2);
    assert_eq!(projections[0].label, "P0");
    assert_eq!(projections[1].label, "P2");
    assert!((projections[1].fx() - 707.0493).abs() < 1.0e-6);
    assert!((projections[1].fy() - 707.0493).abs() < 1.0e-6);
    assert!((projections[1].cx() - 604.0814).abs() < 1.0e-6);
    assert!((projections[1].cy() - 180.5066).abs() < 1.0e-6);
}

#[test]
fn converts_kitti_projection_to_pinhole_camera() {
    let projections =
        parse_kitti_calibration_txt("P2: 700 0 600 40 0 710 180 0 0 0 1 0\n").unwrap();

    let camera = kitti_projection_to_pinhole_camera(&projections, "P2", 7, 1242, 375).unwrap();

    assert_eq!(camera.id, 7);
    assert_eq!(camera.model, CameraModel::Pinhole);
    assert_eq!(camera.width, 1242);
    assert_eq!(camera.height, 375);
    assert_eq!(camera.intrinsics(), Some((700.0, 710.0, 600.0, 180.0)));
}

#[test]
fn reads_kitti_projection_from_file() {
    let dir = tempfile_dir();
    let path = dir.join("calib.txt");
    std::fs::write(&path, "P0: 700 0 600 0 0 700 180 0 0 0 1 0\n").unwrap();

    let camera = read_kitti_pinhole_camera(&path, "P0", 1, 1242, 375).unwrap();

    assert_eq!(camera.intrinsics(), Some((700.0, 700.0, 600.0, 180.0)));
}

#[test]
fn rejects_projection_with_wrong_value_count() {
    let error =
        parse_kitti_calibration_txt("P2: 1 2 3\n").expect_err("short KITTI projection should fail");

    assert!(matches!(
        error,
        CalibrationError::InvalidKittiLine { line_number: 1, .. }
    ));
}

#[test]
fn rejects_missing_projection_label() {
    let projections = parse_kitti_calibration_txt("P0: 700 0 600 0 0 700 180 0 0 0 1 0\n").unwrap();

    let error = kitti_projection_to_pinhole_camera(&projections, "P2", 1, 1242, 375)
        .expect_err("missing projection should fail");

    assert!(matches!(
        error,
        CalibrationError::KittiProjectionNotFound { .. }
    ));
}

#[test]
fn rejects_non_positive_focal_length() {
    let projections = parse_kitti_calibration_txt("P2: 0 0 600 0 0 700 180 0 0 0 1 0\n").unwrap();

    let error = kitti_projection_to_pinhole_camera(&projections, "P2", 1, 1242, 375)
        .expect_err("zero focal length should fail");

    assert!(matches!(
        error,
        CalibrationError::InvalidKittiProjection { .. }
    ));
}

#[test]
fn stereo_baseline_recovers_kitti_value() {
    // KITTI seq 00 calibration (truncated): P0 has zero baseline column,
    // P1 has tx = -3.861448e+02 and shared fx = 7.188560e+02. Their
    // baseline is `-tx / fx ≈ 0.5371` m, the published KITTI baseline.
    let projections = parse_kitti_calibration_txt(
        r#"
        P0: 7.188560e+02 0 6.071928e+02 0 0 7.188560e+02 1.852157e+02 0 0 0 1 0
        P1: 7.188560e+02 0 6.071928e+02 -3.861448e+02 0 7.188560e+02 1.852157e+02 0 0 0 1 0
        "#,
    )
    .unwrap();
    let p0 = &projections[0];
    let p1 = &projections[1];
    let baseline = p1
        .stereo_baseline_from(p0)
        .expect("P0/P1 are a rectified stereo pair");
    // 386.1448 / 718.856 ≈ 0.537166 m (matches the documented KITTI 00
    // stereo baseline of ~0.54 m).
    assert!((baseline - 0.537166).abs() < 1.0e-4, "baseline {baseline}");
}

#[test]
fn stereo_baseline_returns_none_when_intrinsics_differ() {
    let projections = parse_kitti_calibration_txt(
        "P0: 700 0 600 0 0 700 180 0 0 0 1 0\nP2: 720 0 605 -300 0 720 182 0 0 0 1 0\n",
    )
    .unwrap();
    assert!(projections[1]
        .stereo_baseline_from(&projections[0])
        .is_none());
}

#[test]
fn stereo_baseline_returns_none_when_columns_match() {
    // Both P columns have zero tx → no baseline.
    let projections = parse_kitti_calibration_txt("P0: 700 0 600 0 0 700 180 0 0 0 1 0\n").unwrap();
    assert!(projections[0]
        .stereo_baseline_from(&projections[0])
        .is_none());
}

fn tempfile_dir() -> std::path::PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "visloc_calibration_test_{}_{}",
        std::process::id(),
        suffix
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
