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
