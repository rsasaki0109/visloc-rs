//! COLMAP camera records written by the exporters must be valid COLMAP:
//! a self-calibrated radial `[k1, k2]` tail on a `Pinhole` is exported as
//! `OPENCV` (p1 = p2 = 0, same projection), never as a 6-parameter `PINHOLE`.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, CameraModel};
use visloc_io::colmap::{
    colmap_camera_record, read_colmap_binary_model, read_colmap_text_model,
    write_colmap_binary_model_for_3dgs, write_colmap_text_model_for_3dgs, ColmapError,
};
use visloc_vision::features::FeatureSet;
use visloc_vision::stereo_vo::StereoFeature;

fn make_tempdir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
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

fn radial_camera() -> Camera {
    Camera::pinhole_radial(
        3, 1920, 1080, 1353.09, 1338.03, 962.7, 539.93, 0.0346, -0.0235,
    )
}

fn one_frame_inputs() -> (Vec<Pose>, Vec<FeatureSet>, Vec<Vec<StereoFeature>>) {
    let poses = vec![Pose {
        world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::zeros()),
    }];
    let features =
        vec![FeatureSet::new(vec![Point2::new(100.0, 80.0)], vec![vec![0.0_f32; 2]]).unwrap()];
    let stereo = vec![vec![StereoFeature {
        left_index: 0,
        right_index: 0,
        disparity: 5.0,
        point_cam: Point3::new(0.1, 0.2, 4.0),
    }]];
    (poses, features, stereo)
}

fn assert_same_projection(a: &Camera, b: &Camera) {
    for p in [
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.5, -0.3, 1.0),
        Point3::new(-0.7, 0.4, 1.2),
        Point3::new(0.6, 0.35, 0.9),
    ] {
        let ua = a.project(&p).unwrap();
        let ub = b.project(&p).unwrap();
        assert!((ua - ub).norm() < 1e-6, "{ua:?} vs {ub:?}");
    }
}

fn camera_line(dir: &std::path::Path) -> Vec<String> {
    let text = fs::read_to_string(dir.join("cameras.txt")).unwrap();
    let line = text.lines().find(|l| !l.starts_with('#')).unwrap();
    line.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn radial_pinhole_is_recorded_as_opencv_with_zero_tangential() {
    let (model, params) = colmap_camera_record(&radial_camera());
    assert_eq!(model, CameraModel::OpenCv);
    assert_eq!(
        params,
        vec![1353.09, 1338.03, 962.7, 539.93, 0.0346, -0.0235, 0.0, 0.0]
    );
}

#[test]
fn zero_distortion_tail_collapses_to_plain_pinhole() {
    let camera = Camera::pinhole_radial(1, 640, 480, 500.0, 500.0, 320.0, 240.0, 0.0, 0.0);
    let (model, params) = colmap_camera_record(&camera);
    assert_eq!(model, CameraModel::Pinhole);
    assert_eq!(params, vec![500.0, 500.0, 320.0, 240.0]);
}

#[test]
fn text_export_writes_valid_opencv_and_round_trips_projection() {
    let dir = make_tempdir("colmap_camera_text");
    let camera = radial_camera();
    let (poses, features, stereo) = one_frame_inputs();
    write_colmap_text_model_for_3dgs(&dir, &camera, &poses, &features, &stereo, |i| {
        format!("frame_{i:06}.png")
    })
    .unwrap();
    let tokens = camera_line(&dir);
    assert_eq!(tokens[1], "OPENCV");
    assert_eq!(
        tokens.len(),
        4 + 8,
        "OPENCV carries exactly 8 params: {tokens:?}"
    );

    let map = read_colmap_text_model(&dir).unwrap();
    let read = map.cameras.values().next().unwrap();
    assert_eq!(read.model, CameraModel::OpenCv);
    assert_same_projection(&camera, read);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn binary_export_writes_valid_opencv_and_round_trips_projection() {
    let dir = make_tempdir("colmap_camera_bin");
    let camera = radial_camera();
    let (poses, features, stereo) = one_frame_inputs();
    write_colmap_binary_model_for_3dgs(&dir, &camera, &poses, &features, &stereo, |i| {
        format!("frame_{i:06}.png")
    })
    .unwrap();
    let map = read_colmap_binary_model(&dir).unwrap();
    let read = map.cameras.values().next().unwrap();
    assert_eq!(read.model, CameraModel::OpenCv);
    assert_eq!(read.params.len(), 8);
    assert_same_projection(&camera, read);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn export_rejects_param_count_that_is_not_colmap() {
    let dir = make_tempdir("colmap_camera_bad_count");
    let camera = Camera {
        id: 1,
        model: CameraModel::Radial,
        width: 640,
        height: 480,
        params: vec![500.0, 320.0, 240.0],
    };
    let (poses, features, stereo) = one_frame_inputs();
    for result in [
        write_colmap_text_model_for_3dgs(&dir, &camera, &poses, &features, &stereo, |i| {
            format!("f{i}.png")
        }),
        write_colmap_binary_model_for_3dgs(&dir, &camera, &poses, &features, &stereo, |i| {
            format!("f{i}.png")
        }),
    ] {
        match result {
            Err(ColmapError::InvalidExportInput(msg)) => {
                assert!(msg.contains("needs 5 params"), "{msg}")
            }
            other => panic!("expected InvalidExportInput, got {other:?}"),
        }
    }
    fs::remove_dir_all(&dir).ok();
}
