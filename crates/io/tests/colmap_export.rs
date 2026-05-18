//! Round-trip the new 3DGS-oriented COLMAP export through the existing
//! COLMAP text reader.
//!
//! The test exercises [`visloc_io::colmap::write_colmap_text_model_for_3dgs`]
//! on a synthetic 3-frame stereo VO trajectory and asserts that the three
//! files (`cameras.txt`, `images.txt`, `points3D.txt`) are valid COLMAP and
//! that `read_colmap_text_model` reconstructs the same camera + keyframe +
//! landmark counts.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{Camera, CameraModel};
use visloc_io::colmap::{
    read_colmap_binary_model, read_colmap_text_model, write_colmap_binary_model_for_3dgs,
    write_colmap_text_model_for_3dgs, ColmapError,
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

fn synthetic_camera() -> Camera {
    Camera::pinhole(7, 640, 480, 500.0, 500.0, 320.0, 240.0)
}

fn stereo_feature(left_index: usize, right_index: usize, point_cam: Point3<f64>) -> StereoFeature {
    StereoFeature {
        left_index,
        right_index,
        disparity: 5.0,
        point_cam,
    }
}

fn feature_set_from_keypoints(xy: &[(f64, f64)]) -> FeatureSet {
    let keypoints: Vec<Point2<f64>> = xy.iter().map(|&(x, y)| Point2::new(x, y)).collect();
    let descriptors: Vec<Vec<f32>> = (0..xy.len()).map(|_| vec![0.0_f32; 2]).collect();
    FeatureSet::new(keypoints, descriptors).unwrap()
}

#[test]
fn write_colmap_text_model_for_3dgs_round_trips_through_reader() {
    let dir = make_tempdir("colmap_export_round_trip");

    let camera = synthetic_camera();
    // 3 forward-translating poses at world centres (0,0,0), (0,0,1), (0,0,2).
    let poses: Vec<Pose> = (0..3)
        .map(|i| Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(0.0, 0.0, -(i as f64)),
            ),
        })
        .collect();
    // 4 left keypoints per frame; first two participate in stereo features.
    let left_features = vec![
        feature_set_from_keypoints(&[(100.0, 80.0), (200.0, 90.0), (300.0, 100.0), (400.0, 110.0)]),
        feature_set_from_keypoints(&[(101.0, 80.0), (201.0, 90.0), (301.0, 100.0), (401.0, 110.0)]),
        feature_set_from_keypoints(&[(102.0, 80.0), (202.0, 90.0), (302.0, 100.0), (402.0, 110.0)]),
    ];
    // Each frame triangulates two stereo features pointing into the +z half-space.
    let stereo_per_frame = vec![
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 5.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 6.0)),
        ],
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 4.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 5.0)),
        ],
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 3.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 4.0)),
        ],
    ];

    let summary = write_colmap_text_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect("write 3DGS colmap text model");

    assert_eq!(summary.frame_count, 3);
    assert_eq!(summary.landmark_count, 6);
    assert_eq!(summary.observation_count, 6);

    assert!(dir.join("cameras.txt").exists());
    assert!(dir.join("images.txt").exists());
    assert!(dir.join("points3D.txt").exists());

    // Re-read via the existing COLMAP text reader: the three files should
    // parse into a valid VisualMap with the same counts. (The 2D point list
    // on each image line is preserved through write+read, but the existing
    // reader uses image_id as the COLMAP keyframe id, so we just sanity-check
    // counts here.)
    let map = read_colmap_text_model(&dir).expect("read colmap text model");
    assert_eq!(map.cameras.len(), 1);
    assert_eq!(map.keyframes.len(), 3);
    assert_eq!(map.landmarks.len(), 6);

    // Verify a few of the world-frame landmark positions: frame 0 has cam_to_world
    // = identity, so landmark 1 should land at (0.5, 0.0, 5.0).
    let lm1 = map.landmarks.get(&1).expect("landmark 1");
    assert!((lm1.position.x - 0.5).abs() < 1e-9);
    assert!((lm1.position.z - 5.0).abs() < 1e-9);

    // Frame 1's camera centre is at world z=+1, so its first stereo feature
    // (point_cam = (0.5, 0.0, 4.0)) lifts to world z = 5.0.
    let lm3 = map.landmarks.get(&3).expect("landmark 3");
    assert!((lm3.position.z - 5.0).abs() < 1e-9);

    // cameras.txt should encode PINHOLE with our intrinsics.
    let cameras_text = fs::read_to_string(dir.join("cameras.txt")).unwrap();
    assert!(cameras_text.contains("PINHOLE"));
    assert!(cameras_text.contains("640 480"));
    assert!(cameras_text.contains("500"));

    // images.txt second line should embed POINT3D_ID columns for the
    // stereo-paired keypoints (and `-1` for the unpaired ones).
    let images_text = fs::read_to_string(dir.join("images.txt")).unwrap();
    assert!(images_text.contains("000000.png"));
    assert!(images_text.contains("000002.png"));
    assert!(images_text.contains(" -1"));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_binary_model_for_3dgs_round_trips_through_binary_reader() {
    let dir = make_tempdir("colmap_binary_export_round_trip");

    let camera = synthetic_camera();
    // Same 3-frame setup as the text round-trip test.
    let poses: Vec<Pose> = (0..3)
        .map(|i| Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(0.0, 0.0, -(i as f64)),
            ),
        })
        .collect();
    let left_features = vec![
        feature_set_from_keypoints(&[(100.0, 80.0), (200.0, 90.0), (300.0, 100.0), (400.0, 110.0)]),
        feature_set_from_keypoints(&[(101.0, 80.0), (201.0, 90.0), (301.0, 100.0), (401.0, 110.0)]),
        feature_set_from_keypoints(&[(102.0, 80.0), (202.0, 90.0), (302.0, 100.0), (402.0, 110.0)]),
    ];
    let stereo_per_frame = vec![
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 5.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 6.0)),
        ],
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 4.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 5.0)),
        ],
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 3.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 4.0)),
        ],
    ];

    let summary = write_colmap_binary_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect("write 3DGS colmap binary model");

    assert_eq!(summary.frame_count, 3);
    assert_eq!(summary.landmark_count, 6);
    assert_eq!(summary.observation_count, 6);

    assert!(dir.join("cameras.bin").exists());
    assert!(dir.join("images.bin").exists());
    assert!(dir.join("points3D.bin").exists());

    let map = read_colmap_binary_model(&dir).expect("read colmap binary model");
    assert_eq!(map.cameras.len(), 1);
    assert_eq!(map.keyframes.len(), 3);
    assert_eq!(map.landmarks.len(), 6);

    // Sanity-check intrinsics survived the round trip.
    let camera_round = map.cameras.values().next().unwrap();
    assert_eq!(camera_round.width, 640);
    assert_eq!(camera_round.height, 480);
    assert_eq!(camera_round.params, camera.params);

    // Same landmark-position checks as the text test.
    let lm1 = map.landmarks.get(&1).expect("landmark 1");
    assert!((lm1.position.x - 0.5).abs() < 1e-9);
    assert!((lm1.position.z - 5.0).abs() < 1e-9);
    let lm3 = map.landmarks.get(&3).expect("landmark 3");
    assert!((lm3.position.z - 5.0).abs() < 1e-9);

    // Keyframe poses round-trip through the binary writer's quaternion +
    // translation encoding.
    let kf0 = map.keyframes.get(&0).expect("keyframe 0");
    let t0 = kf0.frame.pose.as_ref().unwrap().world_to_camera.translation;
    assert!((t0.norm() - 0.0).abs() < 1e-9);
    let kf2 = map.keyframes.get(&2).expect("keyframe 2");
    let t2 = kf2.frame.pose.as_ref().unwrap().world_to_camera.translation;
    assert!((t2.z - -2.0).abs() < 1e-9);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_binary_model_for_3dgs_rejects_length_mismatch() {
    let dir = make_tempdir("colmap_binary_export_mismatch");
    let camera = synthetic_camera();
    let poses = vec![Pose::identity(), Pose::identity()];
    let left_features = vec![feature_set_from_keypoints(&[(100.0, 80.0)])]; // wrong length
    let stereo_per_frame = vec![vec![], vec![]];
    let err = write_colmap_binary_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect_err("length mismatch should fail");
    assert!(matches!(err, ColmapError::InvalidExportInput(_)));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_text_model_for_3dgs_rejects_length_mismatch() {
    let dir = make_tempdir("colmap_export_mismatch");
    let camera = synthetic_camera();
    let poses = vec![Pose::identity(), Pose::identity()];
    let left_features = vec![feature_set_from_keypoints(&[(100.0, 80.0)])]; // wrong length
    let stereo_per_frame = vec![vec![], vec![]];
    let err = write_colmap_text_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect_err("length mismatch should fail");
    assert!(matches!(err, ColmapError::InvalidExportInput(_)));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_binary_model_for_3dgs_rejects_image_name_with_nul_byte() {
    let dir = make_tempdir("colmap_binary_export_nul_name");
    let camera = synthetic_camera();
    let poses = vec![Pose::identity()];
    let left_features = vec![feature_set_from_keypoints(&[(100.0, 80.0)])];
    let stereo_per_frame = vec![vec![]];
    // COLMAP binary NAME fields are NUL-terminated, so an embedded NUL byte
    // would silently truncate the filename. The writer must reject this
    // up-front instead of writing a corrupted images.bin.
    let err = write_colmap_binary_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |_idx| "bad\0name.png".to_owned(),
    )
    .expect_err("embedded NUL byte should fail");
    assert!(matches!(err, ColmapError::InvalidExportInput(_)));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_binary_model_for_3dgs_rejects_unknown_camera_model() {
    let dir = make_tempdir("colmap_binary_export_unknown_model");
    // `CameraModel::Unknown("BOGUS")` has no COLMAP model id, so the
    // binary writer cannot encode it; the helper must surface
    // InvalidExportInput rather than panic or pick an arbitrary id.
    let camera = Camera {
        id: 7,
        model: CameraModel::Unknown("BOGUS".to_owned()),
        width: 640,
        height: 480,
        params: vec![500.0, 500.0, 320.0, 240.0],
    };
    let poses = vec![Pose::identity()];
    let left_features = vec![feature_set_from_keypoints(&[(100.0, 80.0)])];
    let stereo_per_frame = vec![vec![]];
    let err = write_colmap_binary_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect_err("unknown CameraModel should fail");
    assert!(matches!(err, ColmapError::InvalidExportInput(_)));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_text_and_binary_models_for_3dgs_emit_equivalent_maps() {
    // Justifies the smoke harness emitting both formats from a single VO
    // run and trusting they encode the same map: text and binary writers,
    // fed the same input, must produce maps that re-read to the same
    // camera intrinsics, keyframe poses, and landmark world positions.
    let text_dir = make_tempdir("colmap_text_vs_binary_text");
    let binary_dir = make_tempdir("colmap_text_vs_binary_binary");

    let camera = synthetic_camera();
    let poses: Vec<Pose> = (0..3)
        .map(|i| Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::identity(),
                Vector3::new(0.0, 0.0, -(i as f64)),
            ),
        })
        .collect();
    let left_features = vec![
        feature_set_from_keypoints(&[(100.0, 80.0), (200.0, 90.0), (300.0, 100.0), (400.0, 110.0)]),
        feature_set_from_keypoints(&[(101.0, 80.0), (201.0, 90.0), (301.0, 100.0), (401.0, 110.0)]),
        feature_set_from_keypoints(&[(102.0, 80.0), (202.0, 90.0), (302.0, 100.0), (402.0, 110.0)]),
    ];
    let stereo_per_frame = vec![
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 5.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 6.0)),
        ],
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 4.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 5.0)),
        ],
        vec![
            stereo_feature(0, 0, Point3::new(0.5, 0.0, 3.0)),
            stereo_feature(1, 1, Point3::new(-0.5, 0.0, 4.0)),
        ],
    ];

    let text_summary = write_colmap_text_model_for_3dgs(
        &text_dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect("write text model");
    let binary_summary = write_colmap_binary_model_for_3dgs(
        &binary_dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect("write binary model");

    assert_eq!(text_summary.frame_count, binary_summary.frame_count);
    assert_eq!(text_summary.landmark_count, binary_summary.landmark_count);
    assert_eq!(
        text_summary.observation_count,
        binary_summary.observation_count
    );

    let text_map = read_colmap_text_model(&text_dir).expect("read text model");
    let binary_map = read_colmap_binary_model(&binary_dir).expect("read binary model");

    // Camera intrinsics parity.
    assert_eq!(text_map.cameras.len(), binary_map.cameras.len());
    let text_camera = text_map.cameras.values().next().expect("text camera");
    let binary_camera = binary_map.cameras.values().next().expect("binary camera");
    assert_eq!(text_camera.width, binary_camera.width);
    assert_eq!(text_camera.height, binary_camera.height);
    assert_eq!(text_camera.params, binary_camera.params);

    // Keyframe pose parity. Both readers index keyframes by COLMAP image_id
    // (= frame_idx for our writer), so equal-id keyframes must encode the
    // same pose.
    assert_eq!(text_map.keyframes.len(), binary_map.keyframes.len());
    for (kf_id, text_kf) in &text_map.keyframes {
        let binary_kf = binary_map
            .keyframes
            .get(kf_id)
            .expect("matching binary keyframe");
        let text_pose = text_kf.frame.pose.as_ref().expect("text pose");
        let binary_pose = binary_kf.frame.pose.as_ref().expect("binary pose");
        let t_text = text_pose.world_to_camera.translation;
        let t_binary = binary_pose.world_to_camera.translation;
        assert!((t_text - t_binary).norm() < 1e-9, "kf {kf_id} translation");
        let q_text = text_pose.world_to_camera.rotation;
        let q_binary = binary_pose.world_to_camera.rotation;
        // Quaternion sign is gauge — compare via rotation distance.
        let dq = q_text.rotation_to(&q_binary).angle();
        assert!(dq.abs() < 1e-9, "kf {kf_id} rotation");
    }

    // Landmark world-position parity. Same id space, same camera_to_world
    // lifting → same positions.
    assert_eq!(text_map.landmarks.len(), binary_map.landmarks.len());
    for (lm_id, text_lm) in &text_map.landmarks {
        let binary_lm = binary_map
            .landmarks
            .get(lm_id)
            .expect("matching binary landmark");
        let dp = text_lm.position - binary_lm.position;
        assert!(dp.norm() < 1e-9, "lm {lm_id} position");
    }

    fs::remove_dir_all(&text_dir).ok();
    fs::remove_dir_all(&binary_dir).ok();
}

#[test]
fn write_colmap_text_model_for_3dgs_rejects_unknown_camera_model() {
    // `CameraModel::Unknown("BOGUS")` has no COLMAP model id; the text
    // writer accepts the name string in `cameras.txt`, but the binary
    // counterpart rejects it. To keep the writer pair symmetric so a
    // caller driving both off the same input gets matching outcomes,
    // the text writer rejects the same unknowns the binary writer does.
    let dir = make_tempdir("colmap_text_export_unknown_model");
    let camera = Camera {
        id: 7,
        model: CameraModel::Unknown("BOGUS".to_owned()),
        width: 640,
        height: 480,
        params: vec![500.0, 500.0, 320.0, 240.0],
    };
    let poses = vec![Pose::identity()];
    let left_features = vec![feature_set_from_keypoints(&[(100.0, 80.0)])];
    let stereo_per_frame = vec![vec![]];
    let err = write_colmap_text_model_for_3dgs(
        &dir,
        &camera,
        &poses,
        &left_features,
        &stereo_per_frame,
        |idx| format!("{idx:06}.png"),
    )
    .expect_err("unknown CameraModel should fail in the text writer too");
    assert!(matches!(err, ColmapError::InvalidExportInput(_)));
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_colmap_models_for_3dgs_reject_image_name_with_format_breaking_characters() {
    // The text writer's NAME field is delimited by ASCII whitespace and the
    // images.txt parser alternates a header line and a 2D-points line per
    // image; a NAME containing space / tab / LF / CR would either corrupt
    // the token stream or inject a spurious image record. The binary
    // writer's NAME field is NUL-terminated, so a NAME containing NUL
    // would silently truncate the filename. Sharing a single
    // `validate_colmap_image_name` between the two writers means every
    // character bad for either format is rejected by both — so a caller
    // driving both writers off the same `image_name` closure either gets
    // both files or the same structured error from each.
    let camera = synthetic_camera();
    let poses = vec![Pose::identity()];
    let left_features = vec![feature_set_from_keypoints(&[(100.0, 80.0)])];
    let stereo_per_frame: Vec<Vec<StereoFeature>> = vec![vec![]];

    for (label, bad_name) in [
        ("nul", "bad\0name.png"),
        ("space", "bad name.png"),
        ("tab", "bad\tname.png"),
        ("lf", "bad\nname.png"),
        ("cr", "bad\rname.png"),
    ] {
        let text_dir = make_tempdir(&format!("colmap_text_export_bad_name_{label}"));
        let text_err = write_colmap_text_model_for_3dgs(
            &text_dir,
            &camera,
            &poses,
            &left_features,
            &stereo_per_frame,
            |_idx| bad_name.to_owned(),
        )
        .expect_err(&format!("text writer must reject {label}"));
        assert!(
            matches!(text_err, ColmapError::InvalidExportInput(_)),
            "text writer {label}: {text_err:?}"
        );
        fs::remove_dir_all(&text_dir).ok();

        let binary_dir = make_tempdir(&format!("colmap_binary_export_bad_name_{label}"));
        let binary_err = write_colmap_binary_model_for_3dgs(
            &binary_dir,
            &camera,
            &poses,
            &left_features,
            &stereo_per_frame,
            |_idx| bad_name.to_owned(),
        )
        .expect_err(&format!("binary writer must reject {label}"));
        assert!(
            matches!(binary_err, ColmapError::InvalidExportInput(_)),
            "binary writer {label}: {binary_err:?}"
        );
        fs::remove_dir_all(&binary_dir).ok();
    }
}
