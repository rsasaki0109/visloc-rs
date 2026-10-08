use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use visloc_core::geometry::Pose;
use visloc_core::types::{
    Camera, CameraModel, Frame, Keyframe, Landmark, Observation, VisualMap,
    VisualMapValidationIssue,
};
use visloc_io::colmap::{
    format_cameras_txt, format_images_txt, format_points3d_txt, parse_cameras_bin,
    parse_cameras_txt, parse_images_bin, parse_images_txt, parse_points3d_bin, parse_points3d_txt,
    read_colmap_binary_model, read_colmap_text_model, write_colmap_binary_model,
    write_colmap_text_model, ColmapError, ColmapMapProvider, ColmapMapProviderError,
};
use visloc_localization::{DescriptorProvider, MapProvider};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("colmap_text")
}

fn descriptor_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("descriptors")
        .join("landmarks.txt")
}

fn binary_fixture_dir() -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must be after UNIX epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("visloc_colmap_binary_fixture_{suffix}"))
}

#[test]
fn parses_colmap_cameras_txt() {
    let cameras = parse_cameras_txt(include_str!("fixtures/colmap_text/cameras.txt")).unwrap();

    assert_eq!(cameras.len(), 2);
    assert_eq!(cameras[0].id, 1);
    assert_eq!(cameras[0].model, CameraModel::Pinhole);
    assert_eq!(cameras[0].intrinsics(), Some((500.0, 510.0, 320.0, 240.0)));
    assert_eq!(cameras[1].model, CameraModel::SimplePinhole);
    assert_eq!(cameras[1].intrinsics(), Some((700.0, 700.0, 400.0, 300.0)));
}

#[test]
fn parses_colmap_cameras_bin() {
    let cameras = parse_cameras_bin(&camera_bin()).unwrap();

    assert_eq!(cameras.len(), 2);
    assert_eq!(cameras[0].id, 1);
    assert_eq!(cameras[0].model, CameraModel::Pinhole);
    assert_eq!(cameras[0].intrinsics(), Some((500.0, 510.0, 320.0, 240.0)));
    assert_eq!(cameras[1].id, 2);
    assert_eq!(cameras[1].model, CameraModel::SimplePinhole);
    assert_eq!(cameras[1].intrinsics(), Some((700.0, 700.0, 400.0, 300.0)));
}

#[test]
fn parses_colmap_points3d_txt() {
    let landmarks = parse_points3d_txt(include_str!("fixtures/colmap_text/points3D.txt")).unwrap();

    assert_eq!(landmarks.len(), 2);
    assert_eq!(landmarks[0].id, 1000);
    assert_eq!(landmarks[0].position.x, 1.0);
    assert_eq!(landmarks[1].id, 1001);
    assert_eq!(landmarks[1].position.z, 4.0);
}

#[test]
fn parses_colmap_points3d_bin() {
    let landmarks = parse_points3d_bin(&points3d_bin()).unwrap();

    assert_eq!(landmarks.len(), 2);
    assert_eq!(landmarks[0].id, 1000);
    assert_eq!(landmarks[0].position.x, 1.0);
    assert_eq!(landmarks[1].id, 1001);
    assert_eq!(landmarks[1].position.z, 4.0);
}

#[test]
fn parses_colmap_images_txt() {
    let keyframes = parse_images_txt(include_str!("fixtures/colmap_text/images.txt")).unwrap();

    assert_eq!(keyframes.len(), 2);
    assert_eq!(keyframes[0].frame.id, 10);
    assert_eq!(keyframes[0].frame.camera_id, 1);
    assert_eq!(keyframes[0].frame.keypoints.len(), 3);
    assert_eq!(keyframes[0].observations.len(), 2);
    assert_eq!(keyframes[0].observations[0].landmark_id, 1000);
    assert_eq!(keyframes[0].observations[1].keypoint_index, 2);
    assert_eq!(keyframes[1].frame.id, 11);
    assert_eq!(keyframes[1].observations[0].landmark_id, 1001);
}

#[test]
fn parses_colmap_images_bin() {
    let keyframes = parse_images_bin(&images_bin()).unwrap();

    assert_eq!(keyframes.len(), 2);
    assert_eq!(keyframes[0].frame.id, 10);
    assert_eq!(keyframes[0].frame.camera_id, 1);
    assert_eq!(keyframes[0].frame.keypoints.len(), 3);
    assert_eq!(keyframes[0].observations.len(), 2);
    assert_eq!(keyframes[0].observations[0].landmark_id, 1000);
    assert_eq!(keyframes[0].observations[1].keypoint_index, 2);
    assert_eq!(keyframes[1].frame.id, 11);
    assert_eq!(keyframes[1].observations[0].landmark_id, 1001);
}

#[test]
fn rejects_huge_colmap_binary_counts_before_allocation() {
    let mut cameras = Vec::new();
    push_u64(&mut cameras, u64::MAX);
    assert_invalid_binary(parse_cameras_bin(&cameras), "cameras.bin", "camera_count");

    let mut images = Vec::new();
    push_u64(&mut images, u64::MAX);
    assert_invalid_binary(parse_images_bin(&images), "images.bin", "image_count");

    let mut image_points = Vec::new();
    push_u64(&mut image_points, 1);
    push_image_header(&mut image_points, 10, 1, "");
    push_u64(&mut image_points, u64::MAX);
    assert_invalid_binary(parse_images_bin(&image_points), "images.bin", "point_count");

    let mut points = Vec::new();
    push_u64(&mut points, u64::MAX);
    assert_invalid_binary(parse_points3d_bin(&points), "points3D.bin", "point3D_count");

    assert_invalid_binary(
        parse_points3d_bin(&points3d_bin_with_track_length(u64::MAX)),
        "points3D.bin",
        "track_length",
    );
}

#[test]
fn rejects_truncated_colmap_binary_counts_from_remaining_bytes() {
    let mut cameras = Vec::new();
    push_u64(&mut cameras, 1);
    assert_invalid_binary(parse_cameras_bin(&cameras), "cameras.bin", "remaining");

    let mut images = Vec::new();
    push_u64(&mut images, 1);
    assert_invalid_binary(parse_images_bin(&images), "images.bin", "remaining");

    let mut image_points = Vec::new();
    push_u64(&mut image_points, 1);
    push_image_header(&mut image_points, 10, 1, "");
    push_u64(&mut image_points, 1);
    assert_invalid_binary(parse_images_bin(&image_points), "images.bin", "remaining");

    let mut points = Vec::new();
    push_u64(&mut points, 1);
    assert_invalid_binary(parse_points3d_bin(&points), "points3D.bin", "remaining");

    assert_invalid_binary(
        parse_points3d_bin(&points3d_bin_with_track_length(1)),
        "points3D.bin",
        "remaining",
    );
}

#[test]
fn reads_colmap_text_model_from_directory() {
    let map = read_colmap_text_model(fixture_dir()).unwrap();

    assert_eq!(map.cameras.len(), 2);
    assert_eq!(map.landmarks.len(), 2);
    assert_eq!(map.keyframes.len(), 2);
    assert!(map.cameras.contains_key(&1));
    assert!(map.landmarks.contains_key(&1000));
    assert!(map.keyframes.contains_key(&10));
}

#[test]
fn reads_colmap_binary_model_from_directory() {
    let dir = binary_fixture_dir();
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("cameras.bin"), camera_bin()).unwrap();
    fs::write(dir.join("images.bin"), images_bin()).unwrap();
    fs::write(dir.join("points3D.bin"), points3d_bin()).unwrap();

    let map = read_colmap_binary_model(&dir).unwrap();

    assert_eq!(map.cameras.len(), 2);
    assert_eq!(map.landmarks.len(), 2);
    assert_eq!(map.keyframes.len(), 2);
    assert!(map.cameras.contains_key(&1));
    assert!(map.landmarks.contains_key(&1000));
    assert!(map.keyframes.contains_key(&10));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn colmap_map_provider_loads_binary_model() {
    let dir = binary_fixture_dir();
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("cameras.bin"), camera_bin()).unwrap();
    fs::write(dir.join("images.bin"), images_bin()).unwrap();
    fs::write(dir.join("points3D.bin"), points3d_bin()).unwrap();

    let provider = ColmapMapProvider::from_binary_model_dir_validated(&dir).unwrap();

    assert_eq!(provider.visual_map().cameras.len(), 2);
    assert_eq!(provider.visual_map().landmarks.len(), 2);
    assert!(provider.validate_map().is_valid());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn colmap_map_provider_loads_text_model() {
    let provider = ColmapMapProvider::from_text_model_dir(fixture_dir()).unwrap();
    let map = provider.visual_map();

    assert_eq!(map.cameras.len(), 2);
    assert_eq!(map.landmarks.len(), 2);
    assert!(provider.landmark_descriptor_store().is_none());
    assert!(provider.validate_map().is_valid());
}

#[test]
fn colmap_map_provider_loads_text_model_with_descriptors() {
    let provider = ColmapMapProvider::from_text_model_dir_with_descriptors(
        fixture_dir(),
        descriptor_fixture_path(),
    )
    .unwrap();
    let map = provider.visual_map();
    let descriptor_store = provider.landmark_descriptor_store().unwrap();

    assert_eq!(map.landmarks.len(), 2);
    assert_eq!(descriptor_store.len(), 2);
    assert_eq!(descriptor_store.get(1000).unwrap(), &[0.1, 0.2, 0.3, 0.4]);
    assert!(provider.validate_for_localization().is_valid());
}

#[test]
fn colmap_map_provider_reports_missing_localization_descriptors() {
    let provider = ColmapMapProvider::from_text_model_dir(fixture_dir()).unwrap();

    let report = provider.validate_for_localization();

    assert!(!report.is_valid());
    assert_eq!(report.issue_count(), 2);
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::MissingDescriptorForLandmark { landmark_id: 1000 }));
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::MissingDescriptorForLandmark { landmark_id: 1001 }));
}

#[test]
fn colmap_map_provider_validated_constructors_check_expected_inputs() {
    let structure_only_provider =
        ColmapMapProvider::from_text_model_dir_validated(fixture_dir()).unwrap();
    let localization_provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        fixture_dir(),
        descriptor_fixture_path(),
    )
    .unwrap();

    assert!(structure_only_provider.validate_map().is_valid());
    assert!(localization_provider.validate_for_localization().is_valid());
}

#[test]
fn colmap_map_provider_rejects_invalid_descriptor_store_when_validated() {
    let error = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        fixture_dir(),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("descriptors")
            .join("with_unknown_landmark.txt"),
    )
    .unwrap_err();

    let ColmapMapProviderError::InvalidMap(report) = error else {
        panic!("expected invalid map error");
    };
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::DescriptorForMissingLandmark { landmark_id: 9999 }));
}

#[test]
fn formats_colmap_text_model_sections() {
    let map = writable_map();

    let cameras = format_cameras_txt(&map);
    let images = format_images_txt(&map);
    let points = format_points3d_txt(&map);

    assert!(cameras.contains("1 PINHOLE 640 480 500 510 320 240"));
    assert!(images.contains("10 1 0 0 0 0.1 0.2 0.3 1 image_10.jpg"));
    assert!(images.contains("320 240 1000 10 20 -1 400 200 1001"));
    assert!(points.contains("1000 1 2 3 255 255 255 0 10 0"));
    assert!(points.contains("1001 -1 0.5 4 255 255 255 0 10 2 11 0"));
}

#[test]
fn writes_and_reads_colmap_text_model_round_trip() {
    let dir = binary_fixture_dir();
    let map = writable_map();

    write_colmap_text_model(&map, &dir).unwrap();
    let loaded = read_colmap_text_model(&dir).unwrap();

    assert_eq!(loaded.cameras.len(), 2);
    assert_eq!(loaded.keyframes.len(), 2);
    assert_eq!(loaded.landmarks.len(), 2);
    assert!(loaded.validate().is_valid());
    assert_eq!(
        loaded.keyframes.get(&10).unwrap().observations[1].landmark_id,
        1001
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn writes_and_reads_colmap_binary_model_round_trip() {
    let text_dir = binary_fixture_dir().join("text");
    let binary_dir = binary_fixture_dir().join("binary");
    let map = writable_map();

    write_colmap_text_model(&map, &text_dir).unwrap();
    write_colmap_binary_model(&map, &binary_dir).unwrap();
    let from_text = read_colmap_text_model(&text_dir).unwrap();
    let from_binary = read_colmap_binary_model(&binary_dir).unwrap();

    assert!(from_binary.validate().is_valid());
    // The binary writer mirrors the text writer record-for-record.
    assert_eq!(from_binary, from_text);
    assert_eq!(from_binary.cameras, map.cameras);
    assert_eq!(
        from_binary.keyframes.get(&10).unwrap().observations[1].landmark_id,
        1001
    );
    for (id, landmark) in &map.landmarks {
        assert_eq!(from_binary.landmarks[id].position, landmark.position);
    }
    // points3D.bin carries the same TRACK[] as points3D.txt.
    let points = fs::read(binary_dir.join("points3D.bin")).unwrap();
    let track_entries = map
        .landmarks
        .values()
        .map(|landmark| landmark.observations.len())
        .sum::<usize>();
    assert_eq!(
        points.len(),
        8 + map.landmarks.len() * 51 + track_entries * 8
    );
    fs::remove_dir_all(text_dir.parent().unwrap()).unwrap();
}

#[test]
fn binary_writer_rejects_ids_beyond_colmap_u32_range() {
    let dir = binary_fixture_dir();
    let mut map = writable_map();
    let mut camera = map.cameras.remove(&2).unwrap();
    camera.id = u64::from(u32::MAX) + 1;
    map.cameras.insert(camera.id, camera);

    let error = write_colmap_binary_model(&map, &dir).unwrap_err();
    assert!(
        matches!(error, ColmapError::InvalidExportInput(_)),
        "{error}"
    );
    assert!(!dir.exists());
}

fn camera_bin() -> Vec<u8> {
    let mut bytes = Vec::new();
    push_u64(&mut bytes, 2);

    push_u32(&mut bytes, 1);
    push_i32(&mut bytes, 1);
    push_u64(&mut bytes, 640);
    push_u64(&mut bytes, 480);
    for value in [500.0, 510.0, 320.0, 240.0] {
        push_f64(&mut bytes, value);
    }

    push_u32(&mut bytes, 2);
    push_i32(&mut bytes, 0);
    push_u64(&mut bytes, 800);
    push_u64(&mut bytes, 600);
    for value in [700.0, 400.0, 300.0] {
        push_f64(&mut bytes, value);
    }

    bytes
}

fn writable_map() -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(
        1,
        Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 640,
            height: 480,
            params: vec![500.0, 510.0, 320.0, 240.0],
        },
    );
    map.cameras.insert(
        2,
        Camera {
            id: 2,
            model: CameraModel::SimplePinhole,
            width: 800,
            height: 600,
            params: vec![700.0, 400.0, 300.0],
        },
    );

    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.1, 0.2, 0.3));
    let mut frame_a = Frame::new(10, 1);
    frame_a.pose = Some(pose.clone());
    frame_a.keypoints = vec![
        Point2::new(320.0, 240.0),
        Point2::new(10.0, 20.0),
        Point2::new(400.0, 200.0),
    ];
    let observations_a = vec![
        Observation {
            frame_id: 10,
            landmark_id: 1000,
            keypoint_index: 0,
            xy: frame_a.keypoints[0],
        },
        Observation {
            frame_id: 10,
            landmark_id: 1001,
            keypoint_index: 2,
            xy: frame_a.keypoints[2],
        },
    ];
    map.keyframes.insert(
        10,
        Keyframe {
            frame: frame_a,
            observations: observations_a.clone(),
        },
    );

    let mut frame_b = Frame::new(11, 2);
    frame_b.pose = Some(pose);
    frame_b.keypoints = vec![Point2::new(123.0, 456.0)];
    let observations_b = vec![Observation {
        frame_id: 11,
        landmark_id: 1001,
        keypoint_index: 0,
        xy: frame_b.keypoints[0],
    }];
    map.keyframes.insert(
        11,
        Keyframe {
            frame: frame_b,
            observations: observations_b.clone(),
        },
    );

    let mut landmark_a = Landmark::new(1000, Point3::new(1.0, 2.0, 3.0));
    landmark_a.observations = vec![observations_a[0].clone()];
    map.landmarks.insert(1000, landmark_a);

    let mut landmark_b = Landmark::new(1001, Point3::new(-1.0, 0.5, 4.0));
    landmark_b.observations = vec![observations_a[1].clone(), observations_b[0].clone()];
    map.landmarks.insert(1001, landmark_b);

    map
}

fn images_bin() -> Vec<u8> {
    let mut bytes = Vec::new();
    push_u64(&mut bytes, 2);

    push_image_header(&mut bytes, 10, 1, "image_a.jpg");
    push_u64(&mut bytes, 3);
    push_point2d(&mut bytes, 320.0, 240.0, 1000);
    push_point2d(&mut bytes, 10.0, 20.0, -1);
    push_point2d(&mut bytes, 400.0, 200.0, 1001);

    push_image_header(&mut bytes, 11, 2, "image_b.jpg");
    push_u64(&mut bytes, 1);
    push_point2d(&mut bytes, 123.0, 456.0, 1001);

    bytes
}

fn points3d_bin() -> Vec<u8> {
    let mut bytes = Vec::new();
    push_u64(&mut bytes, 2);

    push_point3d(&mut bytes, 1000, [1.0, 2.0, 3.0], &[(10, 0)]);
    push_point3d(&mut bytes, 1001, [-1.0, 0.5, 4.0], &[(10, 2), (11, 0)]);

    bytes
}

fn points3d_bin_with_track_length(track_length: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_u64(&mut bytes, 1);
    push_u64(&mut bytes, 1000);
    for value in [1.0, 2.0, 3.0] {
        push_f64(&mut bytes, value);
    }
    bytes.extend_from_slice(&[255, 128, 0]);
    push_f64(&mut bytes, 0.25);
    push_u64(&mut bytes, track_length);
    bytes
}

fn push_image_header(bytes: &mut Vec<u8>, image_id: u32, camera_id: u32, name: &str) {
    push_u32(bytes, image_id);
    for value in [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3] {
        push_f64(bytes, value);
    }
    push_u32(bytes, camera_id);
    bytes.extend_from_slice(name.as_bytes());
    bytes.push(0);
}

fn push_point2d(bytes: &mut Vec<u8>, x: f64, y: f64, point_id: i64) {
    push_f64(bytes, x);
    push_f64(bytes, y);
    push_i64(bytes, point_id);
}

fn push_point3d(bytes: &mut Vec<u8>, id: u64, xyz: [f64; 3], track: &[(u32, u32)]) {
    push_u64(bytes, id);
    for value in xyz {
        push_f64(bytes, value);
    }
    bytes.extend_from_slice(&[255, 128, 0]);
    push_f64(bytes, 0.25);
    push_u64(bytes, track.len() as u64);
    for (image_id, point2d_index) in track {
        push_u32(bytes, *image_id);
        push_u32(bytes, *point2d_index);
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn assert_invalid_binary<T>(
    result: Result<T, ColmapError>,
    expected_file: &'static str,
    message_fragment: &str,
) {
    let error = match result {
        Ok(_) => panic!("malformed binary unexpectedly parsed"),
        Err(error) => error,
    };
    match error {
        ColmapError::InvalidBinary { file, message } => {
            assert_eq!(file, expected_file);
            assert!(
                message.contains(message_fragment),
                "expected {message_fragment:?} in {message:?}"
            );
        }
        other => panic!("expected InvalidBinary, got {other:?}"),
    }
}

#[test]
fn imported_opencv_camera_applies_tangential_distortion() {
    // A real COLMAP OPENCV camera with non-zero p1/p2 used to be projected with
    // k1/k2 only. Reference pixel from OpenCV 4.10 `cv2.projectPoints`.
    let cameras = parse_cameras_txt(
        "1 OPENCV 1920 1080 1353.09 1338.03 962.7 539.93 0.0346 -0.0235 0.0012 -0.0008\n",
    )
    .unwrap();
    let pixel = cameras[0].project(&Point3::new(0.5, -0.3, 1.0)).unwrap();
    assert!(
        (pixel.x - 1643.9695843529998).abs() < 1e-8 && (pixel.y - 136.0453501334).abs() < 1e-8,
        "got {pixel:?}"
    );
    let ray = cameras[0].normalize_pixel(&pixel).unwrap();
    assert!(
        (ray.x - 0.5).abs() < 1e-9 && (ray.y + 0.3).abs() < 1e-9,
        "got {ray:?}"
    );
}
