import numpy as np
import pytest

import visloc
from conftest import rotation_from_axis_angle


def assert_same_model(a, b):
    assert a.cameras == b.cameras
    assert sorted(a.images) == sorted(b.images)
    for image_id, image in a.images.items():
        other = b.images[image_id]
        assert other.camera_id == image.camera_id
        np.testing.assert_allclose(other.pose.matrix(), image.pose.matrix(), atol=1e-12)
        np.testing.assert_allclose(other.keypoints, image.keypoints)
        np.testing.assert_array_equal(other.point3d_ids, image.point3d_ids)
    np.testing.assert_array_equal(a.point3d_ids, b.point3d_ids)
    np.testing.assert_allclose(a.points3d, b.points3d)


def test_reads_repository_text_fixture(io_fixture):
    model = visloc.Reconstruction.read_text(io_fixture)
    assert (model.num_cameras, model.num_images, model.num_points3d) == (2, 2, 2)
    assert model.cameras[1].model == "PINHOLE"
    np.testing.assert_allclose(model.cameras[1].params, [500.0, 510.0, 320.0, 240.0])
    assert model.cameras[2].model == "SIMPLE_PINHOLE"

    image = model.images[10]
    assert image.camera_id == 1
    np.testing.assert_allclose(image.pose.translation, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(image.pose.rotation, np.eye(3))
    np.testing.assert_allclose(
        image.keypoints, [[100.0, 120.0], [200.0, 220.0], [300.0, 320.0]]
    )
    np.testing.assert_array_equal(image.point3d_ids, [1000, -1, 1001])
    assert image.point3d_ids.dtype == np.int64
    assert image.num_observations == 2

    rotated = model.images[11].pose
    np.testing.assert_allclose(
        rotated.rotation, rotation_from_axis_angle([0.0, 1.0, 0.0], np.pi / 4), atol=1e-9
    )

    np.testing.assert_array_equal(model.point3d_ids, [1000, 1001])
    assert model.point3d_ids.dtype == np.uint64
    np.testing.assert_allclose(model.points3d, [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])
    assert model.point3d_descriptors is None
    assert model.validate() == []


def test_text_and_binary_round_trips(io_fixture, tmp_path):
    model = visloc.Reconstruction.read(io_fixture)
    model.write_text(tmp_path / "text")
    model.write_binary(tmp_path / "binary")
    assert {p.name for p in (tmp_path / "binary").iterdir()} == {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    }
    from_text = visloc.Reconstruction.read_text(tmp_path / "text")
    from_binary = visloc.Reconstruction.read_binary(tmp_path / "binary")
    assert_same_model(model, from_text)
    assert_same_model(model, from_binary)
    # `read` prefers the binary files and accepts str paths.
    assert_same_model(model, visloc.Reconstruction.read(str(tmp_path / "binary")))
    # Point tracks are rebuilt from the image observations when writing.
    points_txt = (tmp_path / "text" / "points3D.txt").read_text()
    assert "1001 -1 0.5 4 255 255 255 0 10 2 11 0" in points_txt


def test_build_reconstruction_from_numpy(tmp_path, rng):
    camera = visloc.Camera("OPENCV", 640, 480, [500, 500, 320, 240, -0.1, 0.01, 0.0, 0.0], id=7)
    points = rng.uniform(-1.0, 1.0, (30, 3)) + [0.0, 0.0, 5.0]
    ids = np.arange(100, 130)
    model = visloc.Reconstruction()
    model.add_camera(camera)
    model.set_points3d(ids, points, descriptors=rng.normal(size=(30, 16)))
    for image_id, angle in [(1, 0.0), (2, 0.1)]:
        pose = visloc.Pose(rotation_from_axis_angle([0, 1, 0], angle), [angle, 0.0, 0.0])
        keypoints = camera.project(points, pose=pose)
        point_ids = ids.copy()
        point_ids[::5] = -1
        model.add_image(visloc.Image(image_id, 7, pose, keypoints, point_ids))
    assert model.validate() == []
    assert model.point3d_descriptors.shape == (30, 16)
    assert model.point3d_descriptors.dtype == np.float32
    assert "points3d=30" in repr(model)

    model.write_binary(tmp_path)
    loaded = visloc.Reconstruction.read_binary(tmp_path)
    assert_same_model(model, loaded)
    # Descriptors are not part of the COLMAP files.
    assert loaded.point3d_descriptors is None


def test_validate_reports_missing_camera():
    model = visloc.Reconstruction()
    model.add_image(visloc.Image(1, 99, visloc.Pose()))
    issues = model.validate()
    assert len(issues) == 1
    assert "MissingCameraForKeyframe" in issues[0]


def test_load_point3d_descriptors(example_data):
    model = visloc.Reconstruction.read(example_data / "colmap_text")
    assert model.load_point3d_descriptors(example_data / "landmark_descriptors.txt") == 8
    descriptors = model.point3d_descriptors
    assert descriptors.shape == (8, 3)
    np.testing.assert_allclose(descriptors[:, 0], np.arange(8))


def test_io_and_input_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        visloc.Reconstruction.read(tmp_path)
    with pytest.raises(OSError):
        visloc.Reconstruction.read_text(tmp_path / "missing")
    (tmp_path / "cameras.txt").write_text("1 PINHOLE\n")
    with pytest.raises(ValueError, match="invalid COLMAP line"):
        visloc.Reconstruction.read_text(tmp_path)

    model = visloc.Reconstruction()
    with pytest.raises(ValueError, match="duplicate"):
        model.set_points3d([1, 1], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="rows"):
        model.set_points3d([1, 2, 3], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        model.set_points3d([-5], np.zeros((1, 3)))
    with pytest.raises(ValueError, match="keypoints"):
        visloc.Image(1, 1, visloc.Pose(), np.zeros((3, 2)), [1, 2])

    model.add_camera(visloc.Camera("DOUBLE_SPHERE", 640, 480, [1, 1, 1, 1, 0.0, 0.5]))
    with pytest.raises(ValueError, match="COLMAP"):
        model.write_binary(tmp_path / "out")
