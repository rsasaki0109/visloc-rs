import numpy as np
import pytest

import visloc

MODELS = [
    ("SIMPLE_PINHOLE", [500.0, 320.0, 240.0]),
    ("PINHOLE", [500.0, 510.0, 320.0, 240.0]),
    ("PINHOLE", [500.0, 510.0, 320.0, 240.0, -0.1, 0.01]),
    ("SIMPLE_RADIAL", [500.0, 320.0, 240.0, -0.05]),
    ("RADIAL", [500.0, 320.0, 240.0, -0.05, 0.01]),
    ("OPENCV", [500.0, 510.0, 320.0, 240.0, -0.1, 0.01, 0.001, -0.002]),
    ("FULL_OPENCV", [500.0, 510.0, 320.0, 240.0, -0.1, 0.01, 0.001, -0.002, 0.0, 0.0, 0.0, 0.0]),
    ("OPENCV_FISHEYE", [300.0, 300.0, 320.0, 240.0, 0.01, -0.005, 0.001, 0.0]),
    ("SIMPLE_RADIAL_FISHEYE", [300.0, 320.0, 240.0, 0.01]),
    ("RADIAL_FISHEYE", [300.0, 320.0, 240.0, 0.01, -0.002]),
    ("FOV", [300.0, 300.0, 320.0, 240.0, 0.9]),
    ("DOUBLE_SPHERE", [300.0, 300.0, 320.0, 240.0, -0.2, 0.6]),
]


def test_pinhole_projects_like_the_calibration_matrix():
    camera = visloc.Camera.pinhole(500.0, 510.0, 320.0, 240.0, 640, 480)
    assert camera.model == "PINHOLE"
    assert (camera.width, camera.height, camera.id) == (640, 480, 1)
    assert camera.intrinsics == (500.0, 510.0, 320.0, 240.0)
    k = camera.calibration_matrix()
    points = np.array([[0.1, -0.2, 2.0], [0.0, 0.0, 1.0], [-0.5, 0.3, 4.0]])
    expected = (k @ points.T).T
    expected = expected[:, :2] / expected[:, 2:]
    np.testing.assert_allclose(camera.project(points), expected)
    # A single point keeps a 1-D shape.
    single = camera.project([0.0, 0.0, 1.0])
    assert single.shape == (2,)
    np.testing.assert_allclose(single, [320.0, 240.0])


def test_points_behind_the_camera_project_to_nan():
    camera = visloc.Camera.pinhole(500.0, 500.0, 320.0, 240.0, 640, 480)
    pixels = camera.project([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])
    assert np.isnan(pixels[0]).all()
    assert np.isfinite(pixels[1]).all()


@pytest.mark.parametrize("model,params", MODELS)
def test_project_unproject_round_trip(model, params, rng):
    camera = visloc.Camera(model, 640, 480, params, id=3)
    assert camera.model == model
    np.testing.assert_allclose(camera.params, params)
    points = np.column_stack(
        [rng.uniform(-0.4, 0.4, 50), rng.uniform(-0.3, 0.3, 50), rng.uniform(1.0, 5.0, 50)]
    )
    pixels = camera.project(points)
    assert pixels.shape == (50, 2)
    assert np.isfinite(pixels).all()
    rays = camera.unproject(pixels)
    assert rays.shape == (50, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)
    expected = points / np.linalg.norm(points, axis=1, keepdims=True)
    np.testing.assert_allclose(rays, expected, atol=1e-6)
    normalized = camera.normalize(pixels)
    np.testing.assert_allclose(normalized, points[:, :2] / points[:, 2:], atol=1e-5)


def test_project_with_world_to_camera_pose(rng):
    camera = visloc.Camera.pinhole(400.0, 400.0, 320.0, 240.0, 640, 480)
    pose = visloc.Pose.exp([0.1, -0.2, 0.3, 0.05, -0.02, 0.1])
    world = rng.uniform(-1.0, 1.0, (10, 3)) + [0.0, 0.0, 6.0]
    np.testing.assert_allclose(
        camera.project(world, pose=pose), camera.project(pose.transform(world))
    )


def test_camera_validation_errors():
    with pytest.raises(ValueError, match="unsupported camera model"):
        visloc.Camera("NOT_A_MODEL", 640, 480, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expects"):
        visloc.Camera("PINHOLE", 640, 480, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        visloc.Camera("PINHOLE", 640, 480, [1.0, np.nan, 3.0, 4.0])
    camera = visloc.Camera.pinhole(1.0, 1.0, 0.0, 0.0, 10, 10)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        camera.project(np.zeros((4, 2)))


def test_camera_equality_and_model_case():
    a = visloc.Camera("pinhole", 640, 480, np.array([1, 2, 3, 4]))
    b = visloc.Camera.pinhole(1.0, 2.0, 3.0, 4.0, 640, 480)
    assert a == b
    assert a != visloc.Camera.pinhole(1.0, 2.0, 3.0, 4.0, 640, 480, id=2)
    assert "PINHOLE" in repr(a)
