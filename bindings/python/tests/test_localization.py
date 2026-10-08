import numpy as np
import pytest

import visloc
from conftest import rotation_from_axis_angle


def synthetic_scene(rng, count=60, descriptor_dim=32):
    camera = visloc.Camera.pinhole(500.0, 500.0, 320.0, 240.0, 640, 480)
    points = np.column_stack(
        [rng.uniform(-2.0, 2.0, count), rng.uniform(-1.5, 1.5, count), rng.uniform(4.0, 9.0, count)]
    )
    descriptors = rng.normal(size=(count, descriptor_dim)).astype(np.float32)
    model = visloc.Reconstruction()
    model.set_points3d(np.arange(1, count + 1), points, descriptors)
    pose = visloc.Pose(rotation_from_axis_angle([0.2, 1.0, 0.1], 0.15), [0.3, -0.1, 0.4])
    return camera, model, points, descriptors, pose


def test_localize_recovers_query_pose(rng):
    camera, model, points, descriptors, true_pose = synthetic_scene(rng)
    keypoints = camera.project(points, pose=true_pose)
    keypoints += rng.normal(scale=0.3, size=keypoints.shape)
    # Shuffle the query so indices differ from point ids.
    order = rng.permutation(len(points))
    result = visloc.localize(camera, keypoints[order], descriptors[order], model, min_inliers=12)
    assert result.success and bool(result)
    assert result.failure_reason is None
    assert result.inlier_count >= 50
    assert result.mean_reprojection_error < 1.0
    np.testing.assert_allclose(result.pose.matrix(), true_pose.matrix(), atol=2e-2)
    # Inlier bookkeeping maps query indices back to the right 3-D points.
    np.testing.assert_array_equal(
        result.inlier_point3d_ids, order[result.inlier_query_indices] + 1
    )
    assert result.inlier_reprojection_errors.shape == (result.inlier_count,)
    assert "success=True" in repr(result)


def test_localize_tolerates_outlier_matches(rng):
    camera, model, points, descriptors, true_pose = synthetic_scene(rng, count=80)
    keypoints = camera.project(points, pose=true_pose)
    corrupt = rng.choice(80, size=20, replace=False)
    keypoints[corrupt] = rng.uniform([0, 0], [640, 480], size=(20, 2))
    result = visloc.localize(
        camera, keypoints, descriptors, model, ransac_iterations=300, min_inliers=20
    )
    assert result.success
    assert result.outlier_count >= 15
    np.testing.assert_allclose(result.pose.camera_center(), true_pose.camera_center(), atol=1e-3)


def test_localize_failure_reasons(rng):
    camera, model, points, descriptors, pose = synthetic_scene(rng)
    keypoints = camera.project(points, pose=pose)
    no_descriptors = visloc.Reconstruction()
    no_descriptors.set_points3d(np.arange(len(points)), points)
    result = visloc.localize(camera, keypoints, descriptors, no_descriptors)
    assert not result.success
    assert result.pose is None
    assert result.failure_reason == "no_map_descriptors"

    gated = visloc.localize(camera, keypoints, descriptors, model, min_inliers=1000)
    assert not gated.success
    assert gated.failure_reason == "quality_gate_failed"

    with pytest.raises(ValueError, match="descriptors"):
        visloc.localize(camera, keypoints, descriptors[:5], model)


def test_localize_repository_example(example_data):
    model = visloc.Reconstruction.read_text(example_data / "colmap_text")
    model.load_point3d_descriptors(example_data / "landmark_descriptors.txt")
    query = np.loadtxt(example_data / "query_features.txt")
    result = visloc.localize(model.cameras[1], query[:, :2], query[:, 2:], model)
    assert result.success
    assert result.inlier_count == 8
    np.testing.assert_allclose(result.pose.matrix(), np.eye(4), atol=1e-3)


def test_pnp_ransac_from_correspondences(rng):
    camera = visloc.Camera("OPENCV", 640, 480, [450, 460, 320, 240, -0.05, 0.01, 0.0, 0.0])
    true_pose = visloc.Pose.exp([0.2, -0.1, 0.3, 0.05, 0.1, -0.03])
    world = rng.uniform(-1.5, 1.5, (100, 3)) + [0.0, 0.0, 6.0]
    pixels = camera.project(world, pose=true_pose)
    pixels[:25] = rng.uniform([0, 0], [640, 480], size=(25, 2))
    result = visloc.estimate_pose_pnp_ransac(
        camera, pixels, world, ransac_iterations=500, reprojection_threshold=2.0, seed=3
    )
    assert result is not None
    assert result.refinement_applied
    assert result.inlier_count >= 75
    assert set(range(25, 100)) <= set(result.inliers.tolist())
    np.testing.assert_allclose(result.pose.matrix(), true_pose.matrix(), atol=1e-6)

    again = visloc.estimate_pose_pnp_ransac(
        camera, pixels, world, ransac_iterations=500, reprojection_threshold=2.0, seed=3
    )
    np.testing.assert_array_equal(again.pose.matrix(), result.pose.matrix())


def test_pnp_ransac_input_errors_and_failure(rng):
    camera = visloc.Camera.pinhole(500.0, 500.0, 320.0, 240.0, 640, 480)
    with pytest.raises(ValueError, match="same N"):
        visloc.estimate_pose_pnp_ransac(camera, np.zeros((5, 2)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="ransac_iterations"):
        visloc.estimate_pose_pnp_ransac(
            camera, np.zeros((6, 2)), np.ones((6, 3)), ransac_iterations=0
        )
    assert visloc.estimate_pose_pnp_ransac(camera, np.zeros((3, 2)), np.ones((3, 3))) is None
