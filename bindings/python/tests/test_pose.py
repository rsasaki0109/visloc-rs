import numpy as np
import pytest

import visloc
from conftest import rotation_from_axis_angle


def test_identity_and_alias():
    pose = visloc.Pose()
    np.testing.assert_allclose(pose.matrix(), np.eye(4))
    assert visloc.SE3 is visloc.Pose
    np.testing.assert_allclose(visloc.Pose.identity().quaternion(), [1.0, 0.0, 0.0, 0.0])


def test_rotation_translation_matrix_round_trip():
    r = rotation_from_axis_angle([1.0, 2.0, 3.0], 0.7)
    t = np.array([0.5, -1.0, 2.0])
    pose = visloc.Pose(r, t)
    np.testing.assert_allclose(pose.rotation, r, atol=1e-12)
    np.testing.assert_allclose(pose.translation, t)
    m = pose.matrix()
    np.testing.assert_allclose(m[:3, :3], r, atol=1e-12)
    np.testing.assert_allclose(m[:3, 3], t)
    np.testing.assert_allclose(visloc.Pose.from_matrix(m).matrix(), m, atol=1e-12)
    np.testing.assert_allclose(visloc.Pose.from_matrix(m[:3]).matrix(), m, atol=1e-12)


def test_quaternion_orders():
    # 90 degrees about z.
    s = np.sqrt(0.5)
    wxyz = visloc.Pose.from_quaternion([s, 0.0, 0.0, s], [1.0, 2.0, 3.0])
    xyzw = visloc.Pose.from_quaternion([0.0, 0.0, s, s], [1.0, 2.0, 3.0], scalar_first=False)
    np.testing.assert_allclose(wxyz.matrix(), xyzw.matrix(), atol=1e-12)
    np.testing.assert_allclose(
        wxyz.rotation, [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-12
    )
    np.testing.assert_allclose(wxyz.quaternion(), [s, 0.0, 0.0, s], atol=1e-12)
    np.testing.assert_allclose(wxyz.quaternion(scalar_first=False), [0.0, 0.0, s, s], atol=1e-12)
    # Non-unit quaternions are normalized.
    scaled = visloc.Pose.from_quaternion([2.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(scaled.rotation, wxyz.rotation, atol=1e-12)


def test_compose_inverse_and_transform(rng):
    a = visloc.Pose.exp(rng.normal(size=6))
    b = visloc.Pose.exp(rng.normal(size=6))
    np.testing.assert_allclose((a @ b).matrix(), a.matrix() @ b.matrix(), atol=1e-12)
    np.testing.assert_allclose(a.compose(b).matrix(), (a @ b).matrix())
    np.testing.assert_allclose((a @ a.inverse()).matrix(), np.eye(4), atol=1e-12)
    points = rng.normal(size=(20, 3))
    homogeneous = np.column_stack([points, np.ones(20)])
    np.testing.assert_allclose(a.transform(points), (a.matrix() @ homogeneous.T).T[:, :3])
    assert a.transform([1.0, 2.0, 3.0]).shape == (3,)


def test_exp_log_round_trip(rng):
    xi = rng.normal(scale=0.5, size=6)
    np.testing.assert_allclose(visloc.Pose.exp(xi).log(), xi, atol=1e-10)


def test_camera_center_is_minus_rt_t():
    r = rotation_from_axis_angle([0.0, 1.0, 0.0], 0.4)
    t = np.array([1.0, 2.0, 3.0])
    pose = visloc.Pose(r, t)
    np.testing.assert_allclose(pose.camera_center(), -r.T @ t, atol=1e-12)
    np.testing.assert_allclose(pose.transform(pose.camera_center()), 0.0, atol=1e-12)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="orthonormal"):
        visloc.Pose(np.diag([1.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="orthonormal"):
        visloc.Pose(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="norm"):
        visloc.Pose.from_quaternion([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        visloc.Pose.from_matrix(np.eye(3))
    with pytest.raises(ValueError, match=r"\(3,\)"):
        visloc.Pose(np.eye(3), [1.0, 2.0])
