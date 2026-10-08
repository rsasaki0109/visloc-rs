import numpy as np
import pytest

import visloc
from conftest import rotation_from_axis_angle


def circle_trajectory(n=50):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t) * 3.0, np.sin(t) * 2.0, 0.3 * np.sin(2.0 * t)])


def similarity(points, scale, rotation, translation):
    return scale * points @ rotation.T + translation


def test_umeyama_recovers_similarity():
    source = circle_trajectory()
    rotation = rotation_from_axis_angle([1.0, 0.5, -0.2], 0.8)
    target = similarity(source, 2.5, rotation, [1.0, -2.0, 0.5])

    sim3 = visloc.umeyama_alignment(source, target, with_scale=True)
    assert sim3.scale == pytest.approx(2.5)
    np.testing.assert_allclose(sim3.rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(sim3.translation, [1.0, -2.0, 0.5], atol=1e-10)
    np.testing.assert_allclose(sim3.apply(source), target, atol=1e-9)
    homogeneous = np.column_stack([source, np.ones(len(source))])
    np.testing.assert_allclose((sim3.matrix() @ homogeneous.T).T[:, :3], target, atol=1e-9)

    se3 = visloc.umeyama_alignment(source, target)
    assert se3.scale == 1.0

    with pytest.raises(ValueError, match="degenerate"):
        visloc.umeyama_alignment(np.zeros((4, 3)), np.ones((4, 3)))


def test_ate_alignment_modes():
    reference = circle_trajectory()
    rotation = rotation_from_axis_angle([0.0, 0.0, 1.0], 0.6)
    rigid = similarity(reference, 1.0, rotation, [5.0, 1.0, -1.0])
    scaled = similarity(reference, 0.4, rotation, [5.0, 1.0, -1.0])

    assert visloc.evaluate_ate(rigid, reference).rmse == pytest.approx(0.0, abs=1e-9)
    assert visloc.evaluate_ate(rigid, reference, alignment="none").rmse > 1.0
    # Rigid alignment cannot absorb a scale change, Sim(3) can.
    assert visloc.evaluate_ate(scaled, reference, alignment="se3").rmse > 0.5
    result = visloc.evaluate_ate(scaled, reference, alignment="sim3")
    assert result.rmse == pytest.approx(0.0, abs=1e-9)
    assert result.alignment.scale == pytest.approx(2.5)
    np.testing.assert_allclose(result.alignment.apply(scaled), reference, atol=1e-9)

    shifted = reference + [1.0, 2.0, 3.0]
    first = visloc.evaluate_ate(shifted, reference, alignment="first")
    assert first.rmse == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(first.alignment.translation, [-1.0, -2.0, -3.0])

    with pytest.raises(ValueError, match="alignment"):
        visloc.evaluate_ate(rigid, reference, alignment="affine")


def test_ate_statistics_match_numpy(rng):
    reference = circle_trajectory(40)
    estimated = reference + rng.normal(scale=0.05, size=reference.shape)
    result = visloc.evaluate_ate(estimated, reference, alignment="none")
    errors = np.linalg.norm(estimated - reference, axis=1)
    np.testing.assert_allclose(result.errors, errors)
    assert result.rmse == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert result.mean == pytest.approx(errors.mean())
    assert result.median == pytest.approx(np.median(errors))
    assert result.std == pytest.approx(errors.std())
    assert result.min == pytest.approx(errors.min())
    assert result.max == pytest.approx(errors.max())
    np.testing.assert_array_equal(result.frame_ids, np.arange(40))
    assert result.matched_count == 40


def test_ate_matches_by_frame_id():
    reference = circle_trajectory(10)
    estimated = reference[[1, 3, 5, 7]] + [0.0, 0.0, 1.0]
    result = visloc.evaluate_ate(
        np.vstack([estimated, [[9.0, 9.0, 9.0]]]),
        reference,
        alignment="none",
        estimated_ids=[1, 3, 5, 7, 42],
    )
    assert result.matched_count == 4
    assert result.missing_reference_count == 1
    assert result.missing_estimate_count == 6
    np.testing.assert_allclose(result.errors, 1.0)
    np.testing.assert_array_equal(result.frame_ids, [1, 3, 5, 7])

    with pytest.raises(ValueError, match="duplicate"):
        visloc.evaluate_ate(reference, reference, estimated_ids=[0] * 10)
    with pytest.raises(ValueError, match="shares a frame id"):
        visloc.evaluate_ate(reference, reference, estimated_ids=np.arange(100, 110))


def test_ate_pose_input_formats_agree(rng):
    centers = circle_trajectory(20)
    rotations = [rotation_from_axis_angle(rng.normal(size=3), a) for a in rng.uniform(0, 1, 20)]
    matrices = np.tile(np.eye(4), (20, 1, 1))
    tum = np.zeros((20, 7))
    for i, r in enumerate(rotations):
        matrices[i, :3, :3] = r
        matrices[i, :3, 3] = centers[i]
        tum[i, :3] = centers[i]
        tum[i, 3:] = visloc.Pose(r).quaternion(scalar_first=False)
    reference = centers + 0.1
    expected = visloc.evaluate_ate(centers, reference).rmse
    assert visloc.evaluate_ate(matrices, reference).rmse == pytest.approx(expected)
    assert visloc.evaluate_ate(matrices[:, :3, :], reference).rmse == pytest.approx(expected)
    assert visloc.evaluate_ate(tum, reference).rmse == pytest.approx(expected)

    with pytest.raises(ValueError, match="must have shape"):
        visloc.evaluate_ate(np.zeros((5, 5)), reference)


def test_rpe_measures_relative_drift():
    n = 30
    reference = np.tile(np.eye(4), (n, 1, 1))
    reference[:, 0, 3] = np.arange(n, dtype=float)
    estimated = reference.copy()
    estimated[:, 0, 3] *= 1.1  # 10 cm of extra forward motion per step.

    result = visloc.evaluate_rpe(estimated, reference)
    assert result.delta == 1
    assert result.pair_count == n - 1
    assert result.translation["rmse"] == pytest.approx(0.1)
    assert result.rotation_deg["max"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(result.translation_errors, 0.1)
    np.testing.assert_array_equal(result.first_frame_ids, np.arange(n - 1))

    wide = visloc.evaluate_rpe(estimated, reference, delta=5, start_step=5)
    assert wide.translation["mean"] == pytest.approx(0.5)
    assert wide.pair_count == 5

    with pytest.raises(ValueError, match="pairs"):
        visloc.evaluate_rpe(estimated[:1], reference[:1])
