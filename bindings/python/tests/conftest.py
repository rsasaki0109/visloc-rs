from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DATA = REPO_ROOT / "examples" / "data"
IO_FIXTURE = REPO_ROOT / "crates" / "io" / "tests" / "fixtures" / "colmap_text"


def rotation_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def example_data():
    if not (EXAMPLE_DATA / "colmap_text" / "cameras.txt").is_file():
        pytest.skip("repository example data not available")
    return EXAMPLE_DATA


@pytest.fixture
def io_fixture():
    if not (IO_FIXTURE / "cameras.txt").is_file():
        pytest.skip("repository COLMAP fixture not available")
    return IO_FIXTURE
