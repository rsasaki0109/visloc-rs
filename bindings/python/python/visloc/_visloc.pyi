"""Type stubs for the ``visloc._visloc`` extension module."""

from os import PathLike
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

__version__: str

_ArrayLike = npt.ArrayLike
_F64 = npt.NDArray[np.float64]
_F32 = npt.NDArray[np.float32]
_I64 = npt.NDArray[np.int64]
_U64 = npt.NDArray[np.uint64]
_Path = Union[str, PathLike[str]]
_Alignment = Literal["none", "first", "first_translation", "se3", "rigid", "sim3", "similarity"]

class Camera:
    """A camera with a COLMAP model, image size, and parameter vector.

    ``model`` is a COLMAP model name (``SIMPLE_PINHOLE``, ``PINHOLE``,
    ``SIMPLE_RADIAL``, ``RADIAL``, ``OPENCV``, ``FULL_OPENCV``,
    ``OPENCV_FISHEYE``, ``SIMPLE_RADIAL_FISHEYE``, ``RADIAL_FISHEYE``, ``FOV``)
    or ``DOUBLE_SPHERE``; ``params`` uses that model's COLMAP layout.
    """

    def __init__(
        self,
        model: str,
        width: int,
        height: int,
        params: _ArrayLike,
        id: int = 1,
    ) -> None: ...
    @staticmethod
    def pinhole(
        fx: float, fy: float, cx: float, cy: float, width: int, height: int, id: int = 1
    ) -> Camera: ...
    @property
    def id(self) -> int: ...
    @property
    def model(self) -> str: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def params(self) -> _F64: ...
    @property
    def intrinsics(self) -> Optional[tuple[float, float, float, float]]:
        """``(fx, fy, cx, cy)``, or ``None`` for an unknown model."""
    def calibration_matrix(self) -> _F64:
        """3x3 calibration matrix ``K``."""
    def project(self, points: _ArrayLike, pose: Optional[Pose] = None) -> _F64:
        """Project ``(3,)`` / ``(N, 3)`` points to ``(2,)`` / ``(N, 2)`` pixels.

        Points are in the camera frame, or the world frame when ``pose``
        (world-to-camera) is given. Unprojectable points yield ``NaN`` rows.
        """
    def unproject(self, pixels: _ArrayLike) -> _F64:
        """Back-project ``(2,)`` / ``(N, 2)`` pixels to unit camera-frame rays."""
    def normalize(self, pixels: _ArrayLike) -> _F64:
        """Undistort pixels to normalized image coordinates ``(x/z, y/z)``."""
    def __eq__(self, other: object) -> bool: ...

class Pose:
    """A rigid SE(3) transform ``T * p = R p + t``.

    As a camera pose it is world-to-camera (COLMAP convention).
    """

    def __init__(
        self, rotation: Optional[_ArrayLike] = None, translation: Optional[_ArrayLike] = None
    ) -> None: ...
    @staticmethod
    def identity() -> Pose: ...
    @staticmethod
    def from_quaternion(
        quaternion: _ArrayLike,
        translation: Optional[_ArrayLike] = None,
        *,
        scalar_first: bool = True,
    ) -> Pose:
        """Quaternion ``(w, x, y, z)``, or ``(x, y, z, w)`` with ``scalar_first=False``."""
    @staticmethod
    def from_matrix(matrix: _ArrayLike) -> Pose:
        """From a ``(4, 4)`` or ``(3, 4)`` homogeneous matrix."""
    @staticmethod
    def exp(tangent: _ArrayLike) -> Pose:
        """SE(3) exponential of ``[rho; omega]`` (translation first)."""
    def log(self) -> _F64:
        """SE(3) logarithm ``[rho; omega]`` (translation first)."""
    @property
    def rotation(self) -> _F64: ...
    @property
    def translation(self) -> _F64: ...
    def quaternion(self, *, scalar_first: bool = True) -> _F64: ...
    def matrix(self) -> _F64: ...
    def inverse(self) -> Pose: ...
    def compose(self, other: Pose) -> Pose:
        """``self * other`` (apply ``other`` first)."""
    def __matmul__(self, other: Pose) -> Pose: ...
    def transform(self, points: _ArrayLike) -> _F64:
        """Apply to ``(3,)`` / ``(N, 3)`` points."""
    def camera_center(self) -> _F64:
        """World-frame camera center ``-R^T t`` of a world-to-camera pose."""

class Image:
    """A registered image: camera id, world-to-camera pose, and keypoints with
    observed 3-D point ids (``-1`` when untriangulated)."""

    def __init__(
        self,
        id: int,
        camera_id: int,
        pose: Pose,
        keypoints: Optional[_ArrayLike] = None,
        point3d_ids: Optional[_ArrayLike] = None,
    ) -> None: ...
    @property
    def id(self) -> int: ...
    @property
    def camera_id(self) -> int: ...
    @property
    def pose(self) -> Optional[Pose]: ...
    @property
    def keypoints(self) -> _F64: ...
    @property
    def point3d_ids(self) -> _I64: ...
    @property
    def num_observations(self) -> int: ...

class Reconstruction:
    """A sparse COLMAP model (cameras, images, 3-D points)."""

    def __init__(self) -> None: ...
    @staticmethod
    def read_text(path: _Path) -> Reconstruction: ...
    @staticmethod
    def read_binary(path: _Path) -> Reconstruction: ...
    @staticmethod
    def read(path: _Path) -> Reconstruction:
        """Read a model directory, preferring ``*.bin`` over ``*.txt``."""
    def write_text(self, path: _Path) -> None: ...
    def write_binary(self, path: _Path) -> None: ...
    @property
    def cameras(self) -> dict[int, Camera]: ...
    @property
    def images(self) -> dict[int, Image]: ...
    @property
    def point3d_ids(self) -> _U64: ...
    @property
    def points3d(self) -> _F64: ...
    @property
    def point3d_descriptors(self) -> Optional[_F32]: ...
    @property
    def num_cameras(self) -> int: ...
    @property
    def num_images(self) -> int: ...
    @property
    def num_points3d(self) -> int: ...
    def add_camera(self, camera: Camera) -> None: ...
    def add_image(self, image: Image) -> None: ...
    def set_points3d(
        self, ids: _ArrayLike, points: _ArrayLike, descriptors: Optional[_ArrayLike] = None
    ) -> None: ...
    def load_point3d_descriptors(self, path: _Path) -> int: ...
    def validate(self) -> list[str]: ...

class LocalizationResult:
    @property
    def success(self) -> bool: ...
    @property
    def pose(self) -> Optional[Pose]: ...
    @property
    def failure_reason(self) -> Optional[str]: ...
    @property
    def candidate_landmark_count(self) -> int: ...
    @property
    def match_count(self) -> int: ...
    @property
    def correspondence_count(self) -> int: ...
    @property
    def inlier_count(self) -> int: ...
    @property
    def outlier_count(self) -> int: ...
    @property
    def inlier_ratio(self) -> float: ...
    @property
    def mean_reprojection_error(self) -> Optional[float]: ...
    @property
    def median_reprojection_error(self) -> Optional[float]: ...
    @property
    def max_reprojection_error(self) -> Optional[float]: ...
    @property
    def inlier_query_indices(self) -> _I64: ...
    @property
    def inlier_point3d_ids(self) -> _U64: ...
    @property
    def inlier_reprojection_errors(self) -> _F64: ...
    def __bool__(self) -> bool: ...

class PnPRansacResult:
    @property
    def pose(self) -> Pose: ...
    @property
    def inliers(self) -> _I64: ...
    @property
    def inlier_count(self) -> int: ...
    @property
    def inlier_reprojection_errors(self) -> _F64: ...
    @property
    def mean_reprojection_error(self) -> float: ...
    @property
    def median_reprojection_error(self) -> float: ...
    @property
    def max_reprojection_error(self) -> float: ...
    @property
    def refinement_applied(self) -> bool: ...

def localize(
    camera: Camera,
    keypoints: _ArrayLike,
    descriptors: _ArrayLike,
    reconstruction: Reconstruction,
    *,
    ratio: Optional[float] = 0.8,
    ransac_iterations: int = 128,
    reprojection_threshold: float = 4.0,
    min_inliers: int = 0,
    min_inlier_ratio: float = 0.0,
    max_mean_reprojection_error: Optional[float] = None,
    max_median_reprojection_error: Optional[float] = None,
    max_reprojection_error: Optional[float] = None,
    seed: int = 7,
) -> LocalizationResult:
    """Localize ``(N, 2)`` keypoints with ``(N, D)`` descriptors against a
    reconstruction whose 3-D points carry descriptors (PnP + RANSAC)."""

def estimate_pose_pnp_ransac(
    camera: Camera,
    points2d: _ArrayLike,
    points3d: _ArrayLike,
    *,
    ransac_iterations: int = 128,
    reprojection_threshold: float = 4.0,
    seed: int = 7,
    refine: bool = True,
    confidence: Optional[float] = None,
) -> Optional[PnPRansacResult]:
    """World-to-camera pose from ``(N, 2)`` pixels and ``(N, 3)`` world points."""

class SimilarityTransform:
    @property
    def scale(self) -> float: ...
    @property
    def rotation(self) -> _F64: ...
    @property
    def translation(self) -> _F64: ...
    def matrix(self) -> _F64: ...
    def apply(self, points: _ArrayLike) -> _F64: ...

class AteResult:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    matched_count: int
    estimated_count: int
    reference_count: int
    missing_reference_count: int
    missing_estimate_count: int
    alignment: SimilarityTransform
    @property
    def errors(self) -> _F64: ...
    @property
    def frame_ids(self) -> _U64: ...

class RpeResult:
    delta: int
    pair_count: int
    matched_count: int
    translation: dict[str, float]
    rotation_deg: dict[str, float]
    @property
    def translation_errors(self) -> _F64: ...
    @property
    def rotation_errors_deg(self) -> _F64: ...
    @property
    def first_frame_ids(self) -> _U64: ...

def umeyama_alignment(
    source: _ArrayLike, target: _ArrayLike, with_scale: bool = False
) -> SimilarityTransform: ...
def evaluate_ate(
    estimated: _ArrayLike,
    reference: _ArrayLike,
    *,
    alignment: _Alignment = "se3",
    estimated_ids: Optional[_ArrayLike] = None,
    reference_ids: Optional[_ArrayLike] = None,
) -> AteResult:
    """ATE between ``(N, 3)`` centers, ``(N, 7)`` ``[tx ty tz qx qy qz qw]``, or
    ``(N, 4, 4)`` / ``(N, 3, 4)`` camera-to-world trajectories."""

def evaluate_rpe(
    estimated: _ArrayLike,
    reference: _ArrayLike,
    *,
    delta: int = 1,
    start_step: int = 1,
    estimated_ids: Optional[_ArrayLike] = None,
    reference_ids: Optional[_ArrayLike] = None,
) -> RpeResult: ...
