"""Python bindings for visloc-rs.

Pure-Rust visual localization primitives exposed to Python with NumPy interop:

* :class:`Camera` - COLMAP camera models with ``project`` / ``unproject``.
* :class:`Pose` - rigid SE(3) transforms (world-to-camera for camera poses).
* :class:`Reconstruction` / :class:`Image` - COLMAP text and binary models.
* :func:`localize` / :func:`estimate_pose_pnp_ransac` - PnP + RANSAC localization.
* :func:`evaluate_ate` / :func:`evaluate_rpe` / :func:`umeyama_alignment` -
  trajectory evaluation.
"""

from ._visloc import (
    AteResult,
    Camera,
    Image,
    LocalizationResult,
    PnPRansacResult,
    Pose,
    Reconstruction,
    RpeResult,
    SimilarityTransform,
    __version__,
    estimate_pose_pnp_ransac,
    evaluate_ate,
    evaluate_rpe,
    localize,
    umeyama_alignment,
)

#: Alias: :class:`Pose` is a general SE(3) transform.
SE3 = Pose

__all__ = [
    "AteResult",
    "Camera",
    "Image",
    "LocalizationResult",
    "PnPRansacResult",
    "Pose",
    "Reconstruction",
    "RpeResult",
    "SE3",
    "SimilarityTransform",
    "__version__",
    "estimate_pose_pnp_ransac",
    "evaluate_ate",
    "evaluate_rpe",
    "localize",
    "umeyama_alignment",
]
