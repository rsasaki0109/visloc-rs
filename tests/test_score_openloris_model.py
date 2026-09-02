"""Tests for official-GT OpenLORIS model scoring."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "score_openloris_model", ROOT / "scripts" / "score_openloris_model.py"
)
assert SPEC is not None and SPEC.loader is not None
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


class ScoreOpenLorisModelTests(unittest.TestCase):
    def test_slerp_midpoint_rotates_halfway(self) -> None:
        result = score._slerp_xyzw(
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            np.asarray([0.0, 0.0, 1.0, 0.0]),
            0.5,
        )
        self.assertTrue(np.allclose(np.abs(result), [0.0, 0.0, 2**-0.5, 2**-0.5]))

    def test_manifest_and_camera_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "visloc_openloris_corridor_manifest_v1",
                        "scene": "corridor1-1",
                        "images": [
                            {"name": "cam1_000000.png", "camera": 1, "timestamp": "0.5"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = score.load_manifest(path)
            identity = np.eye(4)
            identity[0, 3] = 1.0
            reference = score.interpolate_camera_centres(
                manifest,
                (
                    np.asarray([0.0, 1.0]),
                    np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                    np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
                ),
                {1: identity},
                max_gap_seconds=2.0,
            )
            self.assertTrue(np.allclose(reference["cam1_000000.png"], [2.0, 0.0, 0.0]))

    def test_umeyama_removes_similarity(self) -> None:
        source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        destination = 3.0 * source + np.asarray([4.0, 5.0, 6.0])
        scale_value, rotation, translation = score.umeyama(source, destination)
        aligned = scale_value * (rotation @ source.T).T + translation
        self.assertTrue(np.allclose(aligned, destination))

    def test_component_score_recovers_zero_error_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            query = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            names = [f"cam1_{index:06d}.png" for index in range(len(query))]
            rows = []
            for image_id, (name, centre) in enumerate(zip(names, query), 1):
                translation = -centre
                rows.extend(
                    [
                        f"{image_id} 1 0 0 0 {translation[0]} {translation[1]} {translation[2]} 1 {name}",
                        "",
                    ]
                )
            path.write_text("\n".join(rows), encoding="utf-8")
            reference = {
                name: 2.5 * centre + np.asarray([3.0, -4.0, 5.0])
                for name, centre in zip(names, query)
            }
            result, errors, scored_names = score.score_component(path, reference)
            self.assertEqual(result["registered"], 4)
            self.assertLess(result["rmse_m"], 1e-12)
            self.assertTrue(np.all(errors < 1e-12))
            self.assertEqual(scored_names, names)

    def test_load_camera_extrinsics_composes_second_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trans_matrix.yaml"
            path.write_text(
                "%YAML:1.0\ntrans_matrix:\n"
                "  - parent_frame: base_link\n"
                "    child_frame: t265_fisheye1_optical_frame\n"
                "    matrix: !!opencv-matrix\n"
                "      rows: 4\n      cols: 4\n      dt: d\n"
                "      data: [1,0,0,1, 0,1,0,2, 0,0,1,3, 0,0,0,1]\n"
                "  - parent_frame: t265_fisheye1_optical_frame\n"
                "    child_frame: t265_fisheye2_optical_frame\n"
                "    matrix: !!opencv-matrix\n"
                "      rows: 4\n      cols: 4\n      dt: d\n"
                "      data: [1,0,0,4, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n",
                encoding="utf-8",
            )
            transforms = score.load_camera_extrinsics(path)
            self.assertTrue(np.allclose(transforms[1][:3, 3], [1.0, 2.0, 3.0]))
            self.assertTrue(np.allclose(transforms[2][:3, 3], [5.0, 2.0, 3.0]))

    def test_alias_map_is_reversed_for_colmap_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.tsv"
            path.write_text(
                "flat_name\tcolmap_name\n"
                "cam1_000000.png\trig/camera1/1.0.png\n",
                encoding="utf-8",
            )
            self.assertEqual(
                score.load_colmap_aliases(path),
                {"rig/camera1/1.0.png": "cam1_000000.png"},
            )

    def test_temporal_segments_are_sorted_and_cover_every_error(self) -> None:
        manifest = {
            "late": (1, 3.0),
            "early": (1, 1.0),
            "middle": (2, 2.0),
        }
        segments = score.temporal_error_segments(
            [("late", 3.0), ("early", 1.0), ("middle", 2.0)], manifest, bins=2
        )
        self.assertEqual([row["images"] for row in segments], [2, 1])
        self.assertEqual(segments[0]["timestamp_start"], 1.0)
        self.assertEqual(segments[1]["timestamp_end"], 3.0)
        self.assertAlmostEqual(segments[0]["rmse_m"], math.sqrt(2.5))


if __name__ == "__main__":
    unittest.main()
