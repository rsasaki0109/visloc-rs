"""Tests for the Docker-backed OpenLORIS COLMAP control."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_module("benchmark_electro", ROOT / "scripts" / "benchmark_electro.py")
control = load_module(
    "benchmark_openloris_colmap", ROOT / "scripts" / "benchmark_openloris_colmap.py"
)


class OpenLorisColmapControlTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        names = ["cam1_000000.png", "cam2_000001.png", "cam1_000002.png"]
        image_root = root / "images"
        image_root.mkdir()
        for index, name in enumerate(names):
            (image_root / name).write_bytes(b"not-a-real-png-" + bytes([index]))
        calibration = root / "calibration"
        calibration.mkdir()
        (calibration / "cameras.txt").write_text(
            "1 PINHOLE 848 800 285 286 425 398\n"
            "2 PINHOLE 848 800 284 285 427 397\n",
            encoding="utf-8",
        )
        (calibration / "images.txt").write_text(
            "1 1 0 0 0 0 0 0 1 cam1_000000.png\n\n"
            "2 1 0 0 0 0 0 0 2 cam2_000001.png\n\n"
            "3 1 0 0 0 0 0 0 1 cam1_000002.png\n\n",
            encoding="utf-8",
        )
        candidate = root / "candidates.txt"
        benchmark.write_candidate_manifest(candidate, names, [(0, 1), (0, 2)])
        shards = root / "candidate-shards"
        benchmark.split_candidate_manifest(candidate, shards, 1)
        return image_root, calibration, shards / "index.json"

    def test_prepare_stages_hardlinks_and_binds_docker_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root, calibration, candidate_index = self._fixture(root)
            output = root / "control"
            identity = {
                "reference": "colmap/test:fixed",
                "id": "sha256:" + "a" * 64,
                "repo_digests": ["colmap/test@sha256:" + "b" * 64],
            }
            plan = control.prepare_plan(
                candidate_index=candidate_index,
                image_root=image_root,
                calibration_dir=calibration,
                output_root=output,
                docker_image="colmap/test:fixed",
                docker_identity=identity,
            )
            self.assertEqual(plan["candidate"]["pair_count"], 2)
            self.assertEqual(plan["inputs"]["staging"]["cameras"], {"1": 2, "2": 1})
            self.assertTrue(
                (output / "staged-images" / "cam1" / "cam1_000000.png").samefile(
                    image_root / "cam1_000000.png"
                )
            )
            self.assertEqual(
                plan["commands"]["matches_importer"][0:2], ["colmap", "matches_importer"]
            )
            self.assertIn("--Mapper.multiple_models", plan["commands"]["mapper"])
            params_index = plan["commands"]["feature_extractor"][0]["command"].index(
                "--ImageReader.camera_params"
            )
            self.assertEqual(
                plan["commands"]["feature_extractor"][0]["command"][params_index + 1],
                "285,286,425.5,398.5",
            )
            self.assertEqual(plan["settings"]["opencv_to_colmap_principal_point_shift_px"], 0.5)
            self.assertEqual(len(plan["software"]["runner_sha256"]), 64)
            self.assertFalse(plan["ground_truth_used_for_selection_or_mapping"])

    def test_prepare_calibrated_rig_binds_frames_aliases_and_fixed_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [
                "cam1_000000.png", "cam1_000002.png",
                "cam2_000001.png", "cam2_000003.png",
            ]
            image_root = root / "images"
            image_root.mkdir()
            for name in names:
                (image_root / name).write_bytes(name.encode())
            calibration = root / "calibration"
            calibration.mkdir()
            (calibration / "cameras.txt").write_text(
                "1 PINHOLE 848 800 285 286 425 398\n"
                "2 PINHOLE 848 800 284 285 427 397\n",
                encoding="utf-8",
            )
            (calibration / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 cam1_000000.png\n\n"
                "2 1 0 0 0 0 0 0 1 cam1_000002.png\n\n"
                "3 1 0 0 0 0 0 0 2 cam2_000001.png\n\n"
                "4 1 0 0 0 0 0 0 2 cam2_000003.png\n\n",
                encoding="utf-8",
            )
            candidate = root / "candidates.txt"
            benchmark.write_candidate_manifest(candidate, names, [(0, 2), (1, 3)])
            shards = root / "candidate-shards"
            benchmark.split_candidate_manifest(candidate, shards, 1)
            tier = root / "tier.json"
            tier.write_text(
                json.dumps(
                    {
                        "schema": "visloc_openloris_corridor_manifest_v1",
                        "images": [
                            {"name": names[0], "camera": 1, "timestamp": "1.0"},
                            {"name": names[1], "camera": 1, "timestamp": "2.0"},
                            {"name": names[2], "camera": 2, "timestamp": "1.0"},
                            {"name": names[3], "camera": 2, "timestamp": "2.0"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            transform = root / "trans_matrix.yaml"
            transform.write_text(
                "%YAML:1.0\nparent_frame: t265_fisheye1_optical_frame\n"
                "child_frame: t265_fisheye2_optical_frame\nmatrix: !!opencv-matrix\n"
                "data: [1, 0, 0, 0.064, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]\n",
                encoding="utf-8",
            )
            identity = {
                "reference": "colmap/test:fixed",
                "id": "sha256:" + "a" * 64,
                "repo_digests": [],
            }
            output = root / "control"
            plan = control.prepare_plan(
                candidate_index=shards / "index.json",
                image_root=image_root,
                calibration_dir=calibration,
                tier_manifest=tier,
                rig_transform_matrix=transform,
                output_root=output,
                docker_image="colmap/test:fixed",
                docker_identity=identity,
            )

            self.assertEqual(plan["inputs"]["staging"]["frames"], 2)
            self.assertTrue(plan["settings"]["fixed_calibrated_stereo_rig"])
            self.assertEqual(
                (output / "candidate_pairs.txt").read_text(encoding="utf-8"),
                "rig/camera1/1.0.png rig/camera2/1.0.png\n"
                "rig/camera1/2.0.png rig/camera2/2.0.png\n",
            )
            rig = json.loads((output / "rig_config.json").read_text(encoding="utf-8"))
            second = rig[0]["cameras"][1]
            self.assertEqual(second["cam_from_rig_rotation"], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(second["cam_from_rig_translation"], [-0.064, -0.0, -0.0])
            self.assertIn("--ImageReader.single_camera_per_folder", plan["commands"]["feature_extractor"][0]["command"])
            self.assertEqual(plan["commands"]["rig_configurator"][0:2], ["colmap", "rig_configurator"])
            self.assertEqual(plan["commands"]["mapper"][-2:], ["--Mapper.ba_refine_sensor_from_rig", "0"])

    def test_metrics_wrapper_records_process_and_cgroup_peaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.tsv"
            environment = os.environ.copy()
            environment["VISLOC_METRICS_OUTPUT"] = str(metrics)
            environment["VISLOC_METRICS_POLL_SECONDS"] = "0.01"
            subprocess.run(
                [
                    str(ROOT / "scripts" / "docker_process_metrics.sh"),
                    sys.executable,
                    "-c",
                    "import time; buffer = bytearray(8 * 1024 * 1024); time.sleep(0.05); assert buffer",
                ],
                env=environment,
                check=True,
            )
            result = control.parse_metrics(metrics)
            self.assertEqual(result["status"], 0)
            self.assertGreater(result["wall_ns"], 0)
            self.assertGreater(result["peak_process_hwm_kib"], 0)
            self.assertGreaterEqual(result["cgroup_peak_bytes"], 0)

    def test_parse_metrics_rejects_duplicate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.tsv"
            path.write_text(
                "schema\tvisloc_docker_process_metrics_v1\n"
                "status\t0\nstatus\t1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(control.ValidationError, "malformed"):
                control.parse_metrics(path)

    def test_log_diagnostics_preserves_warning_and_solver_failure_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapper.log"
            path.write_text(
                "I20260101 ordinary\n"
                "W20260101 Linear solver failure. Failed to compute a step\n"
                "W20260101 another warning\n"
                "E20260101 failure\n",
                encoding="utf-8",
            )

    def test_registered_rig_frames_collapse_synchronized_camera_names(self) -> None:
        names = {
            "rig/camera1/1.0.png",
            "rig/camera2/1.0.png",
            "rig/camera1/2.0.png",
        }
        self.assertEqual(control._registered_frame_keys(names, rig_aware=True), {"1.0.png", "2.0.png"})
        with self.assertRaisesRegex(control.ValidationError, "unexpected name"):
            control._registered_frame_keys({"cam1_000.png"}, rig_aware=True)
            self.assertEqual(
                control._log_diagnostics(path),
                {"warning_lines": 2, "error_lines": 1, "linear_solver_failure_lines": 1},
            )

    def test_database_stats_counts_verified_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.db"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
                CREATE TABLE keypoints(image_id INTEGER, rows INTEGER);
                CREATE TABLE descriptors(image_id INTEGER, rows INTEGER);
                CREATE TABLE matches(pair_id INTEGER, rows INTEGER);
                CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER);
                CREATE TABLE cameras(camera_id INTEGER);
                CREATE TABLE rigs(rig_id INTEGER);
                CREATE TABLE rig_sensors(rig_id INTEGER, sensor_id INTEGER);
                CREATE TABLE frames(frame_id INTEGER, rig_id INTEGER);
                CREATE TABLE frame_data(frame_id INTEGER, data_id INTEGER);
                INSERT INTO images VALUES (1, 'a'), (2, 'b'), (3, 'c');
                INSERT INTO keypoints VALUES (1, 5), (2, 7), (3, 3);
                INSERT INTO descriptors VALUES (1, 5), (2, 7), (3, 3);
                INSERT INTO matches VALUES (2147483649, 4), (4294967297, 0);
                INSERT INTO two_view_geometries VALUES (2147483649, 3);
                INSERT INTO cameras VALUES (1), (2);
                INSERT INTO rigs VALUES (1);
                INSERT INTO rig_sensors VALUES (1, 1), (1, 2);
                INSERT INTO frames VALUES (1, 1), (2, 1);
                INSERT INTO frame_data VALUES (1, 1), (1, 2), (2, 3), (2, 4);
                """
            )
            connection.commit()
            connection.close()

            stats = control._database_stats(path)
            self.assertEqual(stats["images"], 3)
            self.assertEqual(stats["keypoints"], 15)
            self.assertEqual(stats["candidate_pair_records"], 2)
            self.assertEqual(stats["candidate_pairs_with_raw_matches"], 1)
            self.assertEqual(stats["raw_correspondences"], 4)
            self.assertEqual(stats["verified_pairs"], 1)
            self.assertEqual(stats["verified_inliers"], 3)
            self.assertEqual(stats["verified_component_sizes"], [2, 1])
            self.assertEqual(
                stats["rig_configuration"],
                {
                    "cameras": 2,
                    "rigs": 1,
                    "rig_sensors": 2,
                    "frames": 2,
                    "frame_data": 4,
                    "min_images_per_frame": 2,
                    "max_images_per_frame": 2,
                },
            )

    def test_summarize_mode_is_mutually_exclusive(self) -> None:
        parser = control._parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--run", "--summarize-existing"])


if __name__ == "__main__":
    unittest.main()
