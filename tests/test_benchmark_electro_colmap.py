"""Tests for the exact-candidate official COLMAP control preparation."""

from __future__ import annotations

import importlib.util
import json
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
colmap = load_module("benchmark_electro_colmap", ROOT / "scripts" / "benchmark_electro_colmap.py")
score = load_module("score_electro_model", ROOT / "scripts" / "score_electro_model.py")


class ElectroColmapControlTests(unittest.TestCase):
    def _calibration(self, root: Path) -> Path:
        calibration = root / "calibration"
        calibration.mkdir()
        (calibration / "cameras.txt").write_text(
            "# cameras\n"
            "1 PINHOLE 10 8 5 5 5 4\n"
            "2 PINHOLE 12 9 6 6 6 4.5\n",
            encoding="utf-8",
        )
        (calibration / "images.txt").write_text(
            "# image assignments\n"
            "1 1 0 0 0 0 0 0 1 cam4_100.png\n"
            "# points omitted\n"
            "2 1 0 0 0 0 0 0 2 cam5_100.png\n"
            "# points omitted\n"
            "3 1 0 0 0 0 0 0 1 cam4_101.png\n",
            encoding="utf-8",
        )
        return calibration

    def _candidate(self, root: Path) -> Path:
        path = root / "candidates.txt"
        benchmark.write_candidate_manifest(
            path,
            ["cam4_100.png", "cam5_100.png", "cam4_101.png"],
            [(0, 1), (0, 2)],
            metadata={"candidate_policy": "vlad-union-v1"},
        )
        return path

    def test_pair_list_preserves_order_and_maps_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            pair_list = root / "pairs.txt"
            info = colmap.write_pair_list(
                candidate,
                pair_list,
                aliases={
                    "cam4_100.png": "cam4/100.png",
                    "cam5_100.png": "cam5/100.png",
                    "cam4_101.png": "cam4/101.png",
                },
            )
            self.assertEqual(
                pair_list.read_text(encoding="utf-8").splitlines(),
                ["cam4/100.png cam5/100.png", "cam4/100.png cam4/101.png"],
            )
            self.assertEqual(info["pair_count"], 2)
            self.assertEqual(info["candidate_image_names"][0], "cam4_100.png")

    def test_prepare_binds_validated_index_and_uses_exact_pair_importer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            candidate_index = root / "candidate_shards"
            benchmark.split_candidate_manifest(candidate, candidate_index, 1)
            calibration = self._calibration(root)
            camera_root = root / "camera_shards"
            (camera_root / "cam4" / "images").mkdir(parents=True)
            (camera_root / "cam5" / "images").mkdir(parents=True)
            image_root = root / "images"
            image_root.mkdir()
            output = root / "control"
            plan = colmap.make_plan(
                candidate_index=candidate_index / "index.json",
                candidate_manifest=None,
                output_root=output,
                image_root=image_root,
                camera_root=camera_root,
                calibration_dir=calibration,
                colmap_binary=Path("/bin/true"),
            )
            self.assertEqual(plan["candidate"]["pair_count"], 2)
            self.assertEqual(plan["colmap"]["matching_settings"]["mode"], "matches_importer")
            self.assertEqual(plan["colmap"]["matching_settings"]["match_type"], "pairs")
            self.assertEqual(len(plan["colmap"]["commands"]["feature_extractor"]), 2)
            match = plan["colmap"]["commands"]["matches_importer"]
            self.assertIn("matches_importer", match)
            self.assertIn("--match_type", match)
            self.assertIn("pairs", match)
            for command in [*plan["colmap"]["commands"]["feature_extractor"], match, plan["colmap"]["commands"]["mapper"]]:
                self.assertNotIn("points3D.txt", command)
                self.assertNotIn("--gt", command)
            self.assertEqual(json.loads((output / "plan.json").read_text())["schema"], colmap.PLAN_SCHEMA)

    def test_score_joins_flat_and_official_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference_images.txt"
            query = root / "query_images.txt"
            names = [
                "1474975187520882738.png",
                "1474975187594610738.png",
                "1474975187668338738.png",
                "1474975187742066738.png",
            ]
            reference_centres = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
            query_centres = [(2 * x + 3, 2 * y - 4, 2 * z + 5) for x, y, z in reference_centres]

            def write_model(path: Path, prefix: str, centres: list[tuple[float, float, float]]) -> None:
                rows: list[str] = []
                for image_id, (name, centre) in enumerate(zip(names, centres), 1):
                    rows.append(
                        f"{image_id} 1 0 0 0 {-centre[0]} {-centre[1]} {-centre[2]} 1 {prefix}{name}"
                    )
                    rows.append("0 0 -1")
                path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            write_model(reference, "images_rig_cam4_undistorted/", reference_centres)
            write_model(query, "cam4_", query_centres)
            result = score.score(reference, query)
            self.assertEqual(result["common"], 4)
            self.assertEqual(result["query_registered"], 4)
            self.assertAlmostEqual(result["sim3_scale"], 0.5, places=6)
            self.assertLess(result["rmse_m"], 1e-9)

    def test_score_rejects_non_electro_name(self) -> None:
        with self.assertRaises(score.ScoreError):
            score.image_key("frame_0001.png")

    def test_score_rejects_identity_staging_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "identity_reference_images.txt"
            query = root / "query_images.txt"
            names = [
                "1474975187520882738.png",
                "1474975187594610738.png",
                "1474975187668338738.png",
            ]

            def write_model(path: Path, centres: list[tuple[float, float, float]]) -> None:
                rows: list[str] = []
                for image_id, (name, centre) in enumerate(zip(names, centres), 1):
                    rows.append(
                        f"{image_id} 1 0 0 0 {-centre[0]} {-centre[1]} {-centre[2]} 1 images_rig_cam4_undistorted/{name}"
                    )
                    rows.append("0 0 -1")
                path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            write_model(reference, [(0.0, 0.0, 0.0)] * len(names))
            write_model(query, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
            with self.assertRaisesRegex(score.ScoreError, "identity staging calibration"):
                score.score(reference, query)

    def test_run_rejects_tampered_pair_list_before_starting_colmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            candidate_index = root / "candidate_shards"
            benchmark.split_candidate_manifest(candidate, candidate_index, 1)
            calibration = self._calibration(root)
            camera_root = root / "camera_shards"
            (camera_root / "cam4" / "images").mkdir(parents=True)
            (camera_root / "cam5" / "images").mkdir(parents=True)
            output = root / "control"
            colmap.make_plan(
                candidate_index=candidate_index / "index.json",
                candidate_manifest=None,
                output_root=output,
                image_root=root / "images",
                camera_root=camera_root,
                calibration_dir=calibration,
                colmap_binary=Path("/bin/true"),
            )
            pair_list = output / "candidate_pairs.txt"
            pair_list.write_text(pair_list.read_text(encoding="utf-8") + "cam4_101.png cam5_100.png\n", encoding="utf-8")
            with self.assertRaisesRegex(colmap.ValidationError, "pair-list hash mismatch"):
                colmap.run_plan(output / "plan.json")

    def test_run_plan_creates_mapper_output_directory_and_records_cam_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root)
            candidate_index = root / "candidate_shards"
            benchmark.split_candidate_manifest(candidate, candidate_index, 1)
            calibration = self._calibration(root)
            camera_root = root / "camera_shards"
            (camera_root / "cam4" / "images").mkdir(parents=True)
            (camera_root / "cam5" / "images").mkdir(parents=True)
            output = root / "control"
            colmap.make_plan(
                candidate_index=candidate_index / "index.json",
                candidate_manifest=None,
                output_root=output,
                image_root=root / "images",
                camera_root=camera_root,
                calibration_dir=calibration,
                colmap_binary=Path("/bin/true"),
            )
            result = colmap.run_plan(output / "plan.json")
            self.assertTrue((output / "models").is_dir())
            self.assertEqual(
                sorted(result["phases"]),
                ["feature_extractor_cam4", "feature_extractor_cam5", "mapper", "matches_importer"],
            )


if __name__ == "__main__":
    unittest.main()
