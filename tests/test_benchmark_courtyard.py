"""Focused, dependency-free tests for the courtyard benchmark runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_courtyard", ROOT / "scripts" / "benchmark_courtyard.py"
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CourtyardBenchmarkTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_and_traversal_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.tsv"
            duplicate.write_text(
                "a_features.txt 1 " + "0" * 64 + "\n"
                "a_features.txt 1 " + "0" * 64 + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(benchmark.ValidationError, "repeats"):
                benchmark.parse_feature_manifest(duplicate)

            traversal = root / "traversal.tsv"
            traversal.write_text("../outside.txt 1 " + "0" * 64 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "simple relative path"):
                benchmark.parse_feature_manifest(traversal)

    def test_feature_hash_and_row_mismatch_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            feature = features / "a_features.txt"
            feature.write_text("0 0\n1 1\n", encoding="utf-8")
            manifest = features / "MANIFEST.tsv"
            manifest.write_text(f"a_features.txt 2 {digest(feature)}\n", encoding="utf-8")
            names, stats = benchmark.validate_features(
                root,
                {
                    "manifest": {
                        "path": "features/MANIFEST.tsv",
                        "sha256": digest(manifest),
                    },
                    "file_count": 1,
                    "total_rows": 2,
                },
            )
            self.assertEqual(names, ["a"])
            self.assertEqual(stats["total_rows"], 2)

            feature.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_features(
                    root,
                    {
                        "manifest": {
                            "path": "features/MANIFEST.tsv",
                            "sha256": digest(manifest),
                        },
                        "file_count": 1,
                        "total_rows": 2,
                    },
                )

    def test_raw_match_parser_checks_indices_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches.txt"
            matches.write_text("2\na.JPG\nb.JPG\n1\n0 1 1\n0 1\n", encoding="utf-8")
            spec = {
                "sha256": digest(matches),
                "pair_count": 1,
                "raw_match_count": 1,
                "candidate_semantics": "all unordered pairs",
            }
            result = benchmark.validate_matches(matches, spec, ["a", "b"], ".JPG", {0: 2, 1: 2})
            self.assertEqual(result["pair_count"], 1)
            self.assertEqual(result["raw_match_count"], 1)

            matches.write_text("2\na.JPG\nb.JPG\n1\n0 1 1\n2 1\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "outside"):
                benchmark.validate_matches(
                    matches,
                    {**spec, "sha256": digest(matches)},
                    ["a", "b"],
                    ".JPG",
                    {0: 2, 1: 2},
                )

    def test_candidate_manifest_round_trip_and_structural_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "candidates.txt"
            manifest.write_text(
                "visloc_candidate_manifest_v1\n"
                "images 3\n"
                "image 0 a.JPG\n"
                "image 1 b.JPG\n"
                "image 2 c.JPG\n"
                "pairs 2\n"
                "pair 0 2\n"
                "pair 1 2\n",
                encoding="utf-8",
            )
            parsed = benchmark.parse_candidate_manifest(manifest, ["a.JPG", "b.JPG", "c.JPG"])
            self.assertEqual(parsed["pair_count"], 2)
            self.assertEqual(parsed["pairs"], [(0, 2), (1, 2)])
            self.assertEqual(benchmark.validate_candidate_manifest(manifest, ["a.JPG", "b.JPG", "c.JPG"], 2)["sha256"], digest(manifest))
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_candidate_manifest(
                    manifest,
                    ["a.JPG", "b.JPG", "c.JPG"],
                    expected_sha256="0" * 64,
                )

            duplicate = root / "duplicate.txt"
            duplicate.write_text(manifest.read_text(encoding="utf-8").replace("pair 1 2", "pair 0 2"), encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "repeats"):
                benchmark.parse_candidate_manifest(duplicate, ["a.JPG", "b.JPG", "c.JPG"])

            reversed_pair = root / "reversed.txt"
            reversed_pair.write_text(manifest.read_text(encoding="utf-8").replace("pair 0 2", "pair 2 0"), encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "0 <= I < J"):
                benchmark.parse_candidate_manifest(reversed_pair, ["a.JPG", "b.JPG", "c.JPG"])

    def test_candidate_schedule_validation_and_mapping_command(self) -> None:
        config = benchmark.load_config(ROOT / "benchmarks" / "courtyard" / "exhaustive_control.json")
        schedule = benchmark.resolve_candidate_schedule(config, "local-vlad-union-3-8-200")
        command = benchmark.build_mapping_command(
            config,
            binary=Path("/bin/visloc-demo"),
            features_dir=Path("/external/features"),
            images_dir=Path("/external/images"),
            calibration_dir=Path("/external/calibration"),
            matches_path=Path("/external/matches_import.txt"),
            output_model=Path("/external/out"),
            candidate_schedule=schedule,
        )
        self.assertIn("--pair-source", command)
        self.assertIn("vlad-union", command)
        self.assertIn("--local-stem-window", command)
        self.assertIn("--candidate-budget", command)
        self.assertNotIn("--exhaustive", command)
        replay = benchmark.build_mapping_command(
            config,
            binary=Path("/bin/visloc-demo"),
            features_dir=Path("/external/features"),
            images_dir=Path("/external/images"),
            calibration_dir=Path("/external/calibration"),
            matches_path=Path("/external/matches_import.txt"),
            output_model=Path("/external/out"),
            candidate_manifest=Path("/external/candidates.txt"),
        )
        self.assertIn("--candidate-manifest", replay)
        self.assertNotIn("--exhaustive", replay)
        with self.assertRaisesRegex(benchmark.ValidationError, "unsupported strategy"):
            benchmark.resolve_candidate_schedule(config, "not-a-schedule")
        bad_config = {**config, "candidate_schedules": {"bad": {"strategy": "vlad"}}}
        with self.assertRaisesRegex(benchmark.ValidationError, "retrieval_topk"):
            benchmark.resolve_candidate_schedule(bad_config, "bad")

    def test_mapping_log_parser_extracts_candidate_and_result_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapper.log"
            path.write_text(
                "view graph: 200 candidate pairs (local stem + VLAD union)\n"
                "verified 172 / 200 pairs, 199871 inlier correspondences\n"
                "reconstruction (track-source=union-find): 38 / 38 images registered, 45016 tracks, mean reproj 0.537 px\n"
                "wrote COLMAP model to /tmp/model (38 images, 45016 points, 148192 observations)\n"
                "elapsed=57.35 maxrss=1068548\n",
                encoding="utf-8",
            )
            parsed = benchmark.parse_mapping_log(path)
            self.assertEqual(parsed["candidate_pairs"], 200)
            self.assertEqual(parsed["verified_pairs"], 172)
            self.assertEqual(parsed["inlier_correspondences"], 199871)
            self.assertEqual(parsed["registered"], 38)
            self.assertEqual(parsed["tracks"], 45016)
            self.assertAlmostEqual(parsed["reprojection_px"], 0.537)
            self.assertEqual(parsed["written_model"]["observations"], 148192)
            self.assertAlmostEqual(parsed["process_elapsed_s"], 57.35)
            self.assertEqual(parsed["peak_rss_kb"], 1068548)

    def test_score_threshold_and_hash_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score = root / "score.txt"
            score.write_text(
                "matched=38/38 est_registered=38\n"
                "sim3_scale=1.0\n"
                "rmse_m=0.020000 rmse_cm=2.00\n",
                encoding="utf-8",
            )
            model_spec = {
                "registered": 38,
                "score_file": {"path": "score.txt", "sha256": digest(score)},
                "score_rmse_m": 0.020000,
            }
            with self.assertRaisesRegex(benchmark.ValidationError, "exceeds threshold"):
                benchmark.validate_score_file(root, {}, model_spec, 0.01)

            score.write_text(score.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "hash mismatch"):
                benchmark.validate_score_file(root, {}, model_spec, 0.1)

    def test_visual_asset_parser_and_config_schema(self) -> None:
        width, height = benchmark._png_dimensions(ROOT / "docs" / "assets" / "courtyard_sfm_comparison.png")
        self.assertEqual((width, height), (1458, 883))
        width, height, frames = benchmark._gif_dimensions_and_frames(ROOT / "docs" / "assets" / "courtyard_sfm_comparison.gif")
        self.assertEqual((width, height, frames), (772, 432, 24))
        config = benchmark.load_config(ROOT / "benchmarks" / "courtyard" / "exhaustive_control.json")
        self.assertEqual(config["benchmark"]["id"], "courtyard-colmap-exhaustive-v1")
        self.assertEqual(config["expected"]["max_rmse_m"], 0.01)

    def test_mapping_command_contains_only_mapping_inputs(self) -> None:
        config = benchmark.load_config(ROOT / "benchmarks" / "courtyard" / "exhaustive_control.json")
        command = benchmark.build_mapping_command(
            config,
            binary=Path("/bin/visloc-demo"),
            features_dir=Path("/external/features"),
            images_dir=Path("/external/images"),
            calibration_dir=Path("/external/calibration"),
            matches_path=Path("/external/matches_import.txt"),
            output_model=Path("/external/out"),
        )
        self.assertIn("--import-matches-file", command)
        self.assertIn("--input-colmap-calibration", command)
        self.assertIn("--next-image-policy", command)
        self.assertNotIn("--gt", command)
        self.assertNotIn("points3D.txt", command)

    def test_config_rejects_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
            with self.assertRaisesRegex(benchmark.ValidationError, "unsupported"):
                benchmark.load_config(path)


if __name__ == "__main__":
    unittest.main()
