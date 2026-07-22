import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_vslam_sota_v4.py"
SPEC = importlib.util.spec_from_file_location("evaluate_vslam_sota_v4", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SEQUENCES = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]
FRAME_COUNTS = {sequence: 100 + index for index, sequence in enumerate(SEQUENCES)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    return {
        "schema_version": 1,
        "sequences": SEQUENCES,
        "full_sequence_frame_counts": FRAME_COUNTS,
        "repetitions": 3,
        "gates": {
            "mean_sequence_sim3_ate_rmse_m_max": 0.02,
            "min_tracked_fraction": 0.9,
            "max_committed_abs_log_scale": 4.0,
            "required_onnx_backend": "cuda",
            "max_ms_per_frame_total": 50.0,
            "max_sampled_peak_working_set_bytes": 8 * 1024**3,
            "max_sampled_peak_gpu_memory_bytes": 6 * 1024**3,
            "queue_bounds": {
                "inactive_edge_cap": 4096,
                "max_free_poses": 256,
                "long_loop_max_indexed_frames": 1000,
            },
        },
    }


class VslamSotaV4Tests(unittest.TestCase):
    def make_matrix(self, root: Path, *, ate: float = 0.019) -> tuple[Path, Path]:
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol()), encoding="utf-8")
        matrix = root / "matrix"
        matrix.mkdir()
        fixed_hashes = {
            "executable_sha256": "1" * 64,
            "model_bundle_sha256": "2" * 64,
            "configuration_sha256": "3" * 64,
            "ort_dylib_sha256": "4" * 64,
        }
        experiment = {
            "schema_version": 1,
            "protocol_sha256": digest(protocol_path),
            **fixed_hashes,
        }
        (matrix / "experiment_manifest.json").write_text(
            json.dumps(experiment), encoding="utf-8"
        )
        bounds = protocol()["gates"]["queue_bounds"]
        for sequence in SEQUENCES:
            for repetition in range(1, 4):
                run = matrix / f"{sequence}_r{repetition}"
                run.mkdir()
                summary = "\n".join(
                    [
                        "onnx_backend_requested=cuda",
                        f"frames_requested={FRAME_COUNTS[sequence]}",
                        f"ate_similarity_rmse_m={ate}",
                        "tracked_fraction=0.99",
                        "ms_per_frame_total=49.0",
                        "sim3_backend_max_abs_log_scale_correction=4.0",
                        "sim3_backend_max_committed_abs_log_scale=0.2",
                        "sim3_backend_scale_jump_rejections_total=1",
                        "global_ba_inactive_edges_retained=4096",
                        "global_ba_max_free_pose_count=256",
                        "long_loop_frames_indexed=1000",
                    ]
                )
                summary_path = run / "summary.txt"
                summary_path.write_text(summary + "\n", encoding="utf-8")
                manifest = {
                    "sequence": sequence,
                    "repetition": repetition,
                    "exit_code": 0,
                    "summary_sha256": digest(summary_path),
                    "sampled_peak_working_set_bytes": 4 * 1024**3,
                    "sampled_peak_gpu_memory_bytes": 4 * 1024**3,
                    "queue_bounds": bounds,
                    "protocol_sha256": digest(protocol_path),
                    **fixed_hashes,
                }
                (run / "run_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
        return matrix, protocol_path

    def test_complete_euroc_matrix_passes_but_is_not_sota_without_public_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            matrix, protocol_path = self.make_matrix(Path(raw))
            report = MODULE.evaluate(matrix, protocol_path, None)
            self.assertEqual(report["successful_runs"], 33)
            self.assertTrue(report["euroc_engineering_gate_pass"])
            self.assertAlmostEqual(report["all_sequence_mean_sim3_ate_rmse_m"], 0.019)
            self.assertAlmostEqual(report["all_sequence_median_sim3_ate_rmse_m"], 0.019)
            self.assertAlmostEqual(report["all_sequence_worst_sim3_ate_rmse_m"], 0.019)
            self.assertEqual(report["scale_jump_rejections_total"], 33)
            self.assertFalse(report["public_frontier_gate_pass"])
            self.assertFalse(report["claimable_sota"])

    def test_failed_run_stays_in_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            matrix, protocol_path = self.make_matrix(Path(raw))
            path = matrix / "MH_01_easy_r1" / "run_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["exit_code"] = 9
            path.write_text(json.dumps(manifest), encoding="utf-8")
            report = MODULE.evaluate(matrix, protocol_path, None)
            self.assertEqual(report["successful_runs"], 32)
            self.assertEqual(report["failed_or_missing_runs"], 1)
            self.assertFalse(report["euroc_engineering_gate_pass"])

    def test_tampered_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            matrix, protocol_path = self.make_matrix(Path(raw))
            summary = matrix / "MH_01_easy_r1" / "summary.txt"
            summary.write_text(summary.read_text(encoding="utf-8") + "tampered=true\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                MODULE.evaluate(matrix, protocol_path, None)

    def test_verified_public_frontier_completes_claim_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            matrix, protocol_path = self.make_matrix(root)
            evidence = root / "public.json"
            evidence.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "verified_public_frontier",
                        "benchmark": "ETH3D SLAM",
                        "result_url": "https://example.test/public-result",
                        "released_artifact_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            report = MODULE.evaluate(matrix, protocol_path, evidence)
            self.assertTrue(report["claimable_sota"])

    def test_slow_run_is_a_failure_even_when_accuracy_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            matrix, protocol_path = self.make_matrix(Path(raw))
            run = matrix / "V2_03_difficult_r3"
            summary_path = run / "summary.txt"
            summary = summary_path.read_text(encoding="utf-8").replace(
                "ms_per_frame_total=49.0", "ms_per_frame_total=50.1"
            )
            summary_path.write_text(summary, encoding="utf-8")
            manifest_path = run / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["summary_sha256"] = digest(summary_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = MODULE.evaluate(matrix, protocol_path, None)
            self.assertEqual(report["successful_runs"], 32)
            failed = [row for row in report["runs"] if not row["success"]]
            self.assertIn("input-rate budget exceeded", failed[0]["reasons"])

    def test_non_strict_backend_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            matrix, protocol_path = self.make_matrix(Path(raw))
            run = matrix / "MH_01_easy_r1"
            summary_path = run / "summary.txt"
            summary_path.write_text(
                summary_path.read_text(encoding="utf-8").replace(
                    "onnx_backend_requested=cuda",
                    "onnx_backend_requested=cuda_then_cpu",
                ),
                encoding="utf-8",
            )
            manifest_path = run / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["summary_sha256"] = digest(summary_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = MODULE.evaluate(matrix, protocol_path, None)
            failed = [row for row in report["runs"] if not row["success"]]
            self.assertEqual(report["successful_runs"], 32)
            self.assertIn("strict CUDA backend", failed[0]["reasons"][0])


if __name__ == "__main__":
    unittest.main()
