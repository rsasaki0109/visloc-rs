import importlib.util
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_vslam_euroc_sota_v4.py"
)
SPEC = importlib.util.spec_from_file_location("run_vslam_euroc_sota_v4", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


BOUNDS = {
    "inactive_edge_cap": 4096,
    "max_free_poses": 256,
    "long_loop_max_indexed_frames": 1000,
}
REQUIRED = [
    "--onnx-cuda",
    "--imu",
    "--loop-closure",
    "--global-ba",
    "--gba-widen-t0",
    "--sim3-backend",
    "--long-loop",
    "--pipeline-prefetch",
    "--gba-inactive-edge-cap",
    "4096",
    "--gba-max-free-poses",
    "256",
    "--ll-max-indexed-frames",
    "1000",
    "--s3b-max-abs-log-scale-correction",
    "4.0",
]
GATES = {
    "queue_bounds": BOUNDS,
    "max_committed_abs_log_scale": 4.0,
    "required_onnx_backend": "cuda",
    "required_onnx_full_update_graph": True,
    "forbid_grouped_onnx_correlation": True,
    "required_native_cuda_correlation": True,
    "required_native_cuda_correlation_abi": 3,
    "required_final_refinement_iterations": 12,
    "required_pipeline_prefetch": True,
}


class VslamSotaV4RunnerTests(unittest.TestCase):
    def test_configuration_requires_full_stack_and_exact_bounds(self) -> None:
        MODULE.validate_configuration(
            {
                "schema_version": 1,
                "arguments": REQUIRED,
                "queue_bounds": BOUNDS,
                "long_loop_superpoint_model": "superpoint.onnx",
            },
            GATES,
        )
        with self.assertRaisesRegex(ValueError, "missing --long-loop"):
            MODULE.validate_configuration(
                {
                    "schema_version": 1,
                    "arguments": [value for value in REQUIRED if value != "--long-loop"],
                    "queue_bounds": BOUNDS,
                    "long_loop_superpoint_model": "superpoint.onnx",
                },
                GATES,
            )
        with self.assertRaisesRegex(ValueError, "runner-owned"):
            MODULE.validate_configuration(
                {
                    "schema_version": 1,
                    "arguments": [*REQUIRED, "--seed", "99"],
                    "queue_bounds": BOUNDS,
                    "long_loop_superpoint_model": "superpoint.onnx",
                },
                GATES,
            )
        with self.assertRaisesRegex(ValueError, "cannot combine strict CUDA"):
            MODULE.validate_configuration(
                {
                    "schema_version": 1,
                    "arguments": [*REQUIRED, "--onnx-cpu"],
                    "queue_bounds": BOUNDS,
                    "long_loop_superpoint_model": "superpoint.onnx",
                },
                GATES,
            )
        with self.assertRaisesRegex(ValueError, "measured-negative"):
            MODULE.validate_configuration(
                {
                    "schema_version": 1,
                    "arguments": [*REQUIRED, "--onnx-correlation"],
                    "queue_bounds": BOUNDS,
                    "long_loop_superpoint_model": "superpoint.onnx",
                },
                GATES,
            )

    def test_model_bundle_hash_binds_paths_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").mkdir()
            first = root / "a" / "weights.onnx"
            first.write_bytes(b"one")
            before = MODULE.model_bundle_sha256(root)
            first.write_bytes(b"two")
            self.assertNotEqual(before, MODULE.model_bundle_sha256(root))
            first.write_bytes(b"one")
            moved = root / "weights.onnx"
            first.rename(moved)
            self.assertNotEqual(before, MODULE.model_bundle_sha256(root))

    def test_nested_sequence_resolution_and_camera_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sequence = root / "machine_hall" / "MH_01_easy"
            csv_dir = sequence / "mav0" / "cam0"
            csv_dir.mkdir(parents=True)
            (csv_dir / "data.csv").write_text(
                "#timestamp,filename\n1,1.png\n2,2.png\n", encoding="utf-8"
            )
            self.assertEqual(MODULE.resolve_sequence(root, "MH_01_easy"), sequence)
            self.assertEqual(MODULE.camera_rows(sequence), 2)

    def test_gpu_peak_parser_selects_target_pid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "gpu.log"
            path.write_text("12, 100\n99, 200\n12, 150\n", encoding="utf-8")
            self.assertEqual(MODULE.parse_gpu_peak(path, 12), 150 * 1024 * 1024)
            self.assertIsNone(MODULE.parse_gpu_peak(path, 77))

    def test_run_one_preserves_summary_and_exit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            summary = run_dir / "summary.txt"
            command = (
                "from pathlib import Path; import time; "
                f"Path({str(summary)!r}).write_text('ok=true\\n', encoding='utf-8'); "
                "time.sleep(0.1)"
            )
            manifest = {
                "sequence": "fixture",
                "repetition": 1,
                "exit_code": None,
            }
            result = MODULE.run_one(
                Path(sys.executable), ["-c", command], run_dir, manifest, 0.02
            )
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(
                result["summary_sha256"], hashlib.sha256(summary.read_bytes()).hexdigest()
            )
            self.assertTrue((run_dir / "run_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
