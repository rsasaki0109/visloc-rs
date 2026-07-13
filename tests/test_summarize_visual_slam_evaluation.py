import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_visual_slam_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_visual_slam_evaluation", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def experiment_manifest(*, schema_version: int = 3, include_gate: bool = True) -> dict:
    protocol = {
        "common_arguments": [],
        "variant_arguments": {
            "no_loop": [],
            "appearance_loop": [
                "--pose-graph-refinement",
                "--pose-graph-refinement-appearance-loops",
                "--pose-graph-refinement-gnc",
                "--pose-graph-refinement-pcm",
                "--pose-graph-refinement-pcm-pairwise-only",
            ],
        },
    }
    if include_gate:
        protocol["resource_gate"] = {
            "minimum_available_physical_bytes": 4 * 1024**3,
            "minimum_commit_headroom_bytes": 4 * 1024**3,
            "sample_interval_seconds": 5,
        }
    return {"schema_version": schema_version, "protocol": protocol}


class MatrixResourceGateTests(unittest.TestCase):
    def write_experiment(self, root: Path, manifest: dict) -> None:
        (root / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_schema_three_resource_gate_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest(include_gate=False))
            with self.assertRaisesRegex(ValueError, "missing its resource gate"):
                MODULE.validate_matrix_protocol(root)

    def test_schema_two_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest(schema_version=2))
            with self.assertRaisesRegex(ValueError, "schema_version=3"):
                MODULE.validate_matrix_protocol(root)

    def test_completed_run_must_satisfy_recorded_resource_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest())
            run_dir = root / "MH_01_easy_no_loop_r01"
            run_dir.mkdir()
            (run_dir / "summary.txt").write_text("", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "name": run_dir.name,
                        "sequence": "MH_01_easy",
                        "variant": "no_loop",
                        "repetition": 1,
                        "exit_code": 0,
                        "validation_error": None,
                        "minimum_available_physical_bytes": 4 * 1024**3 - 1,
                        "minimum_commit_headroom_bytes": 4 * 1024**3,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "physical-memory minimum"):
                MODULE.load_matrix(root)


if __name__ == "__main__":
    unittest.main()
