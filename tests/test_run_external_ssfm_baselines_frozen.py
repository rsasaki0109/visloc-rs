from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_external_ssfm_baselines_frozen import main  # noqa: E402


HELDOUT = REPO_ROOT / "benchmarks" / "protocols" / "ssfm_heldout_euroc_v1.json"
EXTERNAL = (
    REPO_ROOT / "benchmarks" / "protocols" / "ssfm_external_baselines_v1.json"
)


class ExternalSsfmBaselineRunnerTests(unittest.TestCase):
    def make_prepared(self, root: Path, sequence: str, frames: int = 2) -> Path:
        prepared = root / "prepared"
        (prepared / "rect" / "image_0").mkdir(parents=True)
        (prepared / "rect" / "calib.txt").write_text("fixture", encoding="utf-8")
        (prepared / "manifest.json").write_text(
            json.dumps(
                {
                    "protocol_sha256": hashlib.sha256(HELDOUT.read_bytes()).hexdigest(),
                    "ground_truth_read": False,
                    "sequence": sequence,
                    "expected_frames": frames,
                }
            ),
            encoding="utf-8",
        )
        return prepared

    def run_fixture(self, root: Path, setup: dict) -> dict:
        sequence = "V1_02_medium"
        prepared = self.make_prepared(root, sequence)
        setup_path = root / "setup.json"
        setup["external_protocol_sha256"] = hashlib.sha256(
            EXTERNAL.read_bytes()
        ).hexdigest()
        setup_path.write_text(json.dumps(setup), encoding="utf-8")
        output = root / "external"
        with patch(
            "sys.argv",
            [
                "run_external_ssfm_baselines_frozen.py",
                "--protocol",
                str(HELDOUT),
                "--external-protocol",
                str(EXTERNAL),
                "--sequence",
                sequence,
                "--prepared-dir",
                str(prepared),
                "--setup-manifest",
                str(setup_path),
                "--out-dir",
                str(output),
                "--poll-seconds",
                "0.01",
            ],
        ):
            self.assertEqual(main(), 0)
        return json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    def test_setup_failures_remain_two_explicit_dnf_cells(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            setup = {
                "engines": {
                    engine: {
                        "status": "source_unavailable",
                        "reason": "fixture outage",
                        "source_identity": "fixture-unavailable",
                        "attempt": {"command": ["git", "ls-remote"], "returncode": 128},
                    }
                    for engine in ("gluemap", "instantsfm")
                }
            }
            manifest = self.run_fixture(Path(raw_root), setup)
            self.assertFalse(manifest["ground_truth_read"])
            self.assertTrue(manifest["all_engine_processes_exited"])
            self.assertTrue(
                all(cell["status"] == "dnf" for cell in manifest["results"].values())
            )

    @unittest.skipUnless(sys.platform == "win32", "Windows-only bench harness")
    def test_ready_adapters_emit_valid_success_cells(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            adapter = root / "adapter.py"
            adapter.write_text(
                "from pathlib import Path\n"
                "import json,sys\n"
                "out=Path(sys.argv[1])\n"
                "(out/'trajectory.tum').write_text('0 0 0 0 0 0 0 1\\n1 1 0 0 0 0 0 1\\n')\n"
                "(out/'result.json').write_text(json.dumps({"
                "'registered_images':2,'points3d':3,'mean_reprojection_px':0.5}))\n",
                encoding="utf-8",
            )
            adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
            setup = {
                "engines": {
                    engine: {
                        "status": "ready",
                        "source_revision": f"{engine}-fixture-revision",
                        "source_tree_sha256": engine * 8,
                        "adapter": {
                            "path": str(adapter),
                            "sha256": adapter_sha256,
                            "command_template": [
                                sys.executable,
                                "{adapter}",
                                "{output_path}",
                            ],
                        },
                    }
                    for engine in ("gluemap", "instantsfm")
                }
            }
            manifest = self.run_fixture(root, setup)
            for cell in manifest["results"].values():
                self.assertEqual(cell["status"], "success")
                self.assertEqual(cell["registered_images"], 2)
                self.assertEqual(cell["registration_rate"], 1.0)
                self.assertTrue(Path(cell["trajectory"]["path"]).is_file())

    def test_unattempted_setup_is_not_accepted_as_benchmark_dnf(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            setup = {
                "engines": {
                    engine: {
                        "status": "setup_incomplete",
                        "reason": "not installed yet",
                        "source_identity": "fixture",
                        "attempt": {"command": ["fixture"], "returncode": 1},
                    }
                    for engine in ("gluemap", "instantsfm")
                }
            }
            with self.assertRaisesRegex(ValueError, "not an evidence-backed DNF"):
                self.run_fixture(Path(raw_root), setup)

    def test_ready_adapter_dependency_hash_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            adapter = root / "adapter.py"
            dependency = root / "inner.py"
            adapter.write_text("pass\n", encoding="utf-8")
            dependency.write_text("pass\n", encoding="utf-8")
            setup = {
                "engines": {
                    engine: {
                        "status": "ready",
                        "source_revision": "fixture",
                        "adapter": {
                            "path": str(adapter),
                            "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                            "dependencies": [
                                {"path": str(dependency), "sha256": "0" * 64}
                            ],
                            "command_template": [sys.executable, "{adapter}"],
                        },
                    }
                    for engine in ("gluemap", "instantsfm")
                }
            }
            with self.assertRaisesRegex(ValueError, "dependency hash mismatch"):
                self.run_fixture(root, setup)


if __name__ == "__main__":
    unittest.main()
