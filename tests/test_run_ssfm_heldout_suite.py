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

from run_ssfm_heldout_suite import main  # noqa: E402


class HeldoutSsfmSuiteRunnerTests(unittest.TestCase):
    def test_three_preparation_failures_still_produce_complete_suite_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            protocol_path = (
                REPO_ROOT / "benchmarks" / "protocols" / "ssfm_heldout_euroc_v1.json"
            )
            protocol_bytes = protocol_path.read_bytes()
            protocol = json.loads(protocol_bytes)
            protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
            sequences = protocol["selection"]["held_out_sequences"]
            extracted = root / "extracted"
            extracted.mkdir()
            sequence_evidence = {}
            for sequence in sequences:
                (extracted / sequence / "mav0").mkdir(parents=True)
                sequence_evidence[sequence] = {
                    "ground_truth_materialized": False,
                    "ground_truth_read": False,
                }
            (extracted / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "protocol_sha256": protocol_sha256,
                        "ground_truth_read": False,
                        "sequences": sequence_evidence,
                    }
                ),
                encoding="utf-8",
            )
            downloads = root / "downloads"
            downloads.mkdir()
            (downloads / "download_manifest.json").write_text(
                json.dumps({"status": "fixture"}), encoding="utf-8"
            )
            output = root / "suite"
            python = Path(sys.executable)

            with patch(
                "sys.argv",
                [
                    "run_ssfm_heldout_suite.py",
                    "--protocol",
                    str(protocol_path),
                    "--extracted-root",
                    str(extracted),
                    "--download-dir",
                    str(downloads),
                    "--out-dir",
                    str(output),
                    "--hierarchical-exe",
                    str(python),
                    "--hierarchical-build-revision",
                    protocol["policy"]["source_revision"],
                    "--colmap",
                    str(python),
                    "--python",
                    str(python),
                    "--device",
                    "cpu",
                ],
            ):
                self.assertEqual(main(), 0)

            suite_manifest = json.loads(
                (output / "suite_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(suite_manifest["status"], "complete")
            self.assertEqual(list(suite_manifest["runs"]), sequences)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            for aggregate in summary["aggregate"].values():
                self.assertEqual(aggregate["success_count"], 0)
                self.assertEqual(len(aggregate["failures"]), 3)
                self.assertEqual(aggregate["worst_outcome"], "dnf")

    def test_preflight_rejects_materialized_ground_truth(self) -> None:
        from run_ssfm_heldout_suite import validate_extraction

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sequence = "V1_02_medium"
            (root / sequence / "mav0").mkdir(parents=True)
            (root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "protocol_sha256": "a" * 64,
                        "ground_truth_read": False,
                        "sequences": {
                            sequence: {"ground_truth_materialized": True}
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "materialized before suite"):
                validate_extraction(root, [sequence], "a" * 64)

    def test_changed_frozen_files_detects_post_freeze_mutation(self) -> None:
        from run_ssfm_heldout_suite import changed_frozen_files

        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "frozen.txt"
            path.write_text("before", encoding="utf-8")
            evidence = {
                "fixture": {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
            self.assertEqual(changed_frozen_files(evidence), [])
            path.write_text("after", encoding="utf-8")
            self.assertEqual(changed_frozen_files(evidence), ["fixture"])


if __name__ == "__main__":
    unittest.main()
