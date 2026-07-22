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

from summarize_ssfm_heldout_suite import (  # noqa: E402
    ENGINES,
    load_sequence_results,
    main,
)


class HeldoutSsfmSuiteSummaryTests(unittest.TestCase):
    def test_all_runner_failures_are_aggregated_instead_of_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            suite_root = root / "suite"
            sequences = ["V1_02_medium", "V1_03_difficult", "V2_02_medium"]
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "protocol_id": "ssfm-heldout-test",
                        "selection": {"held_out_sequences": sequences},
                    }
                ),
                encoding="utf-8",
            )
            protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            for sequence in sequences:
                sequence_root = suite_root / sequence
                sequence_root.mkdir(parents=True)
                (sequence_root / "manifest.json").write_text(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "protocol_sha256": protocol_sha256,
                            "status": "failed",
                            "failure_reason": "fixture failure",
                        }
                    ),
                    encoding="utf-8",
                )
            output_path = root / "summary.json"

            with patch(
                "sys.argv",
                [
                    "summarize_ssfm_heldout_suite.py",
                    "--protocol",
                    str(protocol_path),
                    "--suite-root",
                    str(suite_root),
                    "--out",
                    str(output_path),
                ],
            ):
                self.assertEqual(main(), 0)

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            for engine in ENGINES:
                aggregate = summary["aggregate"][engine]
                self.assertEqual(aggregate["success_count"], 0)
                self.assertEqual(len(aggregate["failures"]), 3)
                self.assertEqual(aggregate["worst_outcome"], "dnf")
            self.assertFalse(summary["internal_colmap_frontier_gate"]["passed"])

    def test_runner_failure_becomes_three_explicit_dnf_cells(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sequence = "V1_02_medium"
            sequence_root = root / sequence
            sequence_root.mkdir()
            runner = {
                "sequence": sequence,
                "protocol_sha256": "a" * 64,
                "status": "failed",
                "failure_reason": "held-out input preparation failed",
            }
            (sequence_root / "manifest.json").write_text(
                json.dumps(runner), encoding="utf-8"
            )

            evidence, results = load_sequence_results(root, sequence, "a" * 64)

            self.assertEqual(evidence["kind"], "runner_failure")
            self.assertEqual(set(results), set(ENGINES))
            for cell in results.values():
                self.assertEqual(cell["status"], "dnf")
                self.assertEqual(cell["registration_rate"], 0.0)
                self.assertIn("input preparation failed", cell["reason"])

    def test_nonfailed_runner_without_final_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sequence = "V1_03_difficult"
            sequence_root = root / sequence
            sequence_root.mkdir()
            (sequence_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "sequence": sequence,
                        "protocol_sha256": "b" * 64,
                        "status": "success",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "is not failed"):
                load_sequence_results(root, sequence, "b" * 64)

    def test_final_manifest_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            sequence = "V2_02_medium"
            sequence_root = root / sequence
            final_root = sequence_root / "final"
            final_root.mkdir(parents=True)
            cells = {engine: {"status": "dnf", "reason": "fixture"} for engine in ENGINES}
            (final_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "sequence": sequence,
                        "protocol_sha256": "c" * 64,
                        "results": cells,
                    }
                ),
                encoding="utf-8",
            )
            (sequence_root / "manifest.json").write_text("not json", encoding="utf-8")

            evidence, results = load_sequence_results(root, sequence, "c" * 64)

            self.assertEqual(evidence["kind"], "final")
            self.assertEqual(results, cells)


if __name__ == "__main__":
    unittest.main()
