from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "protocols" / "ssfm_external_baselines_v1.json"
)

from ssfm_external_baseline_evidence import (  # noqa: E402
    validate_external_baseline_manifest,
)


class ExternalSsfmBaselineProtocolTests(unittest.TestCase):
    def test_companion_protocol_binds_heldout_freeze(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        heldout = REPO_ROOT / protocol["heldout_protocol"]["path"]
        self.assertEqual(
            hashlib.sha256(heldout.read_bytes()).hexdigest(),
            protocol["heldout_protocol"]["sha256"],
        )
        self.assertEqual(set(protocol["engines"]), {"gluemap", "instantsfm"})
        self.assertEqual(
            set(protocol["reporting"]["required_cells_per_sequence"]),
            {"gluemap", "instantsfm"},
        )

    def test_gt_remains_sealed_until_external_cells_exit(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        policy = protocol["hardware_policy"]["ground_truth_policy"]
        self.assertIn("GT remains unmaterialized", policy)
        self.assertTrue(protocol["hardware_policy"]["serial_with_all_other_timed_engines"])
        self.assertTrue(
            any(
                rule.startswith("No external baseline may run after GT materialization")
                for rule in protocol["stop_rules"]
            )
        )

    def test_source_outage_is_not_prematurely_called_a_dnf(self) -> None:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        instant = protocol["engines"]["instantsfm"]
        self.assertEqual(instant["source_status_at_freeze"], "unavailable")
        self.assertIn("not yet a benchmark DNF", instant["attempt_policy"])
        self.assertEqual(
            protocol["engines"]["gluemap"]["revision"],
            "adc9e4bb5f41014d3f7c157a879edc278588c829",
        )
        self.assertTrue(
            protocol["engines"]["gluemap"]["configuration"]["use_gt_intrinsics"]
        )

    def test_external_manifest_requires_two_honest_pre_gt_cells(self) -> None:
        manifest = {
            "sequence": "V1_02_medium",
            "heldout_protocol_sha256": "a" * 64,
            "external_protocol_sha256": "b" * 64,
            "ground_truth_read": False,
            "all_engine_processes_exited": True,
            "results": {
                engine: {
                    "status": "dnf",
                    "reason": "fixture source/install failure",
                    "source_revision": "unavailable-at-2026-07-22",
                    "attempt": {"command": ["fixture"], "returncode": 1},
                    "trajectory": None,
                }
                for engine in ("gluemap", "instantsfm")
            },
        }
        results = validate_external_baseline_manifest(
            manifest,
            sequence="V1_02_medium",
            heldout_protocol_sha256="a" * 64,
            external_protocol_sha256="b" * 64,
            manifest_dir=Path("."),
        )
        self.assertEqual(set(results), {"gluemap", "instantsfm"})

        manifest["results"].pop("instantsfm")
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            validate_external_baseline_manifest(
                manifest,
                sequence="V1_02_medium",
                heldout_protocol_sha256="a" * 64,
                external_protocol_sha256="b" * 64,
                manifest_dir=Path("."),
            )

    def test_external_manifest_rejects_gt_read_or_unattempted_dnf(self) -> None:
        cell = {
            "status": "dnf",
            "reason": "fixture",
            "source_revision": "fixture",
            "attempt": {"command": ["fixture"], "returncode": 1},
            "trajectory": None,
        }
        manifest = {
            "sequence": "V2_02_medium",
            "heldout_protocol_sha256": "a" * 64,
            "external_protocol_sha256": "b" * 64,
            "ground_truth_read": True,
            "all_engine_processes_exited": True,
            "results": {"gluemap": dict(cell), "instantsfm": dict(cell)},
        }
        with self.assertRaisesRegex(ValueError, "GT isolation"):
            validate_external_baseline_manifest(
                manifest,
                sequence="V2_02_medium",
                heldout_protocol_sha256="a" * 64,
                external_protocol_sha256="b" * 64,
                manifest_dir=Path("."),
            )
        manifest["ground_truth_read"] = False
        manifest["results"]["gluemap"]["attempt"] = {"returncode": 1}
        with self.assertRaisesRegex(ValueError, "attempted command"):
            validate_external_baseline_manifest(
                manifest,
                sequence="V2_02_medium",
                heldout_protocol_sha256="a" * 64,
                external_protocol_sha256="b" * 64,
                manifest_dir=Path("."),
            )


if __name__ == "__main__":
    unittest.main()
