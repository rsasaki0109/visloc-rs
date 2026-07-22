from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "benchmarks" / "protocols" / "ssfm_external_baselines_v1.json"
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


if __name__ == "__main__":
    unittest.main()
