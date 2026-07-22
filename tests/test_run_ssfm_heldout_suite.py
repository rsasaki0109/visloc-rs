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

EXTERNAL_PROTOCOL = (
    REPO_ROOT / "benchmarks" / "protocols" / "ssfm_external_baselines_v1.json"
)

from run_ssfm_heldout_suite import main  # noqa: E402


class HeldoutSsfmSuiteRunnerTests(unittest.TestCase):
    def test_three_sequence_failures_still_produce_five_engine_summary(self) -> None:
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
            external_setup = root / "external_setup.json"
            external_setup.write_text("{}", encoding="utf-8")

            with patch(
                "sys.argv",
                [
                    "run_ssfm_heldout_suite.py",
                    "--protocol",
                    str(protocol_path),
                    "--external-protocol",
                    str(EXTERNAL_PROTOCOL),
                    "--external-setup-manifest",
                    str(external_setup),
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

            from verify_ssfm_heldout_suite import main as verify_main

            audit_path = root / "release_audit.json"
            with patch(
                "sys.argv",
                [
                    "verify_ssfm_heldout_suite.py",
                    "--protocol",
                    str(protocol_path),
                    "--suite-root",
                    str(output),
                    "--out",
                    str(audit_path),
                ],
            ):
                self.assertEqual(verify_main(), 0)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "verified")
            verifier_path = Path(audit["verifier"]["path"])
            self.assertEqual(
                audit["verifier"]["sha256"],
                hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(
                all(
                    cell["kind"] == "runner_failure"
                    for cell in audit["sequence_audit"].values()
                )
            )

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

    def test_release_verifier_rejects_mixed_resource_sampling_cadence(self) -> None:
        from verify_ssfm_heldout_suite import require_uniform_sampling_cadence

        with self.assertRaisesRegex(ValueError, "cadence mismatch"):
            require_uniform_sampling_cadence(
                [("hierarchical", 0.5), ("external", 1.0)]
            )

    def test_release_verifier_accepts_deferred_gt_success_chain(self) -> None:
        from verify_ssfm_heldout_suite import ENGINES, verify_success_sequence

        def evidence(path: Path) -> dict:
            return {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        with tempfile.TemporaryDirectory() as raw_root:
            sequence_root = Path(raw_root)
            for name in (
                "prepared",
                "hierarchical",
                "colmap",
                "external",
                "ground_truth",
                "final",
            ):
                (sequence_root / name).mkdir()
            prepared_path = sequence_root / "prepared" / "manifest.json"
            prepared_path.write_text(
                json.dumps(
                    {
                        "stages": {
                            "rectification": {"resource_poll_seconds": 0.5},
                            "superpoint": {"resource_poll_seconds": 0.5},
                        }
                    }
                ),
                encoding="utf-8",
            )
            hierarchical_path = sequence_root / "hierarchical" / "manifest.json"
            hierarchical_path.write_text(
                json.dumps(
                    {
                        "finished_utc": "2026-07-22T00:00:00+00:00",
                        "protocol": {"ground_truth_read": False},
                        "mapper": {"resource_poll_seconds": 0.5},
                    }
                ),
                encoding="utf-8",
            )
            colmap_path = sequence_root / "colmap" / "manifest.json"
            colmap_path.write_text(
                json.dumps(
                    {
                        "finished_utc": "2026-07-22T00:00:01+00:00",
                        "ground_truth_read": False,
                        "stages": {
                            "feature_extraction": {"resource_poll_seconds": 0.5},
                            "sequential_matching": {"resource_poll_seconds": 0.5},
                            "incremental_mapping": {"resource_poll_seconds": 0.5},
                            "global_mapping": {"resource_poll_seconds": 0.5},
                        },
                    }
                ),
                encoding="utf-8",
            )
            external_protocol_sha256 = "e" * 64
            external_path = sequence_root / "external" / "manifest.json"
            external_path.write_text(
                json.dumps(
                    {
                        "sequence": "V1_02_medium",
                        "heldout_protocol_sha256": "d" * 64,
                        "external_protocol_sha256": external_protocol_sha256,
                        "ground_truth_read": False,
                        "all_engine_processes_exited": True,
                        "finished_utc": "2026-07-22T00:00:01.500000+00:00",
                        "results": {
                            engine: {
                                "status": "dnf",
                                "reason": "fixture",
                                "source_revision": "fixture",
                                "attempt": {"command": ["fixture"], "returncode": 1},
                                "trajectory": None,
                            }
                            for engine in ("gluemap", "instantsfm")
                        },
                    }
                ),
                encoding="utf-8",
            )
            gt_csv = sequence_root / "ground_truth" / "data.csv"
            gt_csv.write_text("#timestamp\n", encoding="utf-8")
            protocol_sha256 = "d" * 64
            gt_manifest_path = sequence_root / "ground_truth" / "manifest.json"
            gt_manifest_path.write_text(
                json.dumps(
                    {
                        "protocol_sha256": protocol_sha256,
                        "materialized_utc": "2026-07-22T00:00:02+00:00",
                        "engine_exit_evidence": {
                            "hierarchical": evidence(hierarchical_path),
                            "colmap": evidence(colmap_path),
                            "external": evidence(external_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            final_path = sequence_root / "final" / "manifest.json"
            final_path.write_text(
                json.dumps(
                    {
                        "protocol_sha256": protocol_sha256,
                        "results": {
                            engine: {
                                "status": "success",
                                "resource_poll_seconds": (
                                    {"frontend": 0.5, "mapper": 0.5}
                                    if engine == "visloc_hierarchical"
                                    else 0.5
                                ),
                            }
                            for engine in ENGINES
                        },
                        "input_manifests": {
                            "prepared": evidence(prepared_path),
                            "hierarchical": evidence(hierarchical_path),
                            "colmap": evidence(colmap_path),
                            "external": evidence(external_path),
                            "ground_truth_materializer": evidence(gt_manifest_path),
                        },
                        "ground_truth": evidence(gt_csv),
                    }
                ),
                encoding="utf-8",
            )
            stages = {
                stage: ["fixture"]
                for stage in (
                    "hierarchical",
                    "colmap",
                    "external",
                    "materialize_ground_truth",
                    "finalize",
                )
            }
            runner = {
                "status": "success",
                "sequence": "V1_02_medium",
                "external_protocol": {"sha256": external_protocol_sha256},
                "ground_truth_materialized_only_after_timed_engines_exited": True,
                "commands": stages,
                "returncodes": {stage: 0 for stage in stages},
                "final_manifest_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
            }

            audit = verify_success_sequence(
                sequence_root,
                runner,
                protocol_sha256,
            )

            self.assertEqual(audit["kind"], "success")
            self.assertEqual(set(audit["result_statuses"]), set(ENGINES))


if __name__ == "__main__":
    unittest.main()
