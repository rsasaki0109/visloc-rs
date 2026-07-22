from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_ssfm_heldout_euroc import extract_archive  # noqa: E402
from materialize_ssfm_heldout_ground_truth import main as materialize_main  # noqa: E402


SEQUENCE = "V1_02_medium"
GT_MEMBER = f"dataset/{SEQUENCE}/mav0/state_groundtruth_estimate0/data.csv"


def write_archive(path: Path) -> bytes:
    ground_truth = b"#timestamp,p_RS_R_x\n1,0.0\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"dataset/{SEQUENCE}/mav0/cam0/data.csv", "timestamp,filename\n")
        archive.writestr(GT_MEMBER, ground_truth)
    return ground_truth


class HeldoutGroundTruthIsolationTests(unittest.TestCase):
    def test_initial_extraction_defers_all_ground_truth_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            archive_path = root / "room.zip"
            write_archive(archive_path)
            output = root / "extracted"

            evidence = extract_archive(archive_path, [SEQUENCE], output)

            self.assertEqual(evidence["deferred_ground_truth_members"], 1)
            self.assertFalse(evidence["ground_truth_materialized"])
            self.assertTrue((output / SEQUENCE / "mav0" / "cam0" / "data.csv").is_file())
            self.assertFalse(
                (output / SEQUENCE / "mav0" / "state_groundtruth_estimate0").exists()
            )

    def test_materializer_requires_and_records_engine_exit_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            archive_path = root / "room.zip"
            expected_ground_truth = write_archive(archive_path)
            protocol_path = root / "protocol.json"
            protocol = {
                "protocol_id": "ssfm-heldout-test",
                "selection": {"held_out_sequences": [SEQUENCE]},
                "inputs": {
                    "official_archives": [
                        {
                            "name": archive_path.name,
                            "size_bytes": archive_path.stat().st_size,
                            "checksum_algorithm": "MD5",
                            "checksum": hashlib.md5(archive_path.read_bytes()).hexdigest(),
                            "selected_sequences": [SEQUENCE],
                        }
                    ]
                },
            }
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
            external_protocol_path = root / "external_protocol.json"
            external_protocol_path.write_text(
                json.dumps(
                    {"heldout_protocol": {"sha256": protocol_sha256}}
                ),
                encoding="utf-8",
            )
            external_protocol_sha256 = hashlib.sha256(
                external_protocol_path.read_bytes()
            ).hexdigest()
            download_root = root / "downloads"
            download_root.mkdir()
            (download_root / "download_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "protocol_sha256": protocol_sha256,
                        "archives": [
                            {
                                "name": archive_path.name,
                                "path": str(archive_path),
                                "size_bytes": archive_path.stat().st_size,
                                "checksum": protocol["inputs"]["official_archives"][0][
                                    "checksum"
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hierarchical_path = root / "hierarchical.json"
            hierarchical_path.write_text(
                json.dumps({"protocol": {"ground_truth_read": False}}),
                encoding="utf-8",
            )
            colmap_path = root / "colmap.json"
            colmap_path.write_text(
                json.dumps({"ground_truth_read": False}), encoding="utf-8"
            )
            external_path = root / "external.json"
            external_path.write_text(
                json.dumps(
                    {
                        "sequence": SEQUENCE,
                        "heldout_protocol_sha256": protocol_sha256,
                        "external_protocol_sha256": external_protocol_sha256,
                        "ground_truth_read": False,
                        "all_engine_processes_exited": True,
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
            output = root / "ground_truth"

            with patch(
                "sys.argv",
                [
                    "materialize_ssfm_heldout_ground_truth.py",
                    "--protocol",
                    str(protocol_path),
                    "--sequence",
                    SEQUENCE,
                    "--download-dir",
                    str(download_root),
                    "--external-protocol",
                    str(external_protocol_path),
                    "--hierarchical-manifest",
                    str(hierarchical_path),
                    "--colmap-manifest",
                    str(colmap_path),
                    "--external-manifest",
                    str(external_path),
                    "--out-dir",
                    str(output),
                ],
            ):
                self.assertEqual(materialize_main(), 0)

            self.assertEqual((output / "data.csv").read_bytes(), expected_ground_truth)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(
                manifest["ground_truth_first_read_after_all_timed_engines_exited"]
            )
            self.assertEqual(
                manifest["engine_exit_evidence"]["hierarchical"]["sha256"],
                hashlib.sha256(hierarchical_path.read_bytes()).hexdigest(),
            )

    def test_materializer_rejects_manifest_that_does_not_prove_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            hierarchical = root / "hierarchical.json"
            hierarchical.write_text(
                json.dumps({"protocol": {"ground_truth_read": True}}),
                encoding="utf-8",
            )
            colmap = root / "colmap.json"
            colmap.write_text(json.dumps({"ground_truth_read": False}), encoding="utf-8")

            from materialize_ssfm_heldout_ground_truth import engine_exit_evidence

            with self.assertRaisesRegex(ValueError, "hierarchical manifest"):
                engine_exit_evidence(
                    hierarchical,
                    colmap,
                    root / "external.json",
                    sequence=SEQUENCE,
                    heldout_protocol_sha256="a" * 64,
                    external_protocol_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
