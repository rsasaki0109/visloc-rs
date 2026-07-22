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

from run_mast3r_slam_submap_gate import (  # noqa: E402
    PINNED_REVISION,
    evidence,
    parse_pass_metrics,
    path_for_probe,
    scale_gate_passes,
    validate_export_manifest,
    validate_probe_source_revision,
)
from verify_mast3r_slam_submap_gate import (  # noqa: E402
    main as verify_main,
    verify_gate_manifest,
)


class Mast3rSlamSubmapGateTests(unittest.TestCase):
    def make_export_manifest(self, root: Path, *, mode: str = "independent_runs") -> Path:
        descriptor_dir = root / "descriptors"
        descriptor_dir.mkdir()
        descriptor_manifest = descriptor_dir / "manifest.csv"
        descriptor_manifest.write_text("arrival_index,keypoints_file\n", encoding="utf-8")
        frozen = root / "frozen.bin"
        frozen.write_bytes(b"frozen")
        sides = {}
        for side, anchor in (("old", 2), ("new", 5)):
            side_dir = root / side
            side_dir.mkdir()
            log = side_dir / "run.log"
            state = side_dir / "optimized_state.npz"
            points = root / f"{side}_anchor_points.txt"
            log.write_text("completed\n", encoding="utf-8")
            state.write_bytes(side.encode())
            points.write_text("# points\n", encoding="utf-8")
            sides[side] = {
                "anchor_arrival": anchor,
                "local_anchor_index": 1,
                "arrivals": [anchor - 1, anchor, anchor + 1],
                "command": ["probe", side] if mode == "independent_runs" else None,
                "run_log": evidence(log) if mode == "independent_runs" else None,
                "state": str(state.resolve()),
                "state_sha256": evidence(state)["sha256"],
                "anchor_points": str(points.resolve()),
                "anchor_points_sha256": evidence(points)["sha256"],
            }
        manifest = {
            "schema_version": 1,
            "official_revision": PINNED_REVISION,
            "execution_mode": mode,
            "independent_process_per_side": mode == "independent_runs",
            "radius": 1,
            "frozen_inputs": [evidence(frozen)],
            "descriptor_manifest": str(descriptor_manifest.resolve()),
            "descriptor_manifest_sha256": evidence(descriptor_manifest)["sha256"],
            "sides": sides,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def make_gate_manifest(self, root: Path, *, weak: bool = False) -> tuple[Path, dict]:
        export_path = self.make_export_manifest(root)
        export = json.loads(export_path.read_text(encoding="utf-8"))
        probe = root / "probe.exe"
        probe.write_bytes(b"probe")
        log = root / "probe.log"
        old_inliers = 11 if weak else 12
        log.write_text(
            "same_side_scale_transfer_status=pass "
            f"old_matches=20 old_inliers={old_inliers} old_metric_per_local=0.2 "
            "new_matches=25 new_inliers=16 new_metric_per_local=0.25 "
            "new_from_old_scale=0.8\n",
            encoding="utf-8",
        )
        metrics = parse_pass_metrics(log.read_text(encoding="utf-8"))
        source = {
            "build_revision": "a" * 40,
            "source": str((root / "probe.rs").resolve()),
            "source_sha256": "b" * 64,
        }
        old_points = export["sides"]["old"]["anchor_points"]
        new_points = export["sides"]["new"]["anchor_points"]
        gate = {
            "schema_version": 1,
            "status": "success",
            "ground_truth_read": False,
            "backend_writeback": False,
            "export_manifest": evidence(export_path),
            "probe_executable": evidence(probe),
            "probe_source": source,
            "lightglue_directory": str((root / "lightglue").resolve()),
            "frozen_inputs": [evidence(export_path), evidence(probe)],
            "command": [
                str(probe),
                "--dump-dir",
                str(root / "descriptors"),
                "--lightglue-dir",
                str(root / "lightglue"),
                "--learned-old-points",
                old_points,
                "--learned-new-points",
                new_points,
                "--old-anchor",
                "2",
                "--new-anchor",
                "5",
                "--radius",
                "1",
            ],
            "returncode": 0,
            "probe_log": evidence(log),
            "changed_inputs": [],
            "same_side_scale_transfer_gate": {
                "passed": scale_gate_passes(metrics),
                "minimum_inlier_ratio": 0.60,
                "metrics": metrics,
                "reason": None,
            },
        }
        path = root / "gate_manifest.json"
        path.write_text(json.dumps(gate), encoding="utf-8")
        return path, source

    def test_accepts_complete_independent_export(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = self.make_export_manifest(Path(raw_root))
            export, descriptor_dir, old_points, new_points = validate_export_manifest(path)
            self.assertEqual(export["execution_mode"], "independent_runs")
            self.assertEqual(descriptor_dir.name, "descriptors")
            self.assertEqual(old_points.name, "old_anchor_points.txt")
            self.assertEqual(new_points.name, "new_anchor_points.txt")

    def test_rejects_extract_only_export(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = self.make_export_manifest(Path(raw_root), mode="extract_only")
            with self.assertRaisesRegex(ValueError, "independent submap runs"):
                validate_export_manifest(path)

    def test_rejects_mutated_frozen_export_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            path = self.make_export_manifest(root)
            (root / "frozen.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_export_manifest(path)

    def test_pass_metrics_require_full_frozen_consensus(self) -> None:
        metrics = parse_pass_metrics(
            "same_side_scale_transfer_status=pass "
            "old_matches=20 old_inliers=12 old_metric_per_local=0.2 "
            "new_matches=25 new_inliers=16 new_metric_per_local=0.25 "
            "new_from_old_scale=0.8\n"
        )
        self.assertIsNotNone(metrics)
        self.assertTrue(scale_gate_passes(metrics))

        weak = parse_pass_metrics(
            "same_side_scale_transfer_status=pass "
            "old_matches=20 old_inliers=11 old_metric_per_local=0.2 "
            "new_matches=25 new_inliers=16 new_metric_per_local=0.25 "
            "new_from_old_scale=0.8\n"
        )
        self.assertIsNotNone(weak)
        self.assertFalse(scale_gate_passes(weak))

    def test_rejects_duplicate_or_nonfinite_pass_lines(self) -> None:
        line = (
            "same_side_scale_transfer_status=pass "
            "old_matches=20 old_inliers=12 old_metric_per_local=nan "
            "new_matches=25 new_inliers=16 new_metric_per_local=0.25 "
            "new_from_old_scale=0.8\n"
        )
        self.assertFalse(scale_gate_passes(parse_pass_metrics(line)))
        self.assertIsNone(parse_pass_metrics(line + line))

        impossible = parse_pass_metrics(
            "same_side_scale_transfer_status=pass "
            "old_matches=20 old_inliers=21 old_metric_per_local=0.2 "
            "new_matches=25 new_inliers=16 new_metric_per_local=0.25 "
            "new_from_old_scale=0.8\n"
        )
        self.assertFalse(scale_gate_passes(impossible))

        inconsistent = parse_pass_metrics(
            "same_side_scale_transfer_status=pass "
            "old_matches=20 old_inliers=12 old_metric_per_local=0.2 "
            "new_matches=25 new_inliers=16 new_metric_per_local=0.25 "
            "new_from_old_scale=1.2\n"
        )
        self.assertFalse(scale_gate_passes(inconsistent))

    def test_probe_source_revision_requires_exact_committed_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            source = Path(raw_root) / "probe.rs"
            source.write_bytes(b"line one\r\nline two\r\n")
            revision = "a" * 40
            with patch("run_mast3r_slam_submap_gate.PROBE_SOURCE", source), patch(
                "run_mast3r_slam_submap_gate.subprocess.check_output",
                return_value=b"line one\nline two\n",
            ):
                result = validate_probe_source_revision(revision)
            self.assertEqual(result["build_revision"], revision)

            with self.assertRaisesRegex(ValueError, "full lowercase"):
                validate_probe_source_revision("abc")

    def test_wsl_converts_only_windows_probe_arguments(self) -> None:
        source = Path("/mnt/e/probe/points.txt")
        with patch("run_mast3r_slam_submap_gate.os.name", "posix"), patch(
            "run_mast3r_slam_submap_gate.subprocess.check_output",
            return_value="E:\\probe\\points.txt\n",
        ) as convert:
            rendered = path_for_probe(source, Path("probe.exe"))
        self.assertEqual(rendered, "E:\\probe\\points.txt")
        self.assertEqual(convert.call_args.args[0][0], "wslpath")

    def test_release_verifier_recomputes_positive_and_negative_gates(self) -> None:
        for weak, expected in ((False, True), (True, False)):
            with self.subTest(weak=weak), tempfile.TemporaryDirectory() as raw_root:
                path, source = self.make_gate_manifest(Path(raw_root), weak=weak)
                with patch(
                    "verify_mast3r_slam_submap_gate.validate_probe_source_revision",
                    return_value=source,
                ):
                    audit = verify_gate_manifest(path)
                self.assertIs(audit["gate_passed"], expected)

    def test_release_verifier_rejects_edited_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path, source = self.make_gate_manifest(Path(raw_root))
            gate = json.loads(path.read_text(encoding="utf-8"))
            gate["same_side_scale_transfer_gate"]["metrics"]["old_inliers"] = 19
            path.write_text(json.dumps(gate), encoding="utf-8")
            with patch(
                "verify_mast3r_slam_submap_gate.validate_probe_source_revision",
                return_value=source,
            ), self.assertRaisesRegex(ValueError, "metrics differ"):
                verify_gate_manifest(path)

    def test_release_verifier_rejects_mutated_log_or_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            path, source = self.make_gate_manifest(root)
            (root / "probe.log").write_text("changed\n", encoding="utf-8")
            with patch(
                "verify_mast3r_slam_submap_gate.validate_probe_source_revision",
                return_value=source,
            ), self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_gate_manifest(path)

        with tempfile.TemporaryDirectory() as raw_root:
            path, source = self.make_gate_manifest(Path(raw_root))
            gate = json.loads(path.read_text(encoding="utf-8"))
            anchor = gate["command"].index("--old-anchor") + 1
            gate["command"][anchor] = "3"
            path.write_text(json.dumps(gate), encoding="utf-8")
            with patch(
                "verify_mast3r_slam_submap_gate.validate_probe_source_revision",
                return_value=source,
            ), self.assertRaisesRegex(ValueError, "old-anchor differs"):
                verify_gate_manifest(path)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            path, source = self.make_gate_manifest(root)
            gate = json.loads(path.read_text(encoding="utf-8"))
            points = gate["command"].index("--learned-old-points") + 1
            gate["command"][points] = str(root / "other" / "old_anchor_points.txt")
            path.write_text(json.dumps(gate), encoding="utf-8")
            with patch(
                "verify_mast3r_slam_submap_gate.validate_probe_source_revision",
                return_value=source,
            ), self.assertRaisesRegex(ValueError, "old-points differs"):
                verify_gate_manifest(path)

    def test_release_verifier_emits_self_bound_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            path, source = self.make_gate_manifest(root)
            output = root / "release_audit.json"
            with patch(
                "verify_mast3r_slam_submap_gate.validate_probe_source_revision",
                return_value=source,
            ), patch(
                "sys.argv",
                [
                    "verify_mast3r_slam_submap_gate.py",
                    "--gate-manifest",
                    str(path),
                    "--out",
                    str(output),
                ],
            ):
                self.assertEqual(verify_main(), 0)
            audit = json.loads(output.read_text(encoding="utf-8"))
            verifier = Path(audit["verifier"]["path"])
            self.assertEqual(audit["status"], "verified")
            self.assertEqual(
                audit["verifier"]["sha256"],
                hashlib.sha256(verifier.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
