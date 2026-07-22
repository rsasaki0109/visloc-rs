from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_mast3r_slam_submap_anchor_points import (  # noqa: E402
    PINNED_REVISION,
    changed_file_evidence,
    file_evidence,
    prepare_output_directory,
    validate_patched_source,
)


class Mast3rSlamSubmapExporterTests(unittest.TestCase):
    def make_source(self, root: Path) -> None:
        (root / "mast3r_slam").mkdir(parents=True)
        (root / "main.py").write_text("patched main\n", encoding="utf-8")
        (root / "mast3r_slam" / "evaluate.py").write_text(
            "patched evaluate\n", encoding="utf-8"
        )

    def test_accepts_only_exact_frozen_patch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.make_source(root)
            patch_path = root / "export.patch"
            patch_bytes = b"diff --git a/main.py b/main.py\n+frozen\n"
            patch_path.write_bytes(patch_bytes)
            with patch(
                "export_mast3r_slam_submap_anchor_points.git_revision",
                return_value=PINNED_REVISION,
            ), patch(
                "export_mast3r_slam_submap_anchor_points.subprocess.check_output",
                side_effect=[
                    " M main.py\n M mast3r_slam/evaluate.py\n",
                    patch_bytes,
                ],
            ):
                evidence = validate_patched_source(root, patch_path)

            self.assertEqual(evidence["official_revision"], PINNED_REVISION)
            self.assertEqual(
                evidence["working_tree_diff_sha256"],
                hashlib.sha256(patch_bytes).hexdigest(),
            )

    def test_rejects_additional_working_tree_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.make_source(root)
            patch_path = root / "export.patch"
            patch_path.write_bytes(b"patch\n")
            with patch(
                "export_mast3r_slam_submap_anchor_points.git_revision",
                return_value=PINNED_REVISION,
            ), patch(
                "export_mast3r_slam_submap_anchor_points.subprocess.check_output",
                return_value=" M main.py\n M mast3r_slam/evaluate.py\n?? extra.py\n",
            ):
                with self.assertRaisesRegex(RuntimeError, "only the frozen export patch"):
                    validate_patched_source(root, patch_path)

    def test_rejects_modified_patch_content(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.make_source(root)
            patch_path = root / "export.patch"
            patch_path.write_bytes(b"expected\n")
            with patch(
                "export_mast3r_slam_submap_anchor_points.git_revision",
                return_value=PINNED_REVISION,
            ), patch(
                "export_mast3r_slam_submap_anchor_points.subprocess.check_output",
                side_effect=[
                    " M main.py\n M mast3r_slam/evaluate.py\n",
                    b"different\n",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "not the frozen export patch"):
                    validate_patched_source(root, patch_path)

    def test_normal_run_requires_fresh_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            output = Path(raw_root) / "new" / "probe"
            prepare_output_directory(output, extract_only=False)
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output, extract_only=False)

    def test_extract_only_refuses_to_overwrite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            output = Path(raw_root)
            (output / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "manifest.json"):
                prepare_output_directory(output, extract_only=True)

    def test_detects_frozen_input_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "input.bin"
            path.write_bytes(b"before")
            evidence = [file_evidence(path)]
            self.assertEqual(changed_file_evidence(evidence), [])
            path.write_bytes(b"after")
            self.assertEqual(changed_file_evidence(evidence), [str(path.resolve())])


if __name__ == "__main__":
    unittest.main()
