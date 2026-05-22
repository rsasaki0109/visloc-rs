from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_smoke_io import (  # noqa: E402
    expand_run_paths,
    read_summary,
    remove_output_dir,
    require_path,
    run_command,
)


class KittiRevisitSmokeIoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.unit_dir = REPO_ROOT / "target" / "kitti_revisit_smoke_io_unit"
        if self.unit_dir.exists():
            shutil.rmtree(self.unit_dir)
        self.unit_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.unit_dir.exists():
            shutil.rmtree(self.unit_dir)

    def test_expand_run_paths_normalizes_optional_readme_asset_path(self) -> None:
        args = Namespace(
            start_dir=Path("~") / "kitti-start",
            revisit_dir=Path("target/revisit"),
            out_dir=Path("target/out"),
            readme_asset_out=Path("~") / "asset.jpg",
        )

        expand_run_paths(args)

        self.assertEqual(args.start_dir, Path.home() / "kitti-start")
        self.assertEqual(args.readme_asset_out, Path.home() / "asset.jpg")
        self.assertEqual(args.revisit_dir, Path("target/revisit"))

    def test_require_path_accepts_files_and_directories(self) -> None:
        file_path = self.unit_dir / "summary.txt"
        file_path.write_text("ok", encoding="utf-8")

        require_path(self.unit_dir, "unit directory")
        require_path(file_path, "unit file")

        with self.assertRaisesRegex(FileNotFoundError, "missing missing file"):
            require_path(self.unit_dir / "missing.txt", "missing file")

    def test_remove_output_dir_refuses_repo_root_and_filesystem_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe output directory"):
            remove_output_dir(REPO_ROOT, REPO_ROOT)
        with self.assertRaisesRegex(ValueError, "unsafe output directory"):
            remove_output_dir(Path(REPO_ROOT.anchor), REPO_ROOT)

    def test_remove_output_dir_recreates_safe_directory(self) -> None:
        nested = self.unit_dir / "out"
        nested.mkdir()
        stale = nested / "stale.txt"
        stale.write_text("stale", encoding="utf-8")

        remove_output_dir(nested, REPO_ROOT)

        self.assertTrue(nested.is_dir())
        self.assertFalse(stale.exists())

    def test_read_summary_reads_utf8_text(self) -> None:
        summary = self.unit_dir / "summary.txt"
        summary.write_text("strongest_from=49\n", encoding="utf-8")

        self.assertEqual(read_summary(summary), "strongest_from=49\n")

    def test_run_command_propagates_subprocess_errors(self) -> None:
        run_command([sys.executable, "-c", "pass"], REPO_ROOT)

        with self.assertRaises(subprocess.CalledProcessError):
            run_command([sys.executable, "-c", "raise SystemExit(7)"], REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
