import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_gluemap_ssfm_wsl.py"
)
SPEC = importlib.util.spec_from_file_location("run_gluemap_ssfm_wsl", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GluemapWslWrapperTests(unittest.TestCase):
    def test_verbose_time_peak_rss_is_converted_from_kibibytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "time.txt"
            path.write_text(
                "Command being timed: test\nMaximum resident set size (kbytes): 12345\n",
                encoding="utf-8",
            )
            self.assertEqual(MODULE.parse_peak_rss_bytes(path), 12345 * 1024)

    def test_wsl_path_requires_absolute_linux_result(self) -> None:
        with patch("subprocess.run", return_value=Mock(stdout="/mnt/e/input\n")) as run:
            rendered = MODULE.wsl_path(Path("input"), "Ubuntu-22.04")
            self.assertEqual(rendered, "/mnt/e/input")
            self.assertIn("wslpath", run.call_args.args[0])
        with patch("subprocess.run", return_value=Mock(stdout="relative\n")):
            with self.assertRaisesRegex(ValueError, "invalid path"):
                MODULE.wsl_path(Path("input"), "Ubuntu-22.04")


if __name__ == "__main__":
    unittest.main()
