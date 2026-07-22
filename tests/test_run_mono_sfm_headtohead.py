import os
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_mono_sfm_headtohead import (
    is_unsupported_caspar_failure,
    reported_mean_reprojection,
    run_logged_measured,
)


class ReportedMeanReprojectionTests(unittest.TestCase):
    def test_reads_reconstruction_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mapping.log"
            log.write_text(
                "reconstruction: 260 / 300 images registered, 9062 tracks, "
                "mean reproj 0.604 px [sfm 800.8s]\n",
                encoding="utf-8",
            )
            self.assertEqual(reported_mean_reprojection(log), 0.604)

    def test_returns_none_when_summary_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mapping.log"
            log.write_text("no reconstruction summary\n", encoding="utf-8")
            self.assertIsNone(reported_mean_reprojection(log))

    def test_only_explicit_caspar_backend_rejection_enables_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mapping.log"
            log.write_text(
                "Check failed: ba_global_backend != "
                "BundleAdjustmentBackend::CASPAR\n",
                encoding="utf-8",
            )
            self.assertTrue(is_unsupported_caspar_failure("CASPAR", log))
            self.assertFalse(is_unsupported_caspar_failure("CERES", log))
            log.write_text("unrelated mapper failure\n", encoding="utf-8")
            self.assertFalse(is_unsupported_caspar_failure("CASPAR", log))

    def test_measured_runner_captures_child_working_set_on_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            measured = run_logged_measured(
                [
                    sys.executable,
                    "-c",
                    "import time; x=bytearray(8*1024*1024); "
                    "print(len(x), flush=True); time.sleep(0.4)",
                ],
                log,
                poll_seconds=0.1,
            )
            self.assertGreaterEqual(measured["wall_seconds"], 0.3)
            self.assertIn("8388608", log.read_text(encoding="utf-8"))
            if os.name == "nt":
                self.assertIsNotNone(measured["peak_process_tree_rss_bytes"])
                self.assertGreater(
                    measured["peak_process_tree_rss_bytes"], 8 * 1024 * 1024
                )
                self.assertEqual(measured["resource_poll_seconds"], 0.1)
            else:
                self.assertIsNone(measured["peak_process_tree_rss_bytes"])


if __name__ == "__main__":
    unittest.main()
