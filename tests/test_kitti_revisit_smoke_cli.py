from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_smoke_cli import (  # noqa: E402
    apply_readme_headline_gate,
    optional_float_env,
    optional_int_env,
    parse_args,
)


class KittiRevisitSmokeCliTest(unittest.TestCase):
    def test_optional_env_helpers_treat_missing_and_empty_as_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(optional_int_env("MISSING"))
            self.assertIsNone(optional_float_env("MISSING"))
        with patch.dict("os.environ", {"EMPTY": ""}, clear=True):
            self.assertIsNone(optional_int_env("EMPTY"))
            self.assertIsNone(optional_float_env("EMPTY"))
        with patch.dict("os.environ", {"INT": "41", "FLOAT": "0.6"}, clear=True):
            self.assertEqual(optional_int_env("INT"), 41)
            self.assertEqual(optional_float_env("FLOAT"), 0.6)

    def test_readme_headline_gate_fills_only_missing_expectations(self) -> None:
        args = Namespace(
            expect_min_candidates=None,
            expect_strongest_from=48,
            expect_strongest_to=None,
            expect_min_strongest_inliers=None,
            expect_min_strongest_ratio=None,
        )

        apply_readme_headline_gate(args)

        self.assertEqual(args.expect_min_candidates, 41)
        self.assertEqual(args.expect_strongest_from, 48)
        self.assertEqual(args.expect_strongest_to, 4501)
        self.assertEqual(args.expect_min_strongest_inliers, 57)
        self.assertEqual(args.expect_min_strongest_ratio, 0.6)

    def test_parse_args_applies_readme_headline_gate(self) -> None:
        argv = [
            "runner.py",
            "--skip-fetch",
            "--start-dir",
            "target/start",
            "--revisit-dir",
            "target/revisit",
            "--out-dir",
            "target/out",
            "--readme-headline-gate",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args("test parser")

        self.assertTrue(args.skip_fetch)
        self.assertEqual(args.start_dir, Path("target/start"))
        self.assertEqual(args.expect_min_candidates, 41)
        self.assertEqual(args.expect_strongest_from, 49)
        self.assertEqual(args.expect_strongest_to, 4501)


if __name__ == "__main__":
    unittest.main()
