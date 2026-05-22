from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from kitti_revisit_smoke_runner import (  # noqa: E402
    cargo_demo_command,
    config_from_args,
    fetch_command,
    render_asset_command,
    smoke_summary_text,
    write_smoke_summary,
)


def args_namespace() -> Namespace:
    return Namespace(
        start_dir=Path("target/start"),
        revisit_dir=Path("target/revisit"),
        out_dir=Path("target/out"),
        start_frames=50,
        revisit_start_frame=4500,
        revisit_frames=30,
        workers=8,
        frontend="deep",
        max_features=200,
        min_matches=30,
        min_inliers=12,
        min_inlier_ratio=0.4,
        max_mean_sampson_error=0.005,
        cargo_profile="release",
        readme_asset_out=Path("docs/assets/kitti.jpg"),
        readme_headline_gate=True,
        expect_min_candidates=41,
        expect_strongest_from=49,
        expect_strongest_to=4501,
        expect_min_strongest_inliers=57,
        expect_min_strongest_ratio=0.6,
    )


class KittiRevisitSmokeRunnerTest(unittest.TestCase):
    def test_config_from_args_maps_expectations(self) -> None:
        config = config_from_args(args_namespace())

        self.assertEqual(config.start_dir, Path("target/start"))
        self.assertEqual(config.expectations.min_candidates, 41)
        self.assertEqual(config.expectations.strongest_from, 49)
        self.assertEqual(config.expectations.min_strongest_ratio, 0.6)

    def test_cargo_demo_command_contains_demo_paths_and_thresholds(self) -> None:
        cmd = cargo_demo_command(config_from_args(args_namespace()))

        self.assertEqual(cmd[:3], ["cargo", "run", "--release"])
        self.assertIn("kitti_revisit_scanner_demo", cmd)
        self.assertIn("target/start/image_0", {part.replace("\\", "/") for part in cmd})
        self.assertIn("target/revisit/calib.txt", {part.replace("\\", "/") for part in cmd})
        self.assertIn("--max-mean-sampson-error", cmd)
        self.assertIn("0.005", cmd)

    def test_dev_cargo_command_omits_release_flag(self) -> None:
        args = args_namespace()
        args.cargo_profile = "dev"

        cmd = cargo_demo_command(config_from_args(args))

        self.assertEqual(cmd[:2], ["cargo", "run"])
        self.assertNotIn("--release", cmd)

    def test_fetch_command_adds_start_frame_only_for_revisit_slice(self) -> None:
        fetch_script = Path("scripts/fetch.py")

        start_cmd = fetch_command(fetch_script, out_dir=Path("start"), max_frames=50, workers=8)
        revisit_cmd = fetch_command(
            fetch_script,
            out_dir=Path("revisit"),
            max_frames=30,
            workers=8,
            start_frame=4500,
        )

        self.assertNotIn("--start-frame", start_cmd)
        self.assertIn("--start-frame", revisit_cmd)
        self.assertIn("4500", revisit_cmd)

    def test_render_asset_command_requires_output_path(self) -> None:
        config = config_from_args(args_namespace())
        cmd = render_asset_command(Path("scripts/render.py"), config)

        self.assertEqual(cmd[-2], "--out")
        self.assertEqual(cmd[-1].replace("\\", "/"), "docs/assets/kitti.jpg")

        args = args_namespace()
        args.readme_asset_out = None
        with self.assertRaisesRegex(ValueError, "readme_asset_out is required"):
            render_asset_command(Path("scripts/render.py"), config_from_args(args))

    def test_smoke_summary_text_includes_expectations(self) -> None:
        summary = smoke_summary_text(config_from_args(args_namespace()), "strongest_from=49\n")

        self.assertIn("readme_headline_gate=true", summary)
        self.assertIn("expect_min_candidates=41", summary)
        self.assertIn("expect_strongest_to=4501", summary)
        self.assertTrue(summary.endswith("strongest_from=49\n"))

    def test_write_smoke_summary_writes_named_file(self) -> None:
        out_dir = REPO_ROOT / "target" / "kitti_revisit_smoke_runner_unit"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.txt"
        if summary_path.exists():
            summary_path.unlink()
        args = args_namespace()
        args.out_dir = out_dir
        config = config_from_args(args)

        path = write_smoke_summary(config, "strongest_from=49\n", filename="summary.txt")

        self.assertEqual(path, summary_path)
        self.assertIn("strongest_from=49", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
