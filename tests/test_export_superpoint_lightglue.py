from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_superpoint_lightglue import (  # noqa: E402
    export_pair,
    frame_left_features_name,
    frame_right_features_name,
    frame_stereo_matches_name,
    frame_temporal_matches_name,
    selected_frame_pairs,
    sequence_frame_outputs_exist,
)


class ExportSuperPointLightGlueTest(unittest.TestCase):
    def test_selected_frame_pairs_applies_start_stride_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_dir = root / "left"
            right_dir = root / "right"
            left_dir.mkdir()
            right_dir.mkdir()
            for index in range(8):
                (left_dir / f"{index:06}.png").write_text("left", encoding="utf-8")
                (right_dir / f"{index:06}.png").write_text("right", encoding="utf-8")

            args = SimpleNamespace(
                left_dir=left_dir,
                right_dir=right_dir,
                extension=".png",
                start_frame=1,
                frame_stride=2,
                frames=3,
            )

            pairs = selected_frame_pairs(args)

            self.assertEqual([left.name for left, _right in pairs], ["000001.png", "000003.png", "000005.png"])

    def test_sequence_frame_outputs_exist_requires_temporal_after_first_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for name in (
                frame_left_features_name(0),
                frame_right_features_name(0),
                frame_stereo_matches_name(0),
            ):
                (out_dir / name).write_text("ok", encoding="utf-8")
            self.assertTrue(sequence_frame_outputs_exist(out_dir, 0))

            for name in (
                frame_left_features_name(1),
                frame_right_features_name(1),
                frame_stereo_matches_name(1),
            ):
                (out_dir / name).write_text("ok", encoding="utf-8")
            self.assertFalse(sequence_frame_outputs_exist(out_dir, 1))

            (out_dir / frame_temporal_matches_name(1)).write_text("ok", encoding="utf-8")
            self.assertTrue(sequence_frame_outputs_exist(out_dir, 1))

    def test_export_pair_skip_existing_requires_non_empty_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for name in ("image0_features.txt", "image1_features.txt"):
                (out_dir / name).write_text("ok", encoding="utf-8")
            (out_dir / "matches.txt").write_text("", encoding="utf-8")

            args = SimpleNamespace(
                skip_existing=True,
                out_dir=out_dir,
                image0=Path("image0.png"),
                image1=Path("image1.png"),
                resolved_device="cpu",
            )

            calls = []

            class DummyImage:
                def to(self, _device: str) -> "DummyImage":
                    return self

            class DummyExtractor:
                def extract(self, _image: DummyImage) -> dict[str, object]:
                    calls.append("extract")
                    return {}

            class DummyMatcher:
                def __call__(self, _features: dict[str, object]) -> dict[str, object]:
                    calls.append("match")
                    return {}

            def fail_write(*_args: object) -> None:
                raise RuntimeError("write reached")

            with self.assertRaises(RuntimeError):
                export_pair(args, DummyExtractor(), DummyMatcher(), lambda _path: DummyImage(), fail_write)
            self.assertEqual(calls, ["extract", "extract", "match"])


if __name__ == "__main__":
    unittest.main()
