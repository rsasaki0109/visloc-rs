import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_superpoint_lightglue", ROOT / "scripts" / "export_superpoint_lightglue.py"
)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def args_for(directory: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "left_dir": directory / "left",
        "right_dir": directory / "right",
        "mono_dir": None,
        "extension": ".png",
        "start_frame": 0,
        "frames": None,
        "frame_stride": 1,
        "start_index": None,
        "end_index": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_feature(path: Path, rows: int = 2, dimension: int = 2) -> None:
    def write(stream: object) -> None:
        stream.write("# X Y SCORE D0 D1 ...\n")  # type: ignore[attr-defined]
        for row in range(rows):
            descriptor = " ".join("0.0" for _ in range(dimension))
            stream.write(f"{row}.0 {row}.0 1.0 {descriptor}\n")  # type: ignore[attr-defined]

    EXPORTER.atomic_text_write(path, write)


def write_match(path: Path, query: int = 0, train: int = 0) -> None:
    def write(stream: object) -> None:
        stream.write("# QUERY_IDX TRAIN_IDX CONFIDENCE DISTANCE\n")  # type: ignore[attr-defined]
        stream.write(f"{query} {train} 0.9 0.1\n")  # type: ignore[attr-defined]

    EXPORTER.atomic_text_write(path, write)


class ExporterSafetyTests(unittest.TestCase):
    def test_atomic_write_replaces_and_cleans_failed_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "nested" / "result.txt"
            output.parent.mkdir()
            output.write_text("old", encoding="utf-8")

            EXPORTER.atomic_text_write(output, lambda stream: stream.write("new"))
            self.assertEqual(output.read_text(encoding="utf-8"), "new")

            def fail(stream: object) -> None:
                stream.write("partial")  # type: ignore[attr-defined]
                raise RuntimeError("injected writer failure")

            with self.assertRaises(RuntimeError):
                EXPORTER.atomic_text_write(output, fail)
            self.assertEqual(output.read_text(encoding="utf-8"), "new")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_explicit_source_range_preserves_indices_and_legacy_slice_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for side in ("left", "right"):
                side_directory = directory / side
                side_directory.mkdir()
                for index in range(6):
                    (side_directory / f"frame_{index:06}.png").touch()

            ranged = EXPORTER.selected_frame_pairs(
                args_for(directory, start_index=2, end_index=5)
            )
            self.assertEqual([entry[0] for entry in ranged], [2, 3, 4])
            self.assertEqual(ranged[0][1].name, "frame_000002.png")

            legacy = EXPORTER.selected_frame_pairs(
                args_for(directory, start_frame=2, frames=2)
            )
            self.assertEqual([entry[0] for entry in legacy], [0, 1])
            self.assertEqual(legacy[0][1].name, "frame_000002.png")

    def test_parser_rejects_ambiguous_or_reversed_ranges(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "export_superpoint_lightglue.py",
                "--left-dir",
                "/tmp/left",
                "--right-dir",
                "/tmp/right",
                "--out-dir",
                "/tmp/out",
                "--start-index",
                "4",
                "--end-index",
                "4",
            ],
        ):
            with self.assertRaises(SystemExit):
                EXPORTER.parse_args()

    def test_structural_validation_rejects_nonempty_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_feature(directory / "left.txt")
            write_feature(directory / "right.txt")
            write_match(directory / "matches.txt")
            self.assertEqual(EXPORTER._feature_file_metadata(directory / "left.txt"), (2, 2))
            self.assertEqual(EXPORTER._match_file_rows(directory / "matches.txt", 2, 2), 1)

            partial = directory / "partial.txt"
            partial.write_text("# X Y SCORE D0 D1 ...\n1 2\n", encoding="utf-8")
            self.assertIsNone(EXPORTER._feature_file_metadata(partial))

            (directory / "frame_000000_left_features.txt").write_text(
                "# X Y SCORE D0 D1 ...\n1 2\n", encoding="utf-8"
            )
            self.assertIsNone(
                EXPORTER.validate_sequence_frame_outputs(directory, 0, None)
            )

    def test_manifest_validates_rows_hashes_and_atomic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            left = directory / "left"
            right = directory / "right"
            left.mkdir()
            right.mkdir()
            entries = []
            for index in range(3):
                left_image = left / f"frame_{index:06}.png"
                right_image = right / f"frame_{index:06}.png"
                left_image.touch()
                right_image.touch()
                write_feature(directory / EXPORTER.frame_left_features_name(index))
                write_feature(directory / EXPORTER.frame_right_features_name(index))
                write_match(directory / EXPORTER.frame_stereo_matches_name(index))
                if index > 0:
                    write_match(directory / EXPORTER.frame_temporal_matches_name(index))
                entries.append((index, left_image, right_image))

            manifest_path = directory / "manifest.json"
            manifest = EXPORTER.build_sequence_manifest(directory, entries)
            EXPORTER._write_manifest(manifest_path, manifest)
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["frame_count"], 3)
            self.assertEqual(parsed["frames"][1]["files"]["temporal_matches"]["rows"], 1)
            self.assertEqual(len(parsed["manifest_sha256"]), 64)
            self.assertEqual(list(directory.glob(".manifest.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
