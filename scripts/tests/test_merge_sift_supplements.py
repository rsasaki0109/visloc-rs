import importlib.util
from pathlib import Path
import unittest
import tempfile

SPEC = importlib.util.spec_from_file_location(
    "merge_sift_supplements", Path(__file__).resolve().parents[1] / "merge_sift_supplements.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeTests(unittest.TestCase):
    def test_bank_resume_verifies_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, supplement, output = [root / n for n in ("base", "supplement", "output")]
            base.mkdir()
            supplement.mkdir()
            (base / "a_features.txt").write_bytes(b"0 0 base\n")
            (base / "b_features.txt").write_bytes(b"1 1 unchanged\n")
            (supplement / "a_features.txt").write_bytes(b"2 2 added\n")
            selection = {"image_names": ["a.png"]}
            first = MODULE.write_bank(base, supplement, selection, output)
            second = MODULE.write_bank(base, supplement, selection, output, True)
            self.assertEqual(first["inventory_sha256"], second["inventory_sha256"])
            self.assertEqual(second["reused"], 2)
            self.assertEqual(second["written"], 0)
            (output / "b_features.txt").unlink()
            partial = MODULE.write_bank(base, supplement, selection, output, True)
            self.assertEqual((partial["reused"], partial["written"]), (1, 1))
            (output / "b_features.txt").write_bytes(b"1 1 Unchanged\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                MODULE.write_bank(base, supplement, selection, output, True)
            self.assertEqual((output / "b_features.txt").read_bytes(), b"1 1 Unchanged\n")

    def test_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "features.txt"
            MODULE.publish_file(destination, [b"first", b"second"])
            self.assertEqual(destination.read_bytes(), b"firstsecond")
            with self.assertRaises(FileExistsError):
                MODULE.publish_file(destination, [b"replacement"])
            self.assertEqual(destination.read_bytes(), b"firstsecond")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_failed_generation_publishes_nothing(self):
        def broken():
            yield b"partial"
            raise ValueError("invalid later input")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                MODULE.publish_file(Path(directory) / "features.txt", broken())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_preserves_prefix_and_supplement_variants(self):
        base = [b"# header\n", b"0 0 base\n"]
        additions = [b"1 0 boundary\n", b"1.01 0 first\n", b"1.01 0 second\n"]
        self.assertEqual(list(MODULE.merged_rows(base, additions)),
                         [base[1], additions[1], additions[2]])

    def test_neighbor_cells_and_negative_coordinates(self):
        self.assertEqual(list(MODULE.merged_rows([b"-0.1 -0.1 base\n"],
                                                [b"0.1 0.1 near\n"])), [b"-0.1 -0.1 base\n"])

    def test_empty_base_keeps_all_supplements(self):
        rows = [b"2 3 a\n", b"2 3 b\n"]
        self.assertEqual(list(MODULE.merged_rows([], rows)), rows)

    def test_bad_coordinates_rejected(self):
        for row in (b"nan 0\n", b"0 inf\n", b"1\n"):
            with self.assertRaises(ValueError):
                list(MODULE.merged_rows([], [row]))


if __name__ == "__main__":
    unittest.main()
