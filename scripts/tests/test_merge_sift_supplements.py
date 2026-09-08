import importlib.util
from pathlib import Path
import unittest
import tempfile
import subprocess
import sys
import os
import errno
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "merge_sift_supplements", Path(__file__).resolve().parents[1] / "merge_sift_supplements.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeTests(unittest.TestCase):
    def test_hardlink_failure_does_not_fallback_or_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, supplement = root / "base", root / "supplement"
            base.mkdir()
            supplement.mkdir()
            (base / "a_features.txt").write_bytes(b"0 0 base\n")
            with patch.object(MODULE.os, "link", side_effect=OSError(errno.EXDEV, "cross-device")):
                with self.assertRaises(OSError):
                    MODULE.write_bank(base, supplement, {"image_names": []}, root / "out",
                                      hardlink_unselected=True)
            self.assertEqual(list((root / "out").iterdir()), [])

    def test_opt_in_hardlinks_preserve_inventory_and_selected_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, supplement = root / "base", root / "supplement"
            base.mkdir()
            supplement.mkdir()
            (base / "a_features.txt").write_bytes(b"0 0 base\n")
            (base / "b_features.txt").write_bytes(b"1 1 unchanged\n")
            (supplement / "a_features.txt").write_bytes(b"2 2 novel\n")
            selection = {"image_names": ["a.png"]}
            control = MODULE.write_bank(base, supplement, selection, root / "copy")
            linked = MODULE.write_bank(base, supplement, selection, root / "linked",
                                       hardlink_unselected=True)
            self.assertEqual(control["inventory_sha256"], linked["inventory_sha256"])
            self.assertTrue(os.path.samefile(base / "b_features.txt", root / "linked/b_features.txt"))
            self.assertFalse(os.path.samefile(base / "b_features.txt", root / "copy/b_features.txt"))
            self.assertFalse(os.path.samefile(base / "a_features.txt", root / "linked/a_features.txt"))
            resumed = MODULE.write_bank(base, supplement, selection, root / "linked",
                                        resume=True, hardlink_unselected=True)
            self.assertEqual(resumed["reused"], 2)
            self.assertEqual(resumed["inventory_sha256"], control["inventory_sha256"])

    @unittest.skipIf(os.name == "nt", "POSIX SIGKILL recovery test")
    def test_sigkill_staging_does_not_block_bank_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, supplement, output = [root / n for n in ("base", "supplement", "output")]
            for folder in (base, supplement, output):
                folder.mkdir()
            (base / "a_features.txt").write_bytes(b"0 0 base\n")
            code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from merge_sift_supplements import publish_file
def interrupted_chunks():
    yield b'x' * 10000
    print('staged', flush=True)
    sys.stdin.read(1)
publish_file(Path(sys.argv[2]) / 'output' / 'a_features.txt',
             interrupted_chunks(), staging_directory=Path(sys.argv[2]))
"""
            child = subprocess.Popen([sys.executable, "-c", code,
                                      str(Path(__file__).resolve().parents[1]), str(root)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "staged")
                child.kill()
                self.assertEqual(child.wait(timeout=10), -9)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                child.stdin.close()
                child.stdout.close()
            self.assertEqual(list(output.iterdir()), [])
            orphans = list(root.glob(".sift-merge-*"))
            self.assertEqual(len(orphans), 1)
            self.assertGreater(orphans[0].stat().st_size, 0)
            result = MODULE.write_bank(base, supplement, {"image_names": []}, output, True)
            self.assertEqual(result["written"], 1)
            self.assertEqual((output / "a_features.txt").read_bytes(), b"0 0 base\n")
            self.assertEqual(list(root.glob(".sift-merge-*")), orphans)

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
