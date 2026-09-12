import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("export_bounded_aliked_lightglue.py")
SPEC = importlib.util.spec_from_file_location("bounded_aliked", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CandidateManifestTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix="visloc-bounded-aliked-test-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root))
        path = root / "candidates.txt"
        path.write_text(text)
        return path

    def test_parses_metadata_bound_pairs(self):
        path = self.write(
            "visloc_candidate_manifest_v1\n"
            "images 3\n"
            "image 0 a.png\nimage 1 b.png\nimage 2 c.png\n"
            "metadata source_sha256 " + "a" * 64 + "\n"
            "pairs 2\npair 0 2\npair 1 2\n"
        )
        self.assertEqual(
            MODULE.parse_candidates(path),
            (["a.png", "b.png", "c.png"], [(0, 2), (1, 2)]),
        )

    def test_rejects_repeated_pair(self):
        path = self.write(
            "visloc_candidate_manifest_v1\n"
            "images 2\nimage 0 a.png\nimage 1 b.png\n"
            "pairs 2\npair 0 1\npair 0 1\n"
        )
        with self.assertRaisesRegex(ValueError, "repeated"):
            MODULE.parse_candidates(path)


if __name__ == "__main__":
    unittest.main()
