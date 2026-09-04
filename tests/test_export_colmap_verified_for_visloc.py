import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_colmap_verified_for_visloc.py"
SPEC = importlib.util.spec_from_file_location("export_colmap_verified_for_visloc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LoadImageNamesTests(unittest.TestCase):
    def write_manifest(self, payload):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return path

    def test_accepts_compact_image_names_manifest(self):
        path = self.write_manifest({"image_names": ["a.png", "b.png"]})
        self.assertEqual(MODULE.load_image_names(path), ["a.png", "b.png"])

    def test_accepts_official_tier_image_records(self):
        path = self.write_manifest(
            {"schema": "fixture", "images": [{"name": "a.png"}, {"name": "b.png"}]}
        )
        self.assertEqual(MODULE.load_image_names(path), ["a.png", "b.png"])

    def test_rejects_missing_or_duplicate_names(self):
        malformed = self.write_manifest({"images": [{"name": "a.png"}, {}]})
        with self.assertRaisesRegex(ValueError, "images\\[\\]\\.name"):
            MODULE.load_image_names(malformed)
        duplicate = self.write_manifest({"images": [{"name": "a.png"}, {"name": "a.png"}]})
        with self.assertRaisesRegex(ValueError, "repeats image names"):
            MODULE.load_image_names(duplicate)


if __name__ == "__main__":
    unittest.main()
