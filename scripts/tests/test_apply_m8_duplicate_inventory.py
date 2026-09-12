from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apply_m8_duplicate_inventory import ROOT, safe_path


class ScopeTests(unittest.TestCase):
    def test_model_and_match_paths(self):
        for name in ('images.txt', 'points3D.txt', 'verified.vps'):
            relative = 'corridor1-1-m8-example/model/' + name
            self.assertEqual(safe_path(relative), ROOT / relative)

    def test_rejects_escape_and_unrelated_artifacts(self):
        for relative in ('/tmp/images.txt', '../images.txt',
                         'corridor1-1-m8-example/../images.txt',
                         'corridor1-1-m5/images.txt', 'corridor1-1-m8-example',
                         'corridor1-1-m8-example/run.log',
                         'corridor1-1-m8-example/raw.png',
                         'corridor1-1-m8-example/cam1_features.txt'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_path(relative)


if __name__ == '__main__':
    unittest.main()
