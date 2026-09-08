from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from replay_native_matching import same_candidate_schedule, snapshot_inventory


class ReplayScheduleTest(unittest.TestCase):
    def test_feature_binding_may_differ(self):
        self.assertTrue(same_candidate_schedule(
            'images 2\nfeature_manifest_sha256 aaa\nshard 0 a b c\n',
            'images 2\nfeature_manifest_sha256 bbb\nshard 0 a b c\n'))

    def test_changed_candidate_rejected(self):
        self.assertFalse(same_candidate_schedule('shard 0 a b c', 'shard 0 a b d'))

    def test_changed_order_rejected(self):
        self.assertFalse(same_candidate_schedule('image 0 a\nimage 1 b', 'image 1 b\nimage 0 a'))

    def test_inventory_sorted_and_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').write_bytes(b'a')
            (root / 'b').write_bytes(b'b')
            before = snapshot_inventory(root, ['a', 'b'])
            self.assertEqual(before, snapshot_inventory(root, ['b', 'a']))
            (root / 'b').write_bytes(b'c')
            self.assertNotEqual(before, snapshot_inventory(root, ['a', 'b']))

    def test_inventory_missing_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                snapshot_inventory(Path(directory), ['missing'])
