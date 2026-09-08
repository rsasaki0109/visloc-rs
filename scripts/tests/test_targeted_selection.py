import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_targeted_selection import unregistered_frames


class TargetSelectionTest(unittest.TestCase):
    frames = {4: ['a', 'b'], 9: ['c', 'd']}
    header = '# retrieval-component-manifest-v1\n'

    def test_missing_frame_derived(self):
        self.assertEqual(unregistered_frames(self.frames, self.header + 'C 0 a\nC 0 b\n'), [9])

    def test_partial_frame_rejected(self):
        with self.assertRaises(ValueError):
            unregistered_frames(self.frames, self.header + 'C 0 a\n')

    def test_unknown_or_duplicate_image_rejected(self):
        for rows in ['C 0 other\n', 'C 0 a\nC 0 a\n']:
            with self.assertRaises(ValueError):
                unregistered_frames(self.frames, self.header + rows)

    def test_schema_required(self):
        with self.assertRaises(ValueError):
            unregistered_frames(self.frames, 'C 0 a\nC 0 b\n')

    def test_no_registration_returns_all_frames(self):
        self.assertEqual(unregistered_frames(self.frames, self.header), [4, 9])
