import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_targeted_rig_candidates import targeted_pairs


class TargetedPairsTest(unittest.TestCase):
    def test_zero_gap_stereo_only(self):
        names, pairs = targeted_pairs({0: ['a', 'b'], 1: ['c', 'd']}, [0], 0)
        self.assertEqual(names, ['a', 'b', 'c', 'd'])
        self.assertEqual(pairs, [(0, 1)])

    def test_overlapping_targets_deduplicated(self):
        _, pairs = targeted_pairs({0: ['a', 'b'], 1: ['c', 'd']}, [0, 1], 1)
        self.assertEqual(pairs, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

    def test_numeric_gap_not_ordinal(self):
        _, pairs = targeted_pairs({0: ['a'], 10: ['b'], 11: ['c']}, [10], 1)
        self.assertEqual(pairs, [(1, 2)])

    def test_invalid_targets(self):
        for targets, gap in [([], 1), ([0, 0], 1), ([2], 1), ([0], -1)]:
            with self.assertRaises(ValueError):
                targeted_pairs({0: ['a', 'b']}, targets, gap)
