from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_openloris_base_extraction import partition_images

class PartitionTests(unittest.TestCase):
    def test_exact_disjoint_balanced_10k(self):
        images = [Path(f'{i}.png') for i in range(10000)]
        groups = partition_images(images, 6)
        flattened = [p for group in groups for p in group]
        self.assertEqual(len(flattened), 10000)
        self.assertEqual(set(flattened), set(images))
        self.assertLessEqual(max(map(len, groups)) - min(map(len, groups)), 1)

    def test_invalid_workers_and_collisions(self):
        for workers in (0, 7, 2):
            with self.assertRaises(ValueError):
                partition_images([Path('a.png')], workers)
        with self.assertRaises(ValueError):
            partition_images([Path('a.png'), Path('a.jpg')], 1)
