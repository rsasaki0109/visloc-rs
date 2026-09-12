from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_openloris_base_extraction import output_disk_budget


class ExtractionDiskBudgetTests(unittest.TestCase):
    def test_round_each_file_not_sum(self):
        self.assertEqual(output_disk_budget([1, 4096, 4097, 0], 4096), 1024**3 + 16384)

    def test_empty_still_reserves_slack(self):
        self.assertEqual(output_disk_budget([], 4096), 1024**3)

    def test_invalid_sizes_and_blocks(self):
        for sizes, block in [([1], 0), ([1], -1), ([-1], 4096)]:
            with self.assertRaises(ValueError):
                output_disk_budget(sizes, block)
