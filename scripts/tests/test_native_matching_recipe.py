from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_matching_recipe import runner_match_flags

class MatchingRecipeTests(unittest.TestCase):
    def command(self, minimum):
        return ['sfm', '--feature-extractor', 'files', '--features-dir', 'features',
                '--input-colmap-calibration', 'calibration', '--persistent-match-worker-plan', 'plan',
                '--out-colmap', 'unused', '--verification-mode', 'full', '--matcher', 'nn',
                '--mapper', 'incremental', '--feature-suffix', '_features.txt',
                '--image-suffix', '.png', '--min-matches', str(minimum), '--match-ratio', '0.8']

    def test_preserve_variant_thresholds(self):
        for value in (12, 30):
            flags = runner_match_flags(self.command(value))
            self.assertEqual(flags[flags.index('--min-matches') + 1], str(value))

    def test_unknown_duplicate_and_nonfinite_fail_closed(self):
        for suffix in [['--future-policy', '1'], ['--min-matches', '12']]:
            with self.assertRaises(ValueError):
                runner_match_flags(self.command(12) + suffix)
        command = self.command(12)
        command[-1] = 'nan'
        with self.assertRaises(ValueError):
            runner_match_flags(command)
