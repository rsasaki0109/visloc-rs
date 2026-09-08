from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_targeted_selection import mapper_command


class RegistrationCommandTests(unittest.TestCase):
    def test_prefix_matches_recorded_recipe(self):
        command, threads = mapper_command('prefix', 'mapper', 'rig', 'bank', Path('new output'), 'snapshot')
        self.assertEqual(threads, 1)
        self.assertEqual(command, ['mapper', '--manifest', 'rig', '--features-dir', 'bank',
                                  '--snapshot', 'snapshot', '--out-colmap', 'new output/model',
                                  '--max-models', '10', '--pair-confidence-tracks',
                                  '--final-ba-min-pose-observations', '32'])

    def test_targeted_adds_only_deferred_prefix_and_threads(self):
        args = ('mapper', 'rig', 'bank', Path('output'), 'snapshot')
        prefix, _ = mapper_command('prefix', *args)
        targeted, threads = mapper_command('targeted', *args)
        self.assertEqual(threads, 8)
        self.assertEqual(targeted, prefix + ['--deferred-registration-pair-prefix', '59961'])

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            mapper_command('typo', 'mapper', 'rig', 'bank', Path('output'), 'snapshot')
