from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_native_candidates import replace_path


class CandidateBindingTests(unittest.TestCase):
    def test_replace_preserves_other_arguments_and_original(self):
        source = ['sfm', '--features-dir', '/old', '--topk', '128']
        result = replace_path(source, '--features-dir', Path('/new bank'))
        self.assertEqual(result, ['sfm', '--features-dir', '/new bank', '--topk', '128'])
        self.assertEqual(source[2], '/old')

    def test_ambiguous_or_missing_flag_is_rejected(self):
        for args in [[], ['--features-dir'], ['--features-dir', 'a', '--features-dir', 'b']]:
            with self.assertRaises(ValueError):
                replace_path(args, '--features-dir', Path('/new'))
