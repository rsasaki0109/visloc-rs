from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_native_candidates import replace_path, bind_feature_input
from benchmark_electro import feature_manifest, write_feature_manifest, ValidationError


class CandidateBindingTests(unittest.TestCase):
    def test_real_manifest_accepts_relocated_bank_rejects_corruption_and_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root / 'new-bank'
            bank.mkdir()
            feature = bank / 'a_features.txt'
            feature.write_text('0 0 1 0\n')
            manifest = root / 'features.json'
            write_feature_manifest(manifest, feature_manifest(bank))
            original = ['sfm', '--features-dir', '/unused-retained-bank']
            command, actual = bind_feature_input(original, manifest, bank)
            self.assertEqual(actual, bank.resolve())
            self.assertEqual(command[2], str(bank.resolve()))
            extra = bank / 'b_features.txt'
            extra.write_text('0 0 1 0\n')
            with self.assertRaisesRegex(ValueError, 'membership'):
                bind_feature_input(original, manifest, bank)
            extra.unlink()
            feature.write_text('1 0 1 0\n')
            with self.assertRaises(ValidationError):
                bind_feature_input(original, manifest, bank)

    def test_replace_preserves_other_arguments_and_original(self):
        source = ['sfm', '--features-dir', '/old', '--topk', '128']
        result = replace_path(source, '--features-dir', Path('/new bank'))
        self.assertEqual(result, ['sfm', '--features-dir', '/new bank', '--topk', '128'])
        self.assertEqual(source[2], '/old')

    def test_ambiguous_or_missing_flag_is_rejected(self):
        for args in [[], ['--features-dir'], ['--features-dir', 'a', '--features-dir', 'b']]:
            with self.assertRaises(ValueError):
                replace_path(args, '--features-dir', Path('/new'))
