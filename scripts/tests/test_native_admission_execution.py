from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_native_admission import render


class AdmissionExecutionTests(unittest.TestCase):
    def test_explicit_paths_remain_single_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new snapshot.vps'
            path.touch()
            stage = {'input_bindings': {'--base-snapshot': '{native_snapshot}'},
                     'argv': ['{admission_binary}', '--base-snapshot', '{native_snapshot}',
                              '--output', '{run_root}/admissions/test.vps']}
            self.assertEqual(render(stage, {'native_snapshot': str(path)}, 'binary', Path('output')),
                             ['binary', '--base-snapshot', str(path), '--output', 'output/admissions/test.vps'])
            for bindings in [{}, {'native_snapshot': str(path), 'run_root': 'override'}]:
                with self.assertRaises(ValueError):
                    render(stage, bindings, 'binary', Path('output'))

    def test_missing_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = {'input_bindings': {'--base-snapshot': '{native_snapshot}'}, 'argv': []}
            with self.assertRaises(FileNotFoundError):
                render(stage, {'native_snapshot': str(Path(directory) / 'missing')}, 'binary', Path('output'))
