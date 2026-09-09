from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native_source_recipe import INPUTS
from execute_native_source import execute_source
from replay_native_candidates import sha


class SourceExecutionTests(unittest.TestCase):
    def test_success_requires_exact_model_membership_and_contents(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bindings = {value[1:-1]: root for value in INPUTS.values()}
                code = ('import pathlib,sys; p=pathlib.Path(sys.argv[2]); '
                        'p.mkdir(); (p / "images.txt").write_text("model")')
                stage = {'id': 'source', 'binary_sha256': sha(Path(sys.executable).resolve()),
                         'argv': ['{mapper_binary}', '-c', code, '--out-colmap',
                                  '{run_root}/sources/source/model'],
                         'expected_model_hashes': {'images.txt': hashlib.sha256(b'model').hexdigest()},
                         'environment': {}, 'unset_environment': [], 'timeout_seconds': 5}
                if mismatch:
                    stage['expected_model_hashes']['missing.txt'] = 'absent'
                    with self.assertRaises(RuntimeError):
                        execute_source(stage, bindings, sys.executable, root)
                else:
                    self.assertEqual(execute_source(stage, bindings, sys.executable, root)['status'], 'pass')
                report = json.loads((root / 'sources/source/report.json').read_text())
                self.assertEqual(report['exit_code'], 0)
                self.assertEqual(report['status'], 'fail' if mismatch else 'pass')

    def test_failed_process_is_recorded_and_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = {value[1:-1]: root for value in INPUTS.values()}
            stage = {'id': 'source', 'binary_sha256': sha(Path(sys.executable).resolve()),
                     'argv': ['{mapper_binary}', '-c', 'raise SystemExit(7)',
                              '--out-colmap', '{run_root}/sources/source/model'],
                     'expected_model_hashes': {'images.txt': 'not-produced'},
                     'environment': {}, 'unset_environment': [], 'timeout_seconds': 5}
            with self.assertRaises(RuntimeError):
                execute_source(stage, bindings, sys.executable, root)
            report = json.loads((root / 'sources/source/report.json').read_text())
            self.assertEqual(report['status'], 'fail')
            self.assertEqual(report['exit_code'], 7)
            with self.assertRaises(FileExistsError):
                execute_source(stage, bindings, sys.executable, root)
