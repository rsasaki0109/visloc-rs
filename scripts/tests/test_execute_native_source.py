from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native_source_recipe import INPUTS, build
from execute_native_source import execute_source, execute_sources
from replay_native_candidates import sha


class SourceExecutionTests(unittest.TestCase):
    def test_source_phase_publication_failure_leaves_no_partial_manifest(self):
        recipe = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
        with tempfile.TemporaryDirectory() as directory:
            with patch('execute_native_source.execute_source', return_value={'status': 'pass'}), \
                    patch('execute_native_source.os.link', side_effect=OSError('publication failed')):
                with self.assertRaises(OSError):
                    execute_sources(recipe, {}, sys.executable, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_source_phase_does_not_publish_nodes_after_failure(self):
        recipe = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
        with tempfile.TemporaryDirectory() as directory:
            with patch('execute_native_source.execute_source', side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):
                    execute_sources(recipe, {}, sys.executable, directory)
            self.assertFalse((Path(directory) / 'nodes.tsv').exists())

    def test_source_phase_calls_every_stage_before_publication(self):
        recipe = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
        with tempfile.TemporaryDirectory() as directory:
            def stage_pass(*args):
                self.assertFalse((Path(directory) / 'nodes.tsv').exists())
                return {'status': 'pass'}
            with patch('execute_native_source.execute_source', side_effect=stage_pass) as execute:
                result = execute_sources(recipe, {}, sys.executable, directory)
            self.assertEqual(execute.call_count, 21)
            self.assertEqual(result['status'], 'pass')
            self.assertEqual(len((Path(directory) / 'nodes.tsv').read_text().splitlines()), 24)

    def test_success_requires_exact_model_membership_and_contents(self):
        for case in ('exact', 'missing', 'changed', 'auxiliary', 'unknown'):
            mismatch = case in ('missing', 'changed', 'unknown')
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bindings = {value[1:-1]: root for value in INPUTS.values()}
                code = ('import pathlib,sys; p=pathlib.Path(sys.argv[2]); '
                        'p.mkdir(); (p / "images.txt").write_text("model")')
                if case == 'auxiliary':
                    code += '; (p / "components.tsv").write_text("components"); (p / "retrieval-components.txt").write_text("retrieval")'
                if case == 'unknown':
                    code += '; (p / "unexpected.txt").write_text("unknown")'
                if case == 'changed':
                    code += '; (p / "images.txt").write_text("changed")'
                stage = {'id': 'source', 'binary_sha256': sha(Path(sys.executable).resolve()),
                         'argv': ['{mapper_binary}', '-c', code, '--out-colmap',
                                  '{run_root}/sources/source/model'],
                         'expected_model_hashes': {'images.txt': hashlib.sha256(b'model').hexdigest()},
                         'environment': {}, 'unset_environment': [], 'timeout_seconds': 5}
                if case == 'missing':
                    stage['expected_model_hashes']['missing.txt'] = 'absent'
                if mismatch:
                    with self.assertRaises(RuntimeError):
                        execute_source(stage, bindings, sys.executable, root)
                else:
                    self.assertEqual(execute_source(stage, bindings, sys.executable, root)['status'], 'pass')
                report = json.loads((root / 'sources/source/report.json').read_text())
                self.assertEqual(report['exit_code'], 0)
                self.assertEqual(report['status'], 'fail' if mismatch else 'pass')
                if case == 'auxiliary':
                    self.assertEqual(set(report['model_hashes']),
                                     {'images.txt', 'components.tsv', 'retrieval-components.txt'})

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
