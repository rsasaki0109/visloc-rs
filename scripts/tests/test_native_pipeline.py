from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_native_pipeline import plan, execute


class PipelineTests(unittest.TestCase):
    def test_generated_dependencies_are_connected(self):
        binaries = {key: key for key in ('candidate_native', 'candidate_native_sha256',
                    'candidate_dense', 'candidate_dense_sha256', 'sfm', 'merge',
                    'compare', 'admission', 'mapper', 'stitch', 'integration')}
        root = Path('/new-run')
        stages = plan(root, binaries, Path('/dataset'))
        ids = [s['id'] for s in stages]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[:4], ['extract-base', 'extract-dense', 'adaptive-selection', 'adaptive-bank'])
        self.assertEqual(ids[-1], 'mapping')
        by_id = {s['id']: s for s in stages}
        for variant in ('native', 'dense'):
            argv = by_id['candidates-' + variant]['argv']
            self.assertEqual(argv[argv.index('--binary') + 1], 'candidate_' + variant)
            self.assertEqual(argv[argv.index('--binary-sha256') + 1], 'candidate_' + variant + '_sha256')
        for variant in ('native', 'adaptive', 'dense', 'targeted7'):
            argv = by_id['match-' + variant]['argv']
            for flag in ('--features-dir', '--candidate-manifest', '--output'):
                self.assertTrue(argv[argv.index(flag) + 1].startswith('/new-run/'))
        for name in ('native-prefix-supplement', 'repair19-admission', 'targeted7-admission'):
            self.assertTrue(all(value.startswith('/new-run/') for key, value in
                                by_id[name]['payload']['data'].items() if key != 'rig_manifest'))
        spec = stages[-1]['payload']['data']
        self.assertEqual(spec['dense_features']['path'], '/new-run/dense/features')
        self.assertEqual(spec['dense_snapshot']['path'], '/new-run/match-dense/run/mapping/verified-merged.vps')

    def test_disk_guard_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            with patch('run_native_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaises(RuntimeError):
                    execute([], root)
            self.assertFalse(root.exists())

    def test_real_stage_failure_stops_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            stage = {'id': 'fail', 'argv': [sys.executable, '-c', 'raise SystemExit(7)'],
                     'payload': None, 'capture': None}
            with patch('run_native_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                with self.assertRaises(RuntimeError):
                    execute([stage, dict(stage, id='never')], root)
            report = json.loads((root / 'pipeline-report.json').read_text())
            self.assertEqual(report['status'], 'fail')
            self.assertEqual(len(report['stages']), 1)
            self.assertFalse((root / 'never.log').exists())
