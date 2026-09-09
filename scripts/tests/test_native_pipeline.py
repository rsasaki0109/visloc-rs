from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_native_pipeline import plan, execute, pinned_files, verify_pins, artifact_lifetimes


class PipelineTests(unittest.TestCase):
    def test_code_evidence_and_binary_pins_detect_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'benchmarks/electro').mkdir(parents=True)
            files = [root / 'scripts/imported.py', root / 'benchmarks/electro/input.json', root / 'binary']
            for path in files:
                path.write_text('original')
            pins = pinned_files(root, {'tool': {'path': str(files[-1])}})
            self.assertEqual(len(pins), 3)
            verify_pins(pins)
            for path in files:
                path.write_text('changed')
                with self.assertRaises(RuntimeError):
                    verify_pins(pins)
                path.write_text('original')

    def test_changed_dependency_prevents_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency = root / 'dependency'
            dependency.write_text('original')
            from replay_native_candidates import sha
            pins = {str(dependency): sha(dependency)}
            code = 'from pathlib import Path; Path(' + repr(str(dependency)) + ').write_text("changed")'
            stage = {'id': 'mutate', 'argv': [sys.executable, '-c', code], 'payload': None, 'capture': None}
            with patch('run_native_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, 'dependency changed'):
                    execute([stage, dict(stage, id='never')], root / 'run', pins)
            report = json.loads((root / 'run/pipeline-report.json').read_text())
            self.assertEqual(report['status'], 'fail')
            self.assertEqual(len(report['stages']), 1)
            self.assertEqual(report['stages'][0]['exit_code'], 0)

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
        lifetimes = artifact_lifetimes(stages)
        self.assertEqual(lifetimes['/new-run/base']['last_consumer'], 'native-prefix-supplement')
        self.assertEqual(lifetimes['/new-run/dense']['last_consumer'], 'mapping')
        self.assertEqual(lifetimes['/new-run/adaptive']['last_consumer'], 'mapping')
        self.assertEqual(lifetimes['/new-run/match-native']['last_consumer'], 'native-prefix-supplement')
        self.assertEqual(lifetimes['/new-run/match-adaptive']['last_consumer'], 'repair19-admission')
        self.assertEqual(lifetimes['/new-run/match-targeted7']['last_consumer'], 'targeted7-admission')

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
