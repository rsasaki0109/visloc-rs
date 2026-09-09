from pathlib import Path
import json
import fcntl
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_native_pipeline import execute
from native_pipeline_resume import validate_resume


class ResumeTests(unittest.TestCase):
    def test_live_executor_lock_prevents_resume_before_report_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'pipeline.lock').open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('run_native_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                    with self.assertRaises(BlockingIOError):
                        execute([], root, resume=True)
            self.assertFalse((root / 'pipeline-report.json').exists())

    def test_real_failure_is_archived_and_completed_child_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            marker = Path(directory) / 'fail'
            marker.touch()
            first = {'id': 'first', 'argv': [sys.executable, '-c',
                     'import pathlib,sys; p=pathlib.Path(sys.argv[-1]); p.mkdir(); (p/"data").write_text("done")',
                     '--output', str(root / 'first')], 'payload': None, 'capture': None}
            second = {'id': 'second', 'argv': [sys.executable, '-c',
                      'import pathlib,sys; p=pathlib.Path(sys.argv[-1]); p.mkdir(); (p/"data").write_text("attempt"); '
                      f'sys.exit(7 if pathlib.Path({str(marker)!r}).exists() else 0)',
                      '--output', str(root / 'second')], 'payload': None, 'capture': None}
            with patch('run_native_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                with self.assertRaises(RuntimeError):
                    execute([first, second], root)
                first_inode = (root / 'first/data').stat().st_ino
                marker.unlink()
                report = execute([first, second], root, resume=True)
            self.assertEqual(report['status'], 'pass')
            self.assertEqual((root / 'first/data').stat().st_ino, first_inode)
            archive = Path(report['resumed_attempts'][0])
            self.assertEqual((archive / 'artifacts/second/data').read_text(), 'attempt')
            self.assertTrue((archive / 'artifacts/second.log').is_file())
            prior = json.loads((archive / 'pipeline-report.json').read_text())
            self.assertEqual(prior['stages'][-1]['exit_code'], 7)

    def test_unobserved_failure_and_changed_plan_are_rejected(self):
        stage = {'id': 'failed', 'argv': [], 'capture': None, 'payload': None}
        report = {'status': 'fail', 'plan': [stage], 'dependency_sha256': {},
                  'stages': [{'id': 'failed', 'exit_code': 7}], 'active_stage': 'failed'}
        with self.assertRaises(ValueError):
            validate_resume(report, [stage], {})
        report['failure_kind'] = 'child-exit'
        self.assertEqual(validate_resume(report, [stage], {}), 0)
        with self.assertRaises(ValueError):
            validate_resume(report, [dict(stage, id='different')], {})
        report['stages'][0]['exit_code'] = -9
        with self.assertRaises(ValueError):
            validate_resume(report, [stage], {})


if __name__ == '__main__':
    unittest.main()
