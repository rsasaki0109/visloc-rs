from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from execute_native_atlas import execute_atlas
from native_atlas_recipe import build
from replay_native_candidates import sha


class AtlasExecutionTests(unittest.TestCase):
    def test_real_child_failure_and_success(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = root / 'input'
                fixture.write_text('fixture')
                binary = Path(sys.executable).resolve()
                stages = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
                for stage in stages:
                    stage['binary_sha256'] = sha(binary)
                    stage['expected_files'] = {'result.txt': sha(fixture)}
                    # Exercise timeout, environment, logging and reports with a
                    # real child; this is deliberately not an SfM parity test.
                    code = ('import pathlib,sys,os; '
                            'assert os.environ["MALLOC_ARENA_MAX"] == "1"; '
                            'p=pathlib.Path(sys.argv[sys.argv.index("--out-dir")+1]); '
                            + ('p=p.parent; ' if stage['id'] != 'stitch' else '')
                            + 'p.joinpath("result.txt").write_text("fixture"); '
                            + f'sys.exit({exit_code})')
                    stage['argv'][1:1] = ['-c', code]
                args = (stages, dict(rig_manifest=fixture, nodes_tsv=fixture),
                        dict(stitch_binary=binary, integration_binary=binary), root / 'run')
                if exit_code:
                    with self.assertRaises(RuntimeError):
                        execute_atlas(*args)
                    self.assertFalse((root / 'run/integrated').exists())
                else:
                    self.assertEqual(len(execute_atlas(*args)), 3)
                report = json.loads((root / 'run/atlas/execution-report.json').read_text())
                self.assertEqual(report['exit_code'], exit_code)
                self.assertEqual(report['status'], 'fail' if exit_code else 'pass')

    def test_success_and_failed_stage_stop(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = root / 'input'
                fixture.write_text('fixture')
                stages = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
                for stage in stages:
                    stage['binary_sha256'] = sha(fixture)
                    stage['expected_files'] = {key: sha(fixture) for key in stage['expected_files']}
                calls = []

                def run(command, **kwargs):
                    stage = stages[len(calls)]
                    calls.append(command)
                    output = Path(command[command.index('--out-dir') + 1])
                    if stage['id'] != 'stitch':
                        output = output.parent
                    for name in stage['expected_files']:
                        path = output / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('wrong' if fail else 'fixture')
                    return SimpleNamespace(returncode=0)

                bindings = dict(rig_manifest=fixture, nodes_tsv=fixture)
                binaries = dict(stitch_binary=fixture, integration_binary=fixture)
                with patch('execute_native_atlas.subprocess.run', side_effect=run):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            execute_atlas(stages, bindings, binaries, root / 'run')
                        self.assertEqual(len(calls), 1)
                        self.assertFalse((root / 'run/integrated').exists())
                    else:
                        reports = execute_atlas(stages, bindings, binaries, root / 'run')
                        self.assertEqual([r['status'] for r in reports], ['pass'] * 3)
                        with self.assertRaises(FileExistsError):
                            execute_atlas(stages, bindings, binaries, root / 'run')
                        self.assertEqual(len(calls), 3)

    def test_binary_mismatch_precedes_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'input'
            fixture.write_text('fixture')
            stages = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
            with self.assertRaisesRegex(ValueError, 'binary hash'):
                execute_atlas(stages, dict(rig_manifest=fixture, nodes_tsv=fixture),
                              dict(stitch_binary=fixture, integration_binary=fixture), root / 'run')
            self.assertFalse((root / 'run').exists())
