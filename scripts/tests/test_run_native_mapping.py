import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_native_mapping import run, validate_inputs


class MappingTests(unittest.TestCase):
    def test_source_failure_never_runs_atlas(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with patch('run_native_mapping.execute_sources', side_effect=RuntimeError('source fail')), \
                    patch('run_native_mapping.execute_atlas') as atlas:
                with self.assertRaises(RuntimeError):
                    run({}, [], {}, {'mapper_binary': 'fake'}, output, {})
                atlas.assert_not_called()
            self.assertEqual(json.loads((output / 'mapping-report.json').read_text())['status'], 'fail')

    def test_atlas_uses_new_nodes_and_failure_is_reported(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'run'
                nodes = str(output / 'nodes.tsv')
                binaries = dict(mapper_binary='mapper', stitch_binary='stitch', integration_binary='integration')
                with patch('run_native_mapping.execute_sources', return_value={'nodes_tsv': nodes}), \
                        patch('run_native_mapping.execute_atlas', return_value=[],
                              side_effect=RuntimeError('atlas fail') if fail else None) as atlas:
                    if fail:
                        with self.assertRaises(RuntimeError):
                            run({}, [], {'rig_manifest': 'rig'}, binaries, output, {})
                    else:
                        run({}, [], {'rig_manifest': 'rig'}, binaries, output, {})
                    self.assertEqual(atlas.call_args.args[1], {'rig_manifest': 'rig', 'nodes_tsv': nodes})
                self.assertEqual(json.loads((output / 'mapping-report.json').read_text())['status'],
                                 'fail' if fail else 'pass')

    def test_incomplete_inputs_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs({})
