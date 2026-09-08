import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native_source_recipe import build, compile_execution, INPUTS


class SourceRecipeTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[2] / 'benchmarks/electro'

    def test_all_sources_and_nodes_bind_without_retained_paths(self):
        recipe = build(self.root)
        self.assertEqual(len(recipe['executions']), 21)
        self.assertEqual(len(recipe['nodes']), 23)
        for row in recipe['executions']:
            self.assertFalse(any(arg.startswith('/') for arg in row['argv']))
            for flag, value in INPUTS.items():
                self.assertEqual(row['argv'][row['argv'].index(flag) + 1], value)
        debug = [r['id'] for r in recipe['executions'] if 'VISLOC_DEFERRED_DEBUG' in r['environment']]
        self.assertEqual(debug, ['source-replay-4200'])

    def test_unknown_input_path_is_rejected(self):
        original = self.root / 'm8-openloris-source-spec-0-v1.json'
        evidence = json.loads(original.read_text())
        evidence['audit']['time'] = evidence['audit']['time'].replace('--manifest ', '--unknown-input ')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / original.name
            path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, 'path flag'):
                compile_execution(path)


if __name__ == '__main__':
    unittest.main()
