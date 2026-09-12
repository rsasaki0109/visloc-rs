from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_atlas_recipe import build, bind, timed_argv


class AtlasRecipeTests(unittest.TestCase):
    def test_reject_unbound_or_missing_path(self):
        for command in (['tool', '--input'], ['tool', '/retained/model'],
                        ['tool', '--input', 'a', '--input', 'b']):
            with self.subTest(command=command), self.assertRaises(ValueError):
                bind(command, {'--input': '{input}'})

    def test_reject_changed_timeout(self):
        with self.assertRaises(ValueError):
            timed_argv({'time': 'Command being timed: "timeout --signal=TERM --kill-after=10s 999s tool"'}, 180)

    def test_all_three_stages_use_new_artifact_paths(self):
        stages = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
        self.assertEqual([s['id'] for s in stages], ['stitch', 'integrate-tail', 'integrate-main'])
        for stage in stages:
            self.assertFalse(any(arg.startswith('/') for arg in stage['argv']))
            self.assertIn('{nodes_tsv}', stage['argv'])
        self.assertIn('{run_root}/atlas/component-001', stages[1]['argv'])
        self.assertIn('{run_root}/atlas/component-000', stages[2]['argv'])
        self.assertEqual(sum(len(s.get('expected_files', {})) for s in stages), 14)
