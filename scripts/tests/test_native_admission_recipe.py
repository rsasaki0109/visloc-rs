from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_native_admission_recipe import build


class AdmissionRecipeTests(unittest.TestCase):
    def test_admissions_bind_produced_dependencies(self):
        recipe = build(Path(__file__).resolve().parents[2] / 'benchmarks/electro')
        self.assertEqual(len(recipe['stages']), 3)
        prefix, repair, targeted = recipe['stages']
        self.assertIn('--include-prefix-supplements', prefix['argv'])
        self.assertEqual(prefix['input_bindings']['--base-features-dir'], '{base_features}')
        self.assertEqual(repair['input_bindings']['--repair-registered-components'], '{prefix_registration_components}')
        self.assertEqual(targeted['input_bindings']['--base-snapshot'], '{repair_snapshot}')
        self.assertIn('--include-all-new-pairs', targeted['argv'])
        for stage in recipe['stages']:
            self.assertFalse(any(arg.startswith('/') for arg in stage['argv']))
            self.assertEqual(len(stage['expected_snapshot_sha256']), 64)
