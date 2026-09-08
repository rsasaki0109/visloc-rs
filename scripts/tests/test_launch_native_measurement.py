from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from launch_native_measurement import launch_argv

class LaunchTests(unittest.TestCase):
    def test_detached_limits_and_literal_arguments(self):
        argv = launch_argv('visloc-test', Path('/run dir'), Path('/work'), 10, ['echo', '$literal'])
        self.assertIn('--expand-environment=no', argv)
        self.assertIn('--property=MemoryMax=2G', argv)
        self.assertIn('--property=RuntimeMaxSec=40', argv)
        self.assertNotIn('--scope', argv)
        self.assertNotIn('--wait', argv)
        self.assertEqual(argv[-2:], ['echo', '$literal'])

    def test_invalid_arguments(self):
        for unit, timeout, command in [('bad/name', 1, ['true']), ('visloc-test', 0, ['true']),
                                       ('visloc-test', float('inf'), ['true']), ('visloc-test', 1, [])]:
            with self.assertRaises(ValueError):
                launch_argv(unit, Path('/out'), Path('/work'), timeout, command)
