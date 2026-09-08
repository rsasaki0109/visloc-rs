from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure_native_scope import validate_limits, resident_kib


class ScopeTests(unittest.TestCase):
    def test_limits_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for maximum, swap, valid in [('2147483648', '0', True),
                                          ('max', '0', False),
                                          ('2147483648', 'max', False)]:
                (root / 'memory.max').write_text(maximum)
                (root / 'memory.swap.max').write_text(swap)
                if valid:
                    validate_limits(root)
                else:
                    with self.assertRaises(ValueError):
                        validate_limits(root)

    def test_rss_not_virtual_or_high_water(self):
        self.assertEqual(resident_kib('VmSize: 999 kB\nVmHWM: 500 kB\nVmRSS: 123 kB'), 123)
        self.assertEqual(resident_kib('State: Z'), 0)
