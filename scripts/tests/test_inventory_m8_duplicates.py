from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inventory_m8_duplicates import inventory


class InventoryTests(unittest.TestCase):
    def test_scope_shared_inode_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / 'corridor1-1-m8-a'
            b = root / 'corridor1-1-m8-b'
            other = root / 'other'
            for path in (a, b, other):
                path.mkdir()
            data = b'x' * (1024 * 1024)
            source = a / 'images.txt'
            source.write_bytes(data)
            target = b / 'images.txt'
            target.write_bytes(data)
            (other / 'images.txt').write_bytes(data)
            (b / 'run.log').write_bytes(data)
            (b / 'points3D.txt').symlink_to(source)
            (a / 'same.vps').hardlink_to(source)
            before = target.stat()
            result = inventory(root)
            self.assertEqual(len(result['groups']), 1)
            self.assertEqual(len(result['groups'][0]['files']), 2)
            self.assertEqual(result['potential_freed_bytes'], before.st_blocks * 512)
            self.assertEqual(target.stat().st_ino, before.st_ino)
            self.assertEqual(target.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
