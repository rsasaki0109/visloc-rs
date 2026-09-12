from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deduplicate_atlas_parity import merge_file
from replay_native_candidates import sha


class DeduplicationTests(unittest.TestCase):
    def test_preserves_paths_and_bytes_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ('source', 'target'))
            source.write_bytes(b'original model')
            target.write_bytes(source.read_bytes())
            digest = sha(source)
            merge_file(source, target, digest)
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
            self.assertEqual(sha(target), digest)
            self.assertEqual(merge_file(source, target, digest), 0)

    def test_mismatch_or_symlink_does_not_replace_target(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ('source', 'target'))
            source.write_bytes(b'original model')
            target.write_bytes(b'different model')
            with self.assertRaises(ValueError):
                merge_file(source, target, sha(source))
            self.assertEqual(target.read_bytes(), b'different model')
            target.unlink()
            target.symlink_to(source)
            with self.assertRaises(ValueError):
                merge_file(source, target, sha(source))
            self.assertTrue(target.is_symlink())


if __name__ == '__main__':
    unittest.main()
