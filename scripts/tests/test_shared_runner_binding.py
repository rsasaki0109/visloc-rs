from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_full_shared_runner import bind_candidates


class SharedRunnerBindingTests(unittest.TestCase):
    def test_regenerated_bytes_and_default(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'reference.txt'
            regenerated = Path(directory) / 'regenerated.txt'
            reference.write_bytes(b'a.png b.png\n')
            regenerated.write_bytes(reference.read_bytes())
            default, digest = bind_candidates(reference)
            selected, actual = bind_candidates(reference, regenerated)
            self.assertEqual(default, reference.resolve())
            self.assertEqual(selected, regenerated.resolve())
            self.assertEqual(digest, actual)
            regenerated.write_bytes(b'a.png c.png\n')
            with self.assertRaisesRegex(ValueError, 'differs'):
                bind_candidates(reference, regenerated)

    def test_missing_regenerated_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'reference.txt'
            reference.write_bytes(b'a b\n')
            with self.assertRaises(FileNotFoundError):
                bind_candidates(reference, Path(directory) / 'missing')
