import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'audit', Path(__file__).resolve().parents[1] / 'audit_snapshot_envelopes.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture(images=2):
    payload = struct.pack('<IQ', 1, images)
    for index in range(images):
        name = f'{index}.png'.encode()
        payload += struct.pack('<Q', len(name)) + name
    payload += struct.pack('<QQQ', 0, 0, images) + bytes(images * 8 + 48)
    payload += bytes(32)  # Two empty configuration strings with hashes.
    payload += bytes(32)  # Pair/edge hashes, accepted count, pair count.
    return audit.MAGIC + struct.pack('<IQ', 1, len(payload)) + payload + bytes(8)


class EnvelopeTests(unittest.TestCase):
    def inspect(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.vps'
            path.write_bytes(data)
            return audit.inspect(path)

    def test_sizes(self):
        row = self.inspect(fixture())
        self.assertEqual(row['images'], 2)
        self.assertEqual(row['pairs'], 0)
        self.assertEqual(row['image_dependent_bytes'], 42)
        self.assertEqual(row['shared_envelope_bytes'], 158)
        self.assertEqual(row['file_bytes'], len(fixture()))

    def test_image_metadata_grows_linearly_per_shard(self):
        first = self.inspect(fixture(2))
        second = self.inspect(fixture(4))
        self.assertEqual(second['image_dependent_bytes'], 2 * first['image_dependent_bytes'])
        self.assertNotEqual(first['envelope_sha256'], second['envelope_sha256'])

    def test_rejects_bad_header_and_lengths(self):
        for data in [b'', fixture()[:-1], fixture() + b'x', b'X' + fixture()[1:]]:
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                self.inspect(data)


if __name__ == '__main__':
    unittest.main()
