"""Unit tests for scripts/lamaria_vrs_fetch.py on a small synthetic VRS file: the index, the records kept, the copy."""
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_vrs_fetch as fetch  # noqa: E402

CAM, IMU, GPS = (1201, 1), (1202, 1), (281, 1)


def records():
    """(record type, stream, size) in file order: configuration and state first, then interleaved data."""
    out = [(fetch.CONFIGURATION, s, 40) for s in (CAM, IMU, GPS)] + [(fetch.STATE, s, 20) for s in (CAM, IMU, GPS)]
    for k in range(6):
        out += [(fetch.DATA, CAM, 5000), (fetch.DATA, IMU, 300), (fetch.DATA, GPS, 60)]
    return out


def vrs_file(path):
    """A VRS file of the records above, with its zstd index in the head and random record bytes."""
    recs = records()
    disk = b"".join(struct.pack(fetch.DISK_RECORD, float(i), size, rt, sid[0], sid[1])
                    for i, (rt, sid, size) in enumerate(recs))
    packed = subprocess.run(["zstd", "-q", "-c"], input=disk, capture_output=True, check=True).stdout
    streams = [CAM, IMU, GPS]
    body = struct.pack("<I", len(streams)) + b"".join(struct.pack("<iH", *s) for s in streams)
    body += struct.pack("<I", len(recs)) + packed
    index_off = struct.calcsize(fetch.FILE_HEADER)
    index = struct.pack(fetch.RECORD_HEADER, 32 + len(body), 0, 0, 1, 0.0, 0, 0, 2, len(disk)) + body
    first_user = index_off + len(index) + 100  # a description record after the index
    magic = struct.unpack("<II", b"VisionRe")
    header = struct.pack(fetch.FILE_HEADER, *magic, 7, index_off, 32, index_off, index_off + len(index),
                         first_user, 0, 0, 0, 0, 0)
    rng = random.Random(0)
    data = bytes(rng.getrandbits(8) for _ in range(sum(size for *_, size in recs)))
    blob = header + index + bytes(100) + data
    Path(path).write_bytes(blob)
    return blob, first_user


@unittest.skipUnless(shutil.which("zstd"), "needs the zstd command")
class FetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.src = Path(cls.tmp.name) / "full.vrs"
        cls.blob, cls.first_user = vrs_file(cls.src)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_index_gives_every_record_at_its_offset(self):
        first_user, recs = fetch.parse_index(self.blob[:self.first_user])
        self.assertEqual(first_user, self.first_user)
        self.assertEqual([(rt, sid, size) for _, size, rt, sid in recs], records())
        self.assertEqual(recs[0][0], self.first_user)
        self.assertEqual(recs[-1][0] + recs[-1][1], len(self.blob))

    def test_kept_are_the_configurations_the_ends_of_each_stream_and_the_gps(self):
        _, recs = fetch.parse_index(self.blob[:self.first_user])
        want, kept = fetch.keep(recs, {GPS})
        self.assertEqual(kept, {CAM: 4, IMU: 4, GPS: 8})
        cam = [(o, s) for o, s, rt, sid in recs if sid == CAM and rt == fetch.DATA]
        self.assertIn(cam[0], want)
        self.assertIn(cam[-1], want)
        self.assertNotIn(cam[1], want)

    def test_ranges_close_together_are_fetched_as_one(self):
        self.assertEqual(fetch.merge([(0, 10), (12, 5), (100, 1)], gap=2), [[0, 16], [100, 100]])
        self.assertEqual(fetch.merge([(0, 10), (12, 5), (100, 1)], gap=1), [[0, 9], [12, 16], [100, 100]])

    def test_the_sparse_copy_holds_the_kept_records_only(self):
        out = Path(self.tmp.name) / "sparse.vrs"
        subprocess.run([sys.executable, str(SCRIPTS / "lamaria_vrs_fetch.py"), str(self.src), str(out),
                        "--streams", "281-1", "--merge-gap", "0"], capture_output=True, check=True)
        copy = out.read_bytes()
        self.assertEqual(len(copy), len(self.blob))
        self.assertEqual(copy[:self.first_user], self.blob[:self.first_user])
        _, recs = fetch.parse_index(self.blob[:self.first_user])
        want, _ = fetch.keep(recs, {GPS})
        for o, s, _, _ in recs:
            expected = self.blob[o:o + s] if (o, s) in want else bytes(s)
            self.assertEqual(copy[o:o + s], expected)


if __name__ == "__main__":
    unittest.main()
