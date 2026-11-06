"""Unit tests for scripts/lamaria_xs_keyframes.py: the keyframes picked, and members read back from a multi-range
response of a synthetic zip."""
import io
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_xs_keyframes as kf  # noqa: E402


def synthetic_zip():
    """A zip of four members, deflated and stored, with their bytes."""
    rng = np.random.default_rng(0)
    data = {f"seq/mav0/cam{c}/data/{t}.png": bytes(rng.integers(0, 4, 3000 + 100 * t, dtype=np.uint8))
            for c in (0, 1) for t in (1, 2)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, (name, b) in enumerate(data.items()):
            z.writestr(name, b, zipfile.ZIP_DEFLATED if k % 2 == 0 else zipfile.ZIP_STORED)
    raw = buf.getvalue()
    return raw, zipfile.ZipFile(io.BytesIO(raw)).infolist(), data


def multipart(raw, ranges, boundary="xyzzy"):
    """A multipart/byteranges body of the spans, as a server sends it."""
    out = b""
    for s, e in ranges:
        e = min(e, len(raw) - 1)
        out += (f"--{boundary}\r\nContent-Type: application/zip\r\nContent-Range: bytes {s}-{e}/{len(raw)}\r\n\r\n"
                .encode() + raw[s:e + 1] + b"\r\n")
    return out + f"--{boundary}--\r\n".encode(), f"multipart/byteranges; boundary={boundary}"


class KeyframesTest(unittest.TestCase):
    def test_keyframes_need_both_cameras_and_a_pose(self):
        members = [{t: None for t in range(0, 100)}, {t: None for t in range(1, 100)}]
        graph, matched = kf.keyframes(members, np.arange(0, 90), kf_step=10, img_step=20)
        np.testing.assert_array_equal(graph, np.arange(1, 90, 10))
        np.testing.assert_array_equal(matched, np.arange(1, 90, 20))

    def test_members_from_a_multipart_response(self):
        raw, infos, data = synthetic_zip()
        ranges = kf.byte_ranges(infos)
        body, ctype = multipart(raw, ranges)
        parts = kf.parse_multipart(body, ctype, ranges[0][0])
        self.assertEqual(sorted(parts), sorted(s for s, _ in ranges))
        for info in infos:
            self.assertEqual(kf.member_bytes(info, parts), data[info.filename])

    def test_a_single_range_response(self):
        raw, infos, data = synthetic_zip()
        s, e = kf.byte_ranges(infos[:1])[0]
        parts = kf.parse_multipart(raw[s:e + 1], "application/zip", s)
        self.assertEqual(kf.member_bytes(infos[0], parts), data[infos[0].filename])

    def test_a_corrupt_member_is_refused(self):
        raw, infos, _ = synthetic_zip()
        info = infos[1]  # stored: a flipped data byte reaches the CRC check
        bad = bytearray(raw)
        bad[info.header_offset + 30 + len(info.filename.encode()) + 10] ^= 0xFF
        s, e = kf.byte_ranges([info])[0]
        parts = kf.parse_multipart(bytes(bad[s:e + 1]), "application/zip", s)
        with self.assertRaises(AssertionError):
            kf.member_bytes(info, parts)


if __name__ == "__main__":
    unittest.main()
