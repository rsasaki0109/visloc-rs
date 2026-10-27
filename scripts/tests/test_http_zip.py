from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import functools
import os
import sys
import tempfile
import threading
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import http_zip


class RangeHandler(SimpleHTTPRequestHandler):
    """The static file server with single `bytes=a-b` ranges, as the LaMAria server answers them."""

    def send_head(self):
        spec = self.headers.get("Range")
        if not spec:
            return super().send_head()
        path = self.translate_path(self.path)
        size = os.path.getsize(path)
        lo, hi = (int(v) for v in spec.removeprefix("bytes=").split("-"))
        hi = min(hi, size - 1)
        f = open(path, "rb")
        f.seek(lo)
        self.send_response(206)
        self.send_header("Content-Length", str(hi - lo + 1))
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.end_headers()
        self.range_left = hi - lo + 1
        return f

    def copyfile(self, source, outputfile):
        if getattr(self, "range_left", None) is None:
            return super().copyfile(source, outputfile)
        outputfile.write(source.read(self.range_left))

    def log_message(self, *args):
        pass


class HttpZipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "www").mkdir()
        self.files = {f"seq/aria/cam0/data/{i:04d}.png": os.urandom(1000 + 37 * i) for i in range(40)}
        self.files["seq/aria/cam0/data.csv"] = b"#t,name\n"
        with zipfile.ZipFile(root / "www" / "seq.zip", "w") as z:
            for i, (name, data) in enumerate(self.files.items()):
                z.writestr(name, data, zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED)
        handler = functools.partial(RangeHandler, directory=str(root / "www"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/seq.zip"
        self.dest = root / "out"
        self.old_chunk = http_zip.CHUNK
        http_zip.CHUNK = 4096  # many range requests on a small archive

    def tearDown(self):
        http_zip.CHUNK = self.old_chunk
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_extracts_every_member_in_parallel_parts(self):
        http_zip.extract(self.url, self.dest, workers=3, log=lambda m: None)
        for name, data in self.files.items():
            self.assertEqual((self.dest / name).read_bytes(), data, name)
        self.assertEqual(list(self.dest.rglob("*.part")), [])

    def test_rerun_skips_members_on_disk(self):
        http_zip.extract(self.url, self.dest, workers=2, log=lambda m: None)
        first = self.dest / "seq/aria/cam0/data/0000.png"
        first.write_bytes(b"x" * first.stat().st_size)  # same size: kept as it is
        http_zip.extract(self.url, self.dest, workers=2, log=lambda m: None)
        self.assertEqual(first.read_bytes(), b"x" * first.stat().st_size)


if __name__ == "__main__":
    unittest.main()
