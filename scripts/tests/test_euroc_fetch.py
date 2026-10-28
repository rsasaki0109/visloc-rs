from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import functools
import io
import os
import re
import sys
import tempfile
import threading
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import euroc_fetch
import http_zip

SEQ = "MH_02_easy"


class FamilyHandler(BaseHTTPRequestHandler):
    """Serves family zips as the Research Collection does: HEAD without a Content-Length,
    single ranges, and every `limit_every`-th request refused with 429 and Retry-After."""

    ranges = True
    limit_every = 3
    counter = None  # [count, lock], per server

    def __init__(self, *args, directory, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)

    def limited(self):
        if not self.limit_every:
            return False
        with self.counter[1]:
            self.counter[0] += 1
            refuse = self.counter[0] % self.limit_every == 0
        if refuse:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
        return refuse

    def path_on_disk(self):
        return os.path.join(self.directory, self.path.strip("/").removesuffix("/content"))

    def do_HEAD(self):
        if not self.limited():
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        if self.limited():
            return
        path = self.path_on_disk()
        size = os.path.getsize(path)
        spec = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        with open(path, "rb") as f:
            if self.ranges and spec:
                lo, hi = int(spec[1]), min(int(spec[2]), size - 1)
                f.seek(lo)
                data = f.read(hi - lo + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
            else:
                data = f.read()
                self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except ConnectionError:  # the size probe hangs up on a whole-file answer
            pass

    def log_message(self, *args):
        pass


def sequence_zip(seq):
    """A sequence zip as EuRoC ships it, and the files fetch() should keep from it."""
    files = {}
    for cam in ("cam0", "cam1"):
        for i in range(12):
            files[f"{seq}/mav0/{cam}/data/{i:05d}.png"] = os.urandom(3000 + 101 * i)
        files[f"{seq}/mav0/{cam}/data.csv"] = b"#timestamp [ns],filename\n"
        files[f"{seq}/mav0/{cam}/sensor.yaml"] = b"sensor_type: camera\n"
    files[f"{seq}/mav0/imu0/data.csv"] = b"#timestamp [ns],w_RS_S_x\n"
    files[f"{seq}/mav0/state_groundtruth_estimate0/data.csv"] = b"#timestamp\n"
    files[f"{seq}/mav0/body.yaml"] = b"comment: body\n"
    files[f"{seq}/mav0/leica0/data.csv"] = b"not kept\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    kept = {n.split("/", 1)[1]: d for n, d in files.items() if "/leica0/" not in n}
    return buf.getvalue(), files, kept


class EurocFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        www = self.root / "www"
        www.mkdir()
        inner, files, self.expected = sequence_zip(SEQ)
        other, _, _ = sequence_zip("MH_01_easy")
        for name, compress in (("stored", zipfile.ZIP_STORED), ("compressed", zipfile.ZIP_DEFLATED)):
            with zipfile.ZipFile(www / f"{name}.zip", "w") as z:
                z.writestr(zipfile.ZipInfo("machine_hall/MH_01_easy/MH_01_easy.zip"), other, compress)
                z.writestr(zipfile.ZipInfo(f"machine_hall/{SEQ}/{SEQ}.zip"), inner, compress)
        with zipfile.ZipFile(www / "flat.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(f"machine_hall/{name}", data)
            z.writestr("machine_hall/MH_01_easy/mav0/cam0/data/00000.png", b"other sequence")
        self.www = www
        self.old_chunk = http_zip.CHUNK
        http_zip.CHUNK = 8192  # many range requests on small archives
        self.old_url = euroc_fetch.family_url
        self.servers = []

    def tearDown(self):
        http_zip.CHUNK = self.old_chunk
        euroc_fetch.family_url = self.old_url
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.tmp.cleanup()

    def serve(self, family, ranges=True, limit_every=3):
        handler_class = type("Handler", (FamilyHandler,), {
            "ranges": ranges, "limit_every": limit_every, "counter": [0, threading.Lock()]})
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(handler_class, directory=str(self.www)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers.append(server)
        euroc_fetch.family_url = lambda seq: f"http://127.0.0.1:{server.server_port}/{family}.zip/content"

    def fetched(self):
        out = euroc_fetch.fetch(SEQ, self.root / "out", workers=2, log=lambda m: None)
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def test_stored_sequence_zip_is_read_by_range_through_429s(self):
        self.serve("stored")
        self.assertEqual(self.fetched(), self.expected)

    def test_compressed_sequence_zip_is_streamed_and_deleted(self):
        self.serve("compressed")
        self.assertEqual(self.fetched(), self.expected)

    def test_flat_family_keeps_only_the_sequence(self):
        self.serve("flat")
        self.assertEqual(self.fetched(), self.expected)

    def test_without_ranges_the_family_is_downloaded_whole(self):
        self.serve("stored", ranges=False, limit_every=0)
        self.assertEqual(self.fetched(), self.expected)
        self.assertTrue((self.root / "out" / "_family").is_dir())

    def test_rerun_skips_members_on_disk(self):
        self.serve("stored")
        self.fetched()
        first = self.root / "out" / SEQ / "mav0/cam0/data/00000.png"
        first.write_bytes(b"x" * first.stat().st_size)  # same size: kept as it is
        self.assertEqual(self.fetched()["mav0/cam0/data/00000.png"], first.read_bytes())

    def test_unknown_sequence_is_refused(self):
        with self.assertRaises(SystemExit):
            euroc_fetch.fetch("MH_09_unknown", self.root / "out")


if __name__ == "__main__":
    unittest.main()
