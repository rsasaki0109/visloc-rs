from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import functools
import io
import os
import sys
import tarfile
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tumvi_fetch


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class TumviFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "www").mkdir()
        prefix = "dataset-room2_512_16"
        files = {}
        for cam in ("cam0", "cam1"):
            for i in range(10):
                files[f"{prefix}/mav0/{cam}/data/{i:019d}.png"] = os.urandom(2000 + 97 * i)
            files[f"{prefix}/mav0/{cam}/data.csv"] = b"#timestamp [ns],filename\n"
        files[f"{prefix}/mav0/imu0/data.csv"] = b"#timestamp [ns],w_RS_S_x\n"
        files[f"{prefix}/mav0/mocap0/data.csv"] = b"#timestamp [ns], p_RS_R_x [m]\n"
        files[f"{prefix}/dso/cam0/camera.txt"] = b"not kept\n"
        files[f"{prefix}/dso/gt_imu.csv"] = b"not kept\n"
        with tarfile.open(root / "www" / "room2.tar", "w") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        self.expected = {n.split("/", 1)[1]: d for n, d in files.items() if "/mav0/" in n}
        handler = functools.partial(QuietHandler, directory=str(root / "www"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/room2.tar"
        self.dest = root / "out"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def fetched(self):
        out = tumvi_fetch.fetch("room2", self.dest, url=self.url, log=lambda m: None)
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}

    def test_keeps_cameras_imu_and_mocap(self):
        self.assertEqual(self.fetched(), self.expected)

    def test_rerun_skips_files_on_disk(self):
        self.fetched()
        first = self.dest / "room2" / "mav0/cam0/data/0000000000000000000.png"
        first.write_bytes(b"x" * first.stat().st_size)  # same size: kept as it is
        self.assertEqual(self.fetched()["mav0/cam0/data/0000000000000000000.png"], first.read_bytes())

    def test_only_room_sequences(self):
        with self.assertRaises(SystemExit):
            tumvi_fetch.fetch("corridor1", self.dest, url=self.url)


if __name__ == "__main__":
    unittest.main()
