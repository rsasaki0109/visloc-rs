#!/usr/bin/env python3
"""Serves the demo page (web/) for phones.

- COOP/COEP headers, so the page is cross-origin isolated and can use the
  wasm-threads build (SharedArrayBuffer).
- `application/wasm` for .wasm, no caching (rebuilds show up on reload).
- Byte ranges (`206 Partial Content`): iOS Safari will not play a video
  from a server without them (the demo's mp4).
- `POST /upload?id=<name>&offset=<bytes>&last=<0|1>`: a recorded session
  arrives in chunks (proxies such as a Cloudflare tunnel cap request bodies)
  and is written to recordings/<name>.vrec.
- `POST /telemetry`: the page's diagnostics, `{session, events: [...]}`,
  appended as JSON lines to telemetry/<session>.jsonl (each event with the
  server's receive time `rx`). Errors, stalls and session starts are also
  printed to the log.

Camera and motion sensors need a secure context: open it through https (for
example `cloudflared tunnel --url http://127.0.0.1:8787`) or on localhost.

    python3 serve.py [--port 8787] [--root web] [--recordings recordings] [--telemetry telemetry]
"""
import argparse
import http.server
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path

MAX_CHUNK = 64 << 20
MAX_FILE = 2 << 30
MAX_TELEMETRY = 1 << 20
# Telemetry events also printed to the server log.
LOGGED_EVENTS = {
    "boot", "worker-ready", "start", "start-failed", "camera", "motion-first", "motion-permission",
    "session", "stop", "stall", "unstall", "worker-stuck", "error", "console", "worker-log",
    "frame-clock", "repeats-off", "sync-applied", "upload",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".mp4": "video/mp4",
    }

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        """Files with an optional single `Range: bytes=a-b` request."""
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
        size = os.path.getsize(path)
        if not m or (m.group(1) == "" and m.group(2) == ""):
            return super().send_head()
        if m.group(1) == "":
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._range_left = None
        while left > 0:
            chunk = source.read(min(left, 1 << 16))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code="-", size="-"):
        # Telemetry arrives every two seconds per phone: log only failures.
        if self.path.split("?")[0].endswith("/telemetry") and str(code) == "200":
            return
        super().log_request(code, size)

    def telemetry(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"error": "bad length"})
        if length <= 0 or length > MAX_TELEMETRY:
            return self.reply(400, {"error": "bad telemetry size"})
        try:
            batch = json.loads(self.rfile.read(length))
        except ValueError:
            return self.reply(400, {"error": "not json"})
        events = batch.get("events") if isinstance(batch, dict) else None
        if not isinstance(events, list):
            return self.reply(400, {"error": "no events"})
        session = re.sub(r"[^A-Za-z0-9._-]", "_", str(batch.get("session", "")))[:80] or "unknown"
        rx = round(time.time(), 3)
        lines = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event["rx"] = rx
            lines.append(json.dumps(event, separators=(",", ":")))
            if event.get("type") in LOGGED_EVENTS:
                sys.stderr.write(f"telemetry {session} {lines[-1][:600]}\n")
        if batch.get("lost") or batch.get("failures"):
            lines.append(json.dumps({"type": "transport", "lost": batch.get("lost"), "failures": batch.get("failures"), "rx": rx}))
        with self.server.telemetry_lock, open(self.server.telemetry / f"{session}.jsonl", "a") as f:
            f.write("".join(line + "\n" for line in lines))
        return self.reply(200, {"ok": True})

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        endpoint = url.path.rstrip("/").split("/")[-1]
        if endpoint == "telemetry":
            return self.telemetry()
        if endpoint != "upload":
            return self.reply(404, {"error": "not found"})
        query = urllib.parse.parse_qs(url.query)
        name = re.sub(r"[^A-Za-z0-9._-]", "_", query.get("id", [""])[0])[:160]
        try:
            offset = int(query.get("offset", ["0"])[0])
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.reply(400, {"error": "bad offset or length"})
        if not name or length <= 0 or length > MAX_CHUNK or offset < 0 or offset + length > MAX_FILE:
            return self.reply(400, {"error": "bad upload"})
        path = self.server.recordings / f"{name}.vrec"
        if offset == 0 and path.exists():
            path.unlink()
        if offset != (path.stat().st_size if path.exists() else 0):
            return self.reply(409, {"error": "chunks out of order", "have": path.stat().st_size if path.exists() else 0})
        data = self.rfile.read(length)
        with open(path, "ab") as f:
            f.write(data)
        done = query.get("last", ["0"])[0] == "1"
        if done:
            sys.stderr.write(f"recording saved: {path} ({path.stat().st_size} bytes)\n")
        return self.reply(200, {"saved": path.name, "bytes": path.stat().st_size, "done": done})


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--root", type=Path, default=here / "web")
    ap.add_argument("--recordings", type=Path, default=here / "recordings")
    ap.add_argument("--telemetry", type=Path, default=here / "telemetry")
    args = ap.parse_args()
    args.recordings.mkdir(parents=True, exist_ok=True)
    args.telemetry.mkdir(parents=True, exist_ok=True)
    os.chdir(args.root)
    server = http.server.ThreadingHTTPServer((args.bind, args.port), Handler)
    server.recordings = args.recordings.resolve()
    server.telemetry = args.telemetry.resolve()
    server.telemetry_lock = threading.Lock()
    print(f"serving {args.root} on http://{args.bind}:{args.port} (recordings -> {server.recordings}, telemetry -> {server.telemetry})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
