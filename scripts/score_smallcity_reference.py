#!/usr/bin/env python3
"""Score a SmallCity reconstruction against the release's own COLMAP model.

The Hierarchical 3DGS SmallCity release ships a COLMAP reconstruction of all
5,822 photos (``camera_calibration/aligned/sparse/0``, metric). It serves as
the reference here; it is not ground truth. ``fetch`` streams just the camera
poses out of its 640 MB ``images.bin`` from the 3 GB release zip, using HTTP
range requests (about 300 MB transferred, nothing large written). ``score``
Sim(3)-aligns a COLMAP text ``images.txt`` to it by camera centres. It
reports the median / p90 / max centre error in metres, plus a separate Sim(3)
fit per capture pass (``pass1``..``pass4``), which separates drift *within*
a pass from misalignment *between* passes. ``gsplat_photos`` names its
working copies ``frame_00000.png``... in sorted input order; such names are
mapped to the sorted release names, or through ``--names`` (its
``<out>/image_names.txt``) when the run used a subset of the photos.

``--trimmed`` refits the Sim(3) without the cameras beyond three times the
median error (three rounds), so that a few cameras far off do not inflate
everyone's error, and lists the cameras still over 1 m. ``--profile`` also
prints, per window of 100 photos in name order, the scale of a Sim(3) fitted
to that window relative to the global one and the window's median error
after its own fit: a scale that swings with the route's period while the
windows' own errors stay small means the model is locally right but bent.

``fetch-images`` downloads one capture pass's rectified photos (1024 px wide)
the same way, optionally only photos START..END of it in name order, e.g.
the 500-photo subsets of ``pass1`` used in ``docs/next_development_plan.md``.

    python3 scripts/score_smallcity_reference.py fetch ref_images.txt
    python3 scripts/score_smallcity_reference.py fetch-images photos pass1 500 1000
    python3 scripts/score_smallcity_reference.py score ref_images.txt out/sparse/0/images.txt
    python3 scripts/score_smallcity_reference.py score --names out/image_names.txt \
        --trimmed --profile ref_images.txt out/sparse/0/images.txt
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import urllib.request
import zipfile

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("score_smallcity_reference.py requires numpy") from exc

URL = (
    "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/"
    "datasets/full_scenes/small_city.zip"
)
MEMBER = "small_city/camera_calibration/aligned/sparse/0/images.bin"


class RangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file over HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(out_path: str) -> None:
    """Write the reference poses as COLMAP ``images.txt`` lines (no points)."""
    archive = zipfile.ZipFile(io.BufferedReader(RangeFile(URL), 8 << 20))
    f = archive.open(MEMBER)

    def read(n: int) -> bytes:
        data = f.read(n)
        if len(data) != n:
            raise SystemExit("truncated images.bin")
        return data

    (count,) = struct.unpack("<Q", read(8))
    with open(out_path, "w") as out:
        for _ in range(count):
            (image_id,) = struct.unpack("<i", read(4))
            qvec = struct.unpack("<4d", read(32))
            tvec = struct.unpack("<3d", read(24))
            (camera_id,) = struct.unpack("<i", read(4))
            name = b""
            while (c := read(1)) != b"\0":
                name += c
            (num_points,) = struct.unpack("<Q", read(8))
            left = num_points * 24
            while left:
                left -= len(f.read(min(left, 1 << 22)))
            fields = [image_id, *qvec, *tvec, camera_id, name.decode()]
            out.write(" ".join(map(str, fields)) + "\n")
    print(f"{count} reference poses -> {out_path}")


def fetch_images(out_dir: str, capture_pass: str, start: int | None, end: int | None) -> None:
    """Write one pass's rectified photos (``pass1_0000.jpg``...) into ``out_dir``."""
    archive = zipfile.ZipFile(io.BufferedReader(RangeFile(URL), 8 << 20))
    prefix = "small_city/camera_calibration/rectified/images/"
    names = sorted(
        n
        for n in archive.namelist()
        if n.startswith(f"{prefix}{capture_pass}_") and n.endswith(".jpg")
    )[start:end]
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(names):
        dst = os.path.join(out_dir, name.rsplit("/", 1)[-1])
        if not os.path.exists(dst):
            with open(dst + ".part", "wb") as out:
                out.write(archive.read(name))
            os.replace(dst + ".part", dst)
        if i % 100 == 0:
            print(f"{i}/{len(names)}", flush=True)
    print(f"{len(names)} photos -> {out_dir}")


def centres(path: str) -> dict[str, np.ndarray]:
    """Camera centres by image name from a COLMAP text ``images.txt``."""
    out: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            e = line.split()
            if line.startswith("#") or len(e) < 10 or "." not in e[9]:
                continue
            qw, qx, qy, qz, tx, ty, tz = map(float, e[1:8])
            r = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            out[e[9]] = -r.T @ np.array([tx, ty, tz])
    return out


def sim3_errors(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre errors after the least-squares Sim(3) taking ``est`` onto ``ref``."""
    mr, me = ref.mean(0), est.mean(0)
    r0, e0 = ref - mr, est - me
    u, s, vt = np.linalg.svd(r0.T @ e0)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (e0**2).sum()
    return np.linalg.norm(r0 - scale * (e0 @ rot.T), axis=1), scale


def sim3_fit(ref: np.ndarray, est: np.ndarray):
    """The least-squares Sim(3) taking ``est`` onto ``ref``, as a function."""
    mr, me = ref.mean(0), est.mean(0)
    r0, e0 = ref - mr, est - me
    u, s, vt = np.linalg.svd(r0.T @ e0)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ d @ vt
    scale = (s * np.diag(d)).sum() / (e0**2).sum()
    return (lambda q: mr + scale * ((q - me) @ rot.T)), scale


def score(
    ref_path: str,
    est_path: str,
    names_path: str | None = None,
    trimmed: bool = False,
    profile: bool = False,
) -> None:
    ref, est = centres(ref_path), centres(est_path)
    if names_path:
        working = dict(line.split()[:2] for line in open(names_path) if line.strip())
        est = {working[n].split("/")[-1]: c for n, c in est.items() if n in working}
    elif any(n.startswith("frame_") for n in est):
        sorted_ref = sorted(ref)
        est = {sorted_ref[int(n[6:].split(".")[0])]: c for n, c in est.items()}
    names = sorted(set(ref) & set(est))
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} images in common with the reference")
    a = np.array([ref[n] for n in names])
    b = np.array([est[n] for n in names])
    err, _ = sim3_errors(a, b)
    if trimmed:
        keep = np.ones(len(names), bool)
        for _ in range(3):
            fit, _ = sim3_fit(a[keep], b[keep])
            err = np.linalg.norm(a - fit(b), axis=1)
            keep = err <= 3 * np.median(err) + 1e-9
    print(
        f"{est_path}: {len(est)} registered, {len(names)} of {len(ref)} reference images; "
        f"centre error median {np.median(err):.3f} m, p90 {np.percentile(err, 90):.3f} m, "
        f"max {err.max():.2f} m"
    )
    if trimmed:
        far = [f"{n} {e:.2f} m" for n, e in zip(names, err) if e > 1.0]
        print(f"  cameras over 1 m (trimmed fit): {len(far)}" + (": " + ", ".join(far) if far else ""))
    if profile:
        _, global_scale = sim3_fit(a, b)
        for i in range(0, len(names) - 50, 50):
            w = slice(i, i + 100)
            fit, scale = sim3_fit(a[w], b[w])
            own = np.linalg.norm(a[w] - fit(b[w]), axis=1)
            print(
                f"  {names[i]}..: scale {scale / global_scale:.4f}, "
                f"own-fit median {np.median(own):.3f} m"
            )
    passes = np.array([n.split("_")[0] for n in names])
    for p in sorted(set(passes)):
        m = passes == p
        own, scale = sim3_errors(a[m], b[m])
        print(
            f"  {p}: {m.sum()} images; global fit median {np.median(err[m]):.3f} m; "
            f"own fit median {np.median(own):.3f} m, p90 {np.percentile(own, 90):.3f} m, "
            f"scale {scale:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="stream the reference poses into a text file")
    f.add_argument("out")
    g = sub.add_parser("fetch-images", help="download one pass's rectified photos")
    g.add_argument("out")
    g.add_argument("capture_pass", help="pass1 .. pass4")
    g.add_argument("start", nargs="?", type=int)
    g.add_argument("end", nargs="?", type=int)
    s = sub.add_parser("score", help="score an images.txt against the reference")
    s.add_argument("--names", help="gsplat_photos' image_names.txt for a subset run")
    s.add_argument("--trimmed", action="store_true", help="refit without far-off cameras")
    s.add_argument("--profile", action="store_true", help="per-100-photo scale and error")
    s.add_argument("reference")
    s.add_argument("images", nargs="+")
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.out)
    elif args.cmd == "fetch-images":
        fetch_images(args.out, args.capture_pass, args.start, args.end)
    else:
        for path in args.images:
            score(args.reference, path, args.names, args.trimmed, args.profile)


if __name__ == "__main__":
    sys.exit(main())
