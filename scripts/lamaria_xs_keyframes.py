#!/usr/bin/env python3
"""A LaMAria test recording ready for the multi-session scripts without its zip: only the keyframes' images.

    XS_DESC=~/lamaria_xs python scripts/lamaria_xs_keyframes.py WORK SEQ ESTIMATE.txt GPS.csv

The stereo frames with a pose in ESTIMATE (the submitted slam/<seq>.txt, nanoseconds) are taken as the BA chain
picks them: every --kf-step-th is a pose graph keyframe (2 Hz), every --img-step-th is matched (1 Hz). Only the
matched keyframes' PNGs are fetched from the remote ASL zip, by multi-range HTTP requests of --per-request members
(three in flight), checked against their CRC, decoded and passed to ALIKED on the GPU (1,024 keypoints, detection
threshold 0; frames of sequence_4_* darker than a mean of 70: non-local-means denoising, then CLAHE, as the chain).
A test recording comes down to about 0.5 GB in 4-5 minutes. Writes under WORK/SEQ: ba/ba3.tum (the estimate as
TUM, seconds), ba/features.npz (the pose graph keyframes), ba_aliked/features.npz (the keypoints; the descriptors go
to XS_DESC/SEQ/aliked_desc.npy, float16) and gps.csv; and WORK/calibrations/SEQ_calib_ba.json from the recording's
pinhole calibration (lamaria_to_basalt_calib.py). A recording with both feature files is skipped. Needs torch,
LightGlue (for ALIKED) and a GPU.
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import http_zip  # noqa: E402
from lamaria_xs_common import desc_dir  # noqa: E402

BASE = "https://cvg-data.inf.ethz.ch/lamaria"
DAYLIT = 70.0


def keyframes(members, tns, kf_step=10, img_step=20):
    """The stereo frames (in both cameras' members) with a pose: (pose graph keyframes, matched keyframes)."""
    tset = set(tns.tolist())
    usable = sorted(t for t in set(members[0]) & set(members[1]) if t in tset)
    return np.array(usable[::kf_step], dtype=np.int64), np.array(usable[::img_step], dtype=np.int64)


def byte_ranges(batch):
    """The (start, end) byte span of each member's local header, name, extra field (up to 256 bytes) and data."""
    return [(i.header_offset, i.header_offset + 30 + len(i.filename.encode()) + 256 + i.compress_size) for i in batch]


def parse_multipart(body, ctype, first):
    """{start offset: bytes} of a range response: multipart/byteranges, or a single range starting at `first`."""
    out = {}
    if not ctype.startswith("multipart/byteranges"):
        out[first] = body
        return out
    bnd = re.search(r"boundary=([^;]+)", ctype).group(1).strip('"').encode()
    pos = 0
    while True:
        i = body.find(b"--" + bnd, pos)
        if i < 0 or body[i + 2 + len(bnd):i + 4 + len(bnd)] == b"--":
            break
        h = body.find(b"\r\n\r\n", i)
        m = re.search(rb"Content-Range: bytes (\d+)-(\d+)/(\d+)", body[i:h], re.I)
        s, e = int(m.group(1)), int(m.group(2))
        out[s] = body[h + 4:h + 4 + e - s + 1]
        pos = h + 4 + e - s + 1
    return out


def member_bytes(info, parts):
    """A member's uncompressed bytes from the fetched spans, its CRC checked."""
    src = max(q for q in parts if q <= info.header_offset)
    raw = parts[src][info.header_offset - src:]
    sig, _, _, _, _, _, _, _, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", raw[:30])
    assert sig == 0x04034B50, "not a local file header"
    data = raw[30 + nlen + xlen:30 + nlen + xlen + info.compress_size]
    out = zlib.decompress(data, -15) if info.compress_type == 8 else data
    assert zlib.crc32(out) == info.CRC, f"CRC mismatch in {info.filename}"
    return out


def fetch(url, batch, attempts=8):
    """The members of a batch of ZipInfo, by one multi-range request."""
    ranges = byte_ranges(batch)
    spec = ",".join(f"{s}-{e}" for s, e in ranges)
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={spec}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body, ctype = r.read(), r.headers.get("Content-Type", "")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {e!r}", flush=True)
            time.sleep(http_zip.retry_delay(attempt, e))
    else:
        raise IOError("range request failed")
    parts = parse_multipart(body, ctype, ranges[0][0])
    return [member_bytes(info, parts) for info in batch]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("work", help="the recordings' work directories, WORK/SEQ")
    p.add_argument("seq")
    p.add_argument("estimate", help="the recording's submitted estimate (nanoseconds)")
    p.add_argument("gps", help="the recording's fixes (lamaria_gps_extract.py)")
    p.add_argument("--kf-step", type=int, default=10, help="stereo frames per pose graph keyframe")
    p.add_argument("--img-step", type=int, default=20, help="stereo frames per matched keyframe")
    p.add_argument("--per-request", type=int, default=100, help="members per multi-range request")
    a = p.parse_args()
    import cv2
    import torch
    from lightglue import ALIKED

    w = f"{a.work}/{a.seq}"
    dpath = f"{desc_dir()}/{a.seq}/aliked_desc.npy"
    for d in (f"{w}/ba", f"{w}/ba_aliked", f"{a.work}/calibrations", os.path.dirname(dpath)):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{w}/ba_aliked/features.npz") and os.path.exists(dpath):
        print(f"{a.seq}: features exist, skip")
        return

    est = np.loadtxt(a.estimate)
    tns = est[:, 0].astype(np.int64)
    np.savetxt(f"{w}/ba/ba3.tum", np.c_[tns / 1e9, est[:, 1:]], fmt="%.9f", header="timestamp tx ty tz qx qy qz qw")
    shutil.copyfile(a.gps, f"{w}/gps.csv")
    pin = f"{w}/pinhole.json"
    urllib.request.urlretrieve(f"{BASE}/pinhole_calibrations/test/{a.seq}.json", pin)
    subprocess.run([sys.executable, str(SCRIPTS / "lamaria_to_basalt_calib.py"), "--pinhole-calib", pin, "--out",
                    f"{a.work}/calibrations/{a.seq}_calib_ba.json"], check=True, capture_output=True)

    url = f"{BASE}/asl_folder/test/{a.seq}.zip"
    z = http_zip.open_zip(url)
    members = [{}, {}]
    for i in z.infolist():
        for c in (0, 1):
            if f"/cam{c}/data/" in i.filename and i.filename.endswith(".png"):
                members[c][int(i.filename.rsplit("/", 1)[1][:-4])] = i
    graph_kf, kf_ns = keyframes(members, tns, a.kf_step, a.img_step)
    infos = [members[c][int(t)] for t in kf_ns for c in (0, 1)]
    print(f"{a.seq}: {len(graph_kf)} pose graph keyframes, {len(kf_ns)} matched, "
          f"{sum(i.compress_size for i in infos) / 1e9:.2f} GB of "
          f"{sum(i.compress_size for m in members for i in m.values()) / 1e9:.2f} GB", flush=True)
    low = a.seq.startswith("sequence_4_")

    dev = "cuda"
    ext = ALIKED(max_num_keypoints=1024, detection_threshold=0).eval().to(dev)
    raw_path = dpath + ".part"
    xs, counts, whs = [], [], []
    t0 = time.time()
    fetched = 0
    batches = [infos[b0:b0 + a.per_request] for b0 in range(0, len(infos), a.per_request)]
    with open(raw_path, "wb") as fd, ThreadPoolExecutor(3) as pool:  # requests in flight while the GPU works
        for bi, pngs in enumerate(pool.map(lambda b: fetch(url, b), batches)):
            for png in pngs:
                fetched += len(png)
                im = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
                if low and im.mean() < DAYLIT:
                    im = cv2.fastNlMeansDenoising(im, None, 7, 7, 21)
                    im = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(im)
                with torch.inference_mode():
                    f = ext.extract(torch.from_numpy(im).to(dev)[None, None].float() / 255)
                xs.append(f["keypoints"][0].cpu().numpy().astype(np.float32))
                fd.write(f["descriptors"][0].cpu().numpy().astype(np.float16).tobytes())
                counts.append(len(xs[-1]))
                whs.append(im.shape[::-1])
            if bi % 10 == 0:
                el = time.time() - t0
                print(f"  {min((bi + 1) * a.per_request, len(infos))}/{len(infos)} images, {fetched / 1e9:.2f} GB, "
                      f"{el:.0f} s ({fetched / 1e6 / max(el, 1):.0f} MB/s)", flush=True)
    total = int(np.sum(counts))
    mm = np.lib.format.open_memmap(dpath, mode="w+", dtype=np.float16, shape=(total, 128))
    with open(raw_path, "rb") as fd:
        row = 0
        while block := fd.read(1 << 26):
            nrow = len(block) // 256
            mm[row:row + nrow] = np.frombuffer(block, np.float16).reshape(-1, 128)
            row += nrow
    mm.flush()
    del mm
    os.remove(raw_path)
    np.savez(f"{w}/ba_aliked/features.npz", kf_ns=kf_ns, offsets=np.concatenate([[0], np.cumsum(counts)]),
             xy=np.concatenate(xs), size=np.full(total, np.nan, np.float32), extractor=np.array("aliked"),
             image_wh=np.array(whs, np.int32))
    np.savez(f"{w}/ba/features.npz", kf_ns=graph_kf)
    print(f"{a.seq}: {len(kf_ns)} keyframes, {total} keypoints, {fetched / 1e9:.2f} GB fetched in "
          f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
