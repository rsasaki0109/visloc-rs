"""Upscale JPEGs 2x and carry over their APP1 Exif segment. usage: SRC DST"""
import io, os, sys
from PIL import Image
src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    if not f.lower().endswith(".jpg"): continue
    data = open(os.path.join(src, f), "rb").read()
    app1 = b""
    if data[2:4] == b"\xff\xe1":
        n = int.from_bytes(data[4:6], "big"); app1 = data[2:4 + n]
    im = Image.open(io.BytesIO(data)); im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=95); out = buf.getvalue()
    open(os.path.join(dst, f), "wb").write(out[:2] + app1 + out[2:])
