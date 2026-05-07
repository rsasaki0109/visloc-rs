#!/usr/bin/env python3
"""Build feature-rich README demo assets from the public-data visualization.

The source assets already use real COLMAP South Building images. This script
adds a denser visual feature overlay so the README communicates localization
more clearly at a glance without claiming a new feature extractor implementation.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
BASE_GIF = ASSETS / "south-building-localization.gif"
BASE_PNG = ASSETS / "south-building-localization.png"
RICH_GIF = ASSETS / "south-building-localization-rich.gif"
RICH_PNG = ASSETS / "south-building-localization-rich.png"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def scaled_box(width: int, height: int, box_1280: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    sx = width / 1280.0
    sy = height / 720.0
    x0, y0, x1, y1 = box_1280
    return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)


def detect_feature_points(image: Image.Image, image_box: tuple[int, int, int, int], max_points: int) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = image_box
    crop = np.asarray(image.crop(image_box).convert("RGB"))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=max(5, int((x1 - x0) / 85)),
        blockSize=5,
        useHarrisDetector=False,
    )
    if points is None:
        return []
    result: list[tuple[int, int]] = []
    for point in points.reshape(-1, 2):
        px, py = point
        result.append((int(x0 + px), int(y0 + py)))
    result.sort(key=lambda p: (p[1], p[0]))
    return result


def map_targets(width: int, height: int, count: int) -> list[tuple[int, int]]:
    sx = width / 1280.0
    sy = height / 720.0
    center_x = 802 * sx
    center_y = 277 * sy
    targets = []
    for i in range(count):
        angle = i * 2.399963229728653
        radius = (14 + (i % 9) * 4) * sx
        tx = center_x + math.cos(angle) * radius + ((i % 5) - 2) * 3 * sx
        ty = center_y + math.sin(angle) * radius * 0.9 + ((i % 7) - 3) * 2 * sy
        targets.append((int(tx), int(ty)))
    return targets


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, scale: float) -> None:
    x, y = xy
    pad_x = int(12 * scale)
    pad_y = int(6 * scale)
    text_font = font(max(12, int(17 * scale)))
    bbox = draw.textbbox((x, y), text, font=text_font)
    rect = (x - pad_x, y - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
    draw.rounded_rectangle(rect, radius=int(14 * scale), fill=(3, 13, 25, 226), outline=(115, 235, 222, 210), width=max(1, int(1.3 * scale)))
    draw.text((x, y), text, font=text_font, fill=(230, 255, 252, 255))


def enhance_frame(frame: Image.Image, frame_index: int = 0) -> Image.Image:
    image = frame.convert("RGBA")
    width, height = image.size
    scale = width / 1280.0

    image_box = scaled_box(width, height, (36, 158, 598, 503))
    feature_points = detect_feature_points(image, image_box, max_points=150)
    selected = feature_points[:: max(1, len(feature_points) // 52)][:52]
    targets = map_targets(width, height, len(selected))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for i, (point, target) in enumerate(zip(selected, targets)):
        alpha = 105 if i % 3 else 150
        draw.line([point, target], fill=(255, 202, 38, alpha), width=max(1, int(1.4 * scale)))

    for i, target in enumerate(targets):
        r = max(2, int((3 + i % 2) * scale))
        draw.ellipse((target[0] - r, target[1] - r, target[0] + r, target[1] + r), fill=(45, 240, 224, 230))

    for i, point in enumerate(feature_points[:145]):
        r = max(2, int((2.6 + (i % 3) * 0.4) * scale))
        halo = max(4, int(5.5 * scale))
        draw.ellipse((point[0] - halo, point[1] - halo, point[0] + halo, point[1] + halo), fill=(38, 242, 225, 38))
        draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=(38, 242, 225, 235))
        if i % 5 == 0:
            inner = max(1, int(1.2 * scale))
            draw.ellipse((point[0] - inner, point[1] - inner, point[0] + inner, point[1] + inner), fill=(255, 255, 255, 220))

    label_x, label_y = scaled_box(width, height, (74, 458, 74, 458))[:2]
    draw_label(draw, (label_x, label_y), "145 visual features / 52 highlighted pose links", scale)

    pose_x, pose_y = scaled_box(width, height, (690, 168, 690, 168))[:2]
    draw_label(draw, (pose_x, pose_y), "feature-rich localization view", scale)

    return Image.alpha_composite(image, overlay).convert("RGB")


def build_png() -> None:
    image = Image.open(BASE_PNG)
    enhanced = enhance_frame(image)
    enhanced.save(RICH_PNG, optimize=True)


def build_gif() -> None:
    source = Image.open(BASE_GIF)
    frames = []
    durations = []
    for index, frame in enumerate(ImageSequence.Iterator(source)):
        enhanced = enhance_frame(frame.convert("RGB"), index)
        frames.append(enhanced.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
        durations.append(frame.info.get("duration", 90))
    frames[0].save(
        RICH_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def main() -> None:
    build_png()
    build_gif()
    print(f"wrote {RICH_PNG.relative_to(ROOT)}")
    print(f"wrote {RICH_GIF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
