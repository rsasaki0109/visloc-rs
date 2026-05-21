"""Pillow drawing helpers for the KITTI revisit README asset renderer."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kitti_revisit_report_asset_data import Candidate, SvgLine


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def resize_fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.LANCZOS)
    offset = ((target_w - resized.width) // 2, (target_h - resized.height) // 2)
    return resized, offset


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x, y = xy
    pad_x, pad_y = 14, 8
    w, h = text_size(draw, text, font)
    draw.rounded_rectangle(
        (x, y, x + w + pad_x * 2, y + h + pad_y * 2),
        radius=8,
        fill=fill,
    )
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=(255, 255, 255))


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + max_width, y + 38), radius=6, fill=(15, 23, 42))
    draw.text((x + 14, y + 8), text, font=font, fill=(255, 255, 255))


def draw_verified_overlay(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    from_img: Image.Image,
    to_img: Image.Image,
    lines: list[SvgLine],
    xy: tuple[int, int],
    size: tuple[int, int],
    label_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    candidate: Candidate,
) -> None:
    x, y = xy
    width, height = size
    gap = 18
    pair_w = (width - gap) // 2
    pair_h = height
    left_img, left_offset = resize_fit(from_img, (pair_w, pair_h))
    right_img, right_offset = resize_fit(to_img, (pair_w, pair_h))
    left_xy = (x, y)
    right_xy = (x + pair_w + gap, y)

    draw.rounded_rectangle((x - 1, y - 1, x + width + 1, y + height + 1), radius=8, fill=(226, 232, 240))
    canvas.paste(left_img, (left_xy[0] + left_offset[0], left_xy[1] + left_offset[1]))
    canvas.paste(right_img, (right_xy[0] + right_offset[0], right_xy[1] + right_offset[1]))
    draw.rectangle((x + pair_w, y, x + pair_w + gap, y + pair_h), fill=(248, 250, 252))

    scale_left_x = left_img.width / from_img.width
    scale_left_y = left_img.height / from_img.height
    scale_right_x = right_img.width / to_img.width
    scale_right_y = right_img.height / to_img.height
    svg_right_x = from_img.width + 24.0

    for index, (x1, y1, x2, y2) in enumerate(lines[:80]):
        color = (15, 118, 110) if index % 2 == 0 else (180, 83, 9)
        px1 = left_xy[0] + left_offset[0] + x1 * scale_left_x
        py1 = left_xy[1] + left_offset[1] + y1 * scale_left_y
        px2 = right_xy[0] + right_offset[0] + (x2 - svg_right_x) * scale_right_x
        py2 = right_xy[1] + right_offset[1] + y2 * scale_right_y
        draw.line((px1, py1, px2, py2), fill=color, width=2)
        draw.ellipse((px1 - 2, py1 - 2, px1 + 2, py1 + 2), fill=color)
        draw.ellipse((px2 - 2, py2 - 2, px2 + 2, py2 + 2), fill=color)

    draw_label(
        draw,
        (x + left_offset[0] + 12, y + left_offset[1] + 12),
        f"matched keyframe {candidate['matched_keyframe_id']}",
        label_font,
        310,
    )
    draw_label(
        draw,
        (right_xy[0] + right_offset[0] + 12, y + right_offset[1] + 12),
        f"query frame {candidate['query_frame_id']}",
        label_font,
        260,
    )

    caption = (
        f"{candidate['inliers']}/{candidate['inliers']} verified inlier matches shown; "
        f"{candidate['inliers']}/{candidate['matches']} frontend correspondences accepted"
    )
    cap_w, cap_h = text_size(draw, caption, small_font)
    cap_x = x + 12
    cap_y = y + height - cap_h - 18
    draw.rounded_rectangle(
        (cap_x - 10, cap_y - 8, cap_x + cap_w + 10, cap_y + cap_h + 8),
        radius=6,
        fill=(15, 23, 42),
    )
    draw.text((cap_x, cap_y - 1), caption, font=small_font, fill=(226, 232, 240))
