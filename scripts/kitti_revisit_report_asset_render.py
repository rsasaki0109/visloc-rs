"""Image assembly for the KITTI revisit README asset renderer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from kitti_revisit_report_asset_canvas import draw_badge, draw_verified_overlay, load_font
from kitti_revisit_report_asset_data import (
    Candidate,
    SvgLine,
    find_overlay_svg,
    find_pair_images,
    first_summary_int,
    parse_svg_lines,
    read_candidates,
    read_summary,
    strongest_candidate,
)
from kitti_revisit_report_asset_layout import ReportAssetLayout


@dataclass(frozen=True)
class ReportAssetInputs:
    summary: dict[str, str]
    candidates: list[Candidate]
    strongest: Candidate
    from_image: Image.Image
    to_image: Image.Image
    overlay_lines: list[SvgLine]


def load_report_asset_inputs(report_dir: Path) -> ReportAssetInputs:
    summary = read_summary(report_dir / "summary.txt")
    candidates = read_candidates(report_dir / "candidates.csv")
    strongest = strongest_candidate(candidates)
    from_path, to_path = find_pair_images(report_dir, strongest)
    overlay_lines = parse_svg_lines(find_overlay_svg(report_dir, strongest))
    return ReportAssetInputs(
        summary=summary,
        candidates=candidates,
        strongest=strongest,
        from_image=Image.open(from_path).convert("RGB"),
        to_image=Image.open(to_path).convert("RGB"),
        overlay_lines=overlay_lines,
    )


def render_asset_image(inputs: ReportAssetInputs, width: int) -> Image.Image:
    layout = ReportAssetLayout(width)
    canvas = Image.new("RGB", layout.canvas_size, (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(42, bold=True)
    subtitle_font = load_font(24)
    badge_font = load_font(22, bold=True)
    label_font = load_font(20, bold=True)
    small_font = load_font(18)

    strongest = inputs.strongest
    title = "KITTI 00 Real Revisit Loop Candidate"
    subtitle = (
        f"{len(inputs.candidates)} verified cross-segment candidates; "
        f"strongest {strongest['matched_keyframe_id']} -> {strongest['query_frame_id']}"
    )
    draw.text((layout.margin, 28), title, font=title_font, fill=(15, 23, 42))
    draw.text((layout.margin, 84), subtitle, font=subtitle_font, fill=(71, 85, 105))

    draw_verified_overlay(
        canvas,
        draw,
        inputs.from_image,
        inputs.to_image,
        inputs.overlay_lines,
        layout.overlay_xy,
        layout.overlay_size,
        label_font,
        small_font,
        strongest,
    )

    draw_badge(
        draw,
        layout.badge_xy(0),
        f"{strongest['inliers']} verified inliers",
        badge_font,
        (15, 118, 110),
    )
    draw_badge(
        draw,
        layout.badge_xy(1),
        f"{strongest['matches']} correspondences",
        badge_font,
        (37, 99, 235),
    )
    draw_badge(
        draw,
        layout.badge_xy(2),
        f"ratio {strongest['inlier_ratio']:.3f}",
        badge_font,
        (79, 70, 229),
    )
    draw_badge(
        draw,
        layout.badge_xy(3),
        f"score {strongest['score']:.0f}",
        badge_font,
        (180, 83, 9),
    )

    settings = (
        f"Deep HogLike + MutualSoftmax; {first_summary_int(inputs.summary, 'segment_a_frames', 50)} "
        f"start frames x {first_summary_int(inputs.summary, 'segment_b_frames', 30)} revisit frames; "
        f"{first_summary_int(inputs.summary, 'max_features', 200)} features/frame"
    )
    draw.text(layout.settings_xy, settings, font=small_font, fill=(71, 85, 105))
    return canvas


def save_asset_image(image: Image.Image, out_path: Path, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=quality, optimize=True)
