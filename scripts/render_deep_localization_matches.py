#!/usr/bin/env python3
"""Render real classical-vs-deep COLMAP localization matches as a side-by-side PNG.

Consumes `correspondences.json` written by
`examples/deep_localization_demo.rs --out-dir <dir>` together with the
map and query JPGs from the COLMAP South Building dataset. Produces a
two-row PNG: classical pipeline on top, deep pipeline below, each row
showing the map image and the query image side-by-side with the actual
inlier match lines drawn between them.

Replaces `scripts/build_rich_readme_demo.py` (which used a Python
Shi-Tomasi overlay for visual flair only — not an honest depiction of
the deep pipeline). The renderer here draws *only* what the Rust
pipeline classified as inliers, so the numbers in the title bar match
the demo's `inliers: NNN` console output exactly.

Usage:
    cargo run --release --features image-io --example deep_localization_demo -- \\
        --root ~/datasets/south-building/south-building \\
        --map-image P1180141.JPG --query-image P1180144.JPG \\
        --out-dir target/deep_localization_real
    python3 scripts/render_deep_localization_matches.py \\
        --correspondences target/deep_localization_real/correspondences.json \\
        --images-dir ~/datasets/south-building/south-building/images \\
        --output docs/assets/south-building-deep-vs-classical-matches.png

Asset-generation tool, not part of CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frontends",
        nargs="+",
        default=["classical", "deep"],
        help="frontend ids to include (default: classical deep). The renderer "
        "draws each as one row in the output composite.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=200,
        help="cap match lines per panel for readability (default 200). The "
        "title bar still shows the full inlier count.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=900,
        help="downscale each pane to this width in pixels (default 900). "
        "Smaller values keep the README asset under 1 MB.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not available; install with: pip install Pillow", file=sys.stderr)
        return 2

    payload = json.loads(args.correspondences.read_text())
    map_image_name = payload["map_image"]
    query_image_name = payload["query_image"]
    frontends_by_id = {frontend["id"]: frontend for frontend in payload["frontends"]}

    map_path = args.images_dir / map_image_name
    query_path = args.images_dir / query_image_name
    if not map_path.exists() or not query_path.exists():
        print(f"map or query image missing under {args.images_dir}", file=sys.stderr)
        return 1

    map_img_full = Image.open(map_path).convert("RGB")
    query_img_full = Image.open(query_path).convert("RGB")
    original_width = map_img_full.width
    original_height = map_img_full.height

    target_w = args.image_width
    scale = target_w / original_width
    target_h = int(original_height * scale)
    map_img = map_img_full.resize((target_w, target_h), Image.LANCZOS)
    query_img = query_img_full.resize((target_w, target_h), Image.LANCZOS)

    rows = []
    for frontend_id in args.frontends:
        frontend = frontends_by_id.get(frontend_id)
        if frontend is None:
            print(f"frontend {frontend_id!r} missing in correspondences", file=sys.stderr)
            return 1
        pairs = frontend["inlier_pairs"]
        rendered_cap = min(len(pairs), args.max_lines)
        # Even sampling so the cap doesn't bias toward early matches.
        if len(pairs) > rendered_cap:
            stride = max(1, len(pairs) // rendered_cap)
            pairs = pairs[::stride][:rendered_cap]

        title_bar_h = 44
        header_h = 28
        composite = Image.new("RGB", (target_w * 2 + 8, target_h + title_bar_h + header_h), "white")
        draw = ImageDraw.Draw(composite)

        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except OSError:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()

        draw.rectangle([(0, 0), (composite.width, title_bar_h)], fill=(35, 35, 35))
        title = f"{frontend['label']} — matches: {frontend['match_count']}, inliers: {frontend['inlier_count']}"
        if len(pairs) < frontend['inlier_count']:
            title += f" (showing {len(pairs)} sampled match lines)"
        draw.text((12, 12), title, fill="white", font=font_title)

        draw.text((10, title_bar_h + 6), f"map: {map_image_name}", fill="#444444", font=font_header)
        draw.text((target_w + 18, title_bar_h + 6), f"query: {query_image_name}", fill="#444444", font=font_header)

        composite.paste(map_img, (0, title_bar_h + header_h))
        composite.paste(query_img, (target_w + 8, title_bar_h + header_h))

        # Match-line colour. Classical = warm orange, Deep = cyan-green
        # so the two rows read as distinct A/B colours at a glance.
        if frontend_id == "classical":
            line_color = (255, 140, 60)
            dot_color = (255, 100, 30)
        elif frontend_id.startswith("deep"):
            line_color = (50, 200, 220)
            dot_color = (20, 170, 200)
        else:
            line_color = (200, 60, 200)
            dot_color = (160, 30, 160)

        # Each pair: map_xy on the left pane, query_xy on the right pane.
        # Coordinates in correspondences.json are in original-image pixels;
        # we scale them to the downscaled pane.
        offset_left_x = 0
        offset_right_x = target_w + 8
        offset_y = title_bar_h + header_h
        for pair in pairs:
            mx = int(pair["map_xy"][0] * scale + offset_left_x)
            my = int(pair["map_xy"][1] * scale + offset_y)
            qx = int(pair["query_xy"][0] * scale + offset_right_x)
            qy = int(pair["query_xy"][1] * scale + offset_y)
            draw.line([(mx, my), (qx, qy)], fill=line_color, width=1)
            r = 2
            draw.ellipse((mx - r, my - r, mx + r, my + r), fill=dot_color)
            draw.ellipse((qx - r, qy - r, qx + r, qy + r), fill=dot_color)

        rows.append(composite)

    # Stack rows vertically.
    final_width = rows[0].width
    final_height = sum(row.height for row in rows) + (len(rows) - 1) * 6
    final = Image.new("RGB", (final_width, final_height), "#f0f0f0")
    y = 0
    for row in rows:
        final.paste(row, (0, y))
        y += row.height + 6

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Photographic content compresses dramatically better as JPEG;
    # the README target is "feature-rich asset under 1 MB", and a
    # quality-88 JPEG hits that with no visible match-line aliasing.
    if args.output.suffix.lower() in {".jpg", ".jpeg"}:
        final.save(args.output, quality=88, optimize=True)
    else:
        final.save(args.output, optimize=True)
    inlier_counts = ", ".join(
        f"{frontends_by_id[f]['label']}={frontends_by_id[f]['inlier_count']}"
        for f in args.frontends
    )
    size_kb = args.output.stat().st_size // 1024
    print(f"wrote {args.output} ({size_kb} KB; {inlier_counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
