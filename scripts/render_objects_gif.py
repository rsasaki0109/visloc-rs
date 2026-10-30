#!/usr/bin/env python3
"""Render the metric-objects demo GIF: a KITTI raw drive with each car's
height, distance and speed in metres from one camera and an IMU, a top view,
and the same drive with the IMU off.

    python3 scripts/render_objects_gif.py --euroc data/kitti_euroc/0084 \\
        --boxes boxes.csv --objects ground_objects.csv --feet ground_tracks.csv \\
        --trajectory mono/trajectory.tum --visual-only mono_vo/trajectory.tum \\
        --matches matches.csv --out docs/assets/hero_metric_objects_kitti.gif

- **Top:** cam0 with the YOLOX boxes. Each object placed on the ground is
  labelled with its track's height, its distance in that frame, and its
  speed (the track's line fit, marked "avg") or "parked". An object whose
  height is implausible for its class is grey with a question mark. A grey
  box without a label was not placed.
- **Bottom left:** a top view that follows the camera, heading up, with a
  ring every 10 m: the camera's path, each object's foot, and the recent
  path of the moving ones. Still objects stay where they were seen.
- **Bottom right:** the whole drive from above. Mono + IMU is aligned to the
  ground truth by a rotation and a translation. The visual-only VO has no
  metric scale: its numbers are drawn as metres, rotated and moved onto the
  ground truth but not scaled. Beside it, the camera's own speed from each.

Inputs: `--euroc` is `kitti_raw_to_euroc.py`'s output (cam0, the calibration,
`gt_imu.tum`). `--boxes` is `yolox_track_demo --boxes-csv`'s output.
`--objects` and `--feet` are `objects_metric_offline`'s `ground_objects.csv`
and `ground_tracks.csv`. `--trajectory` and `--visual-only` are the TUM
trajectories of `basalt_euroc_vio_demo --mono` (IMU poses) and `--mono-vo`
(camera poses). `--matches` (optional) is `evaluate_kitti_objects.py --out`'s
output, for the height and speed errors in the header.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path("/usr/share/fonts/opentype/inter")

# The chart palette's dark steps, on a dark surface (the video is dark too).
CANVAS = (17, 17, 16)
SURFACE = (26, 26, 25)
FOV = (36, 36, 34)
GRID = (58, 58, 55)
MUTED = (128, 127, 120)
TEXT2 = (195, 194, 183)
TEXT = (240, 239, 236)
BLUE = (57, 135, 229)  # the camera, mono + IMU
ORANGE = (217, 89, 38)  # vehicles
RED = (230, 103, 103)  # visual only
GREY_BOX = (150, 149, 142)
CLASS_COLORS = {"car": ORANGE, "truck": ORANGE, "bus": ORANGE}
SS = 2  # supersampling of the map panels


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONTS / f"Inter-{weight}.otf"), size)
    except OSError:
        return ImageFont.truetype("DejaVuSans.ttf", size)


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices of (n, 4) quaternions (x, y, z, w)."""
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Times (ns), positions and rotations of a TUM file in seconds."""
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fields = line.split()
            whole, _, frac = fields[0].partition(".")
            rows.append((int(whole) * 1_000_000_000 + int(frac.ljust(9, "0")[:9]), *map(float, fields[1:8])))
    rows.sort()
    times = np.array([r[0] for r in rows], dtype=np.int64)
    data = np.array([r[1:] for r in rows])
    return times, data[:, :3], quat_matrix(data[:, 3:7])


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    u, d, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s) / len(src))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    scale = (d * np.diag(sign)).sum() / (src - mu_s).var(0).sum() if with_scale else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def nearest(times: np.ndarray, t: int, tolerance_ns: int) -> int | None:
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    if abs(times[i - 1] - t) < abs(times[i] - t):
        i -= 1
    return i if abs(times[i] - t) <= tolerance_ns else None


def speed_at(times: np.ndarray, positions: np.ndarray, t: int) -> float | None:
    """Horizontal speed over the second around t."""
    if t < times[0] or t > times[-1]:
        return None
    a = int(np.searchsorted(times, t - 500_000_000))
    b = min(int(np.searchsorted(times, t + 500_000_000)), len(times) - 1)
    if b <= a:
        return None
    return float(np.linalg.norm(positions[b, :2] - positions[a, :2]) / ((times[b] - times[a]) * 1e-9))


def path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def build_palette(grey_levels: int) -> Image.Image:
    """Grey levels for the video plus the overlay colours and their blends
    with the surface, so lines keep their colour and edges stay smooth."""
    accents = [CANVAS, SURFACE, FOV, GRID, MUTED, TEXT2, TEXT, BLUE, ORANGE, RED, GREY_BOX, (12, 12, 11)]
    entries = list(accents)
    for colour in (BLUE, ORANGE, RED, TEXT2, TEXT, GREY_BOX, MUTED):
        for a in (0.25, 0.5, 0.75):
            entries.append(tuple(int(round(a * c + (1 - a) * s)) for c, s in zip(colour, SURFACE)))
    entries += [(v, v, v) for v in np.linspace(0, 255, grey_levels).astype(int)]
    flat = [c for rgb in entries for c in rgb]
    palette = Image.new("P", (1, 1))
    palette.putpalette(flat + flat[-3:] * (256 - len(entries)))
    return palette


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--euroc", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--objects", required=True, type=Path)
    parser.add_argument("--feet", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--visual-only", required=True, type=Path)
    parser.add_argument("--matches", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="KITTI raw 2011_09_26_drive_0084")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--step", type=int, default=3, help="use every n-th camera frame")
    parser.add_argument("--fps", type=float, default=7.0)
    parser.add_argument("--hold", type=float, default=2.5, help="seconds on the last frame")
    parser.add_argument("--grey-levels", type=int, default=16, help="grey levels for the video")
    parser.add_argument("--crop-top", type=int, default=90, help="image rows to drop at the top (sky)")
    parser.add_argument("--only", type=int, nargs="*", help="render these GIF frames to PNGs instead")
    args = parser.parse_args()

    # Camera frames and poses.
    with open(args.euroc / "mav0" / "cam0" / "data.csv") as handle:
        frames = sorted(int(r[0]) for r in csv.reader(handle) if r and not r[0].startswith("#"))
    calib = json.loads((args.euroc / "calibration.json").read_text())["value0"]
    t_ic = calib["T_imu_cam"][0]
    imu_cam_t = np.array([t_ic["px"], t_ic["py"], t_ic["pz"]])
    imu_cam_r = quat_matrix(np.array([[t_ic["qx"], t_ic["qy"], t_ic["qz"], t_ic["qw"]]]))[0]
    intr = calib["intrinsics"][0]["intrinsics"]
    image_w, image_h = calib["resolution"][0]

    gt_t, gt_p, gt_r = load_tum(args.euroc / "gt_imu.tum")
    gt_cam = gt_p + gt_r @ imu_cam_t
    vio_t, vio_p, vio_r = load_tum(args.trajectory)
    vio_cam = vio_p + vio_r @ imu_cam_t
    vio_axis = (vio_r @ imu_cam_r)[:, :, 2]  # optical axis in the world
    vo_t, vo_p, _ = load_tum(args.visual_only)

    def associate(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pairs = [(i, nearest(gt_t, t, 5_000_000)) for i, t in enumerate(times)]
        pairs = [(i, j) for i, j in pairs if j is not None]
        return np.array([i for i, _ in pairs]), np.array([j for _, j in pairs])

    i, j = associate(vio_t)
    _, rot, trans = umeyama(vio_cam[i], gt_cam[j], with_scale=False)
    vio_on_gt = vio_cam @ rot.T + trans
    i, j = associate(vo_t)
    vo_scale, vo_rot, _ = umeyama(vo_p[i], gt_cam[j], with_scale=True)
    vo_on_gt = (vo_p - vo_p[0]) @ vo_rot.T + gt_cam[j[0]]  # rotated and moved, not scaled
    print(f"visual only: Sim(3) scale {vo_scale:.1f}, {path_length(vo_p):.1f} units against "
          f"{path_length(gt_cam[j]):.0f} m")

    # Objects.
    objects = {r["object_id"]: r for r in read_csv(args.objects)}
    boxes = defaultdict(list)
    for r in read_csv(args.boxes):
        boxes[int(r["timestamp_ns"])].append(r)
    feet = defaultdict(list)
    for r in read_csv(args.feet):
        feet[r["object_id"]].append((int(r["timestamp_ns"]), float(r["x"]), float(r["y"]), float(r["distance"])))
    for rows in feet.values():
        rows.sort()
    note = ""
    if args.matches:
        rows = [r for r in read_csv(args.matches) if r["class"] in ("car", "person")]
        moving = [r for r in rows if float(r["gt_speed"]) >= 1.0]
        if rows:
            err = statistics.median(abs(float(r["height"]) / float(r["gt_height"]) - 1) for r in rows)
            kinds = {r["class"] for r in rows}
            what = f"{kinds.pop()}s" if len(kinds) == 1 else "objects"
            note = f"against KITTI's 3D boxes: heights {100 * err:.0f} % off (median, {len(rows)} {what})"
        if moving:
            err = statistics.median(abs(float(r["speed"]) - float(r["gt_speed"])) for r in moving)
            note += f", moving ones' speeds {err:.1f} m/s off ({len(moving)})"

    def foot_at(oid: str, t: int, tolerance_ns: int = 150_000_000):
        rows = feet.get(oid)
        if not rows:
            return None
        k = nearest(np.array([r[0] for r in rows], dtype=np.int64), t, tolerance_ns)
        return rows[k] if k is not None else None

    # Layout.
    width = args.width
    u = width / 900
    header_h = round(50 * u)
    crop = args.crop_top
    video_h = round((image_h - crop) * width / image_w)
    gap = round(8 * u)
    panel_w = (width - gap) // 2
    panel_h = round(280 * u)
    height = header_h + video_h + gap + panel_h + gap
    f_title = font("SemiBold", round(17 * u))
    f_sub = font("Medium", round(12 * u))
    f_tag = font("SemiBold", round(13 * u))
    f_tag2 = font("Medium", round(11 * u))
    f_map_head = font("SemiBold", round(14 * u * SS))
    f_map_small = font("Medium", round(11 * u * SS))
    f_map_value = font("SemiBold", round(13 * u * SS))
    palette = build_palette(args.grey_levels)
    sx = width / image_w

    # Whole-drive view: a fixed north-up extent.
    route = np.vstack([gt_cam[:, :2], vio_on_gt[:, :2]])
    lo, hi = route.min(0), route.max(0)
    map_top = round(52 * u)
    usable_w, usable_h = panel_w - round(24 * u), panel_h - map_top - round(14 * u)
    ppm_route = min(usable_w / (hi - lo)[0], usable_h / (hi - lo)[1])
    route_origin = ((lo + hi) / 2, (panel_w / 2, map_top + usable_h / 2))

    def route_xy(p) -> tuple[float, float]:
        (cx, cy), (ox, oy) = route_origin
        return (SS * (ox + (p[0] - cx) * ppm_route), SS * (oy - (p[1] - cy) * ppm_route))

    # Where the route leaves room for the speed readout.
    corners = {"top-left": (0, 0), "top-right": (1, 0), "bottom-left": (0, 1), "bottom-right": (1, 1)}
    pts = np.array([route_xy(p) for p in gt_cam[::5]]) / SS
    readout_w, readout_h = round(176 * u), round(86 * u)

    def corner_box(corner):
        cx, cy = corners[corner]
        x0 = round(10 * u) if cx == 0 else panel_w - readout_w - round(10 * u)
        y0 = map_top if cy == 0 else panel_h - readout_h - round(10 * u)
        return x0, y0, x0 + readout_w, y0 + readout_h

    def crowding(corner):
        x0, y0, x1, y1 = corner_box(corner)
        return int(((pts[:, 0] > x0 - 10) & (pts[:, 0] < x1 + 10) & (pts[:, 1] > y0 - 10) & (pts[:, 1] < y1 + 10)).sum())

    readout_corner = min(corners, key=crowding)

    local_ppm = (panel_h - round(44 * u)) / 72.0
    fov_left = np.arctan2(intr["cx"], intr["fx"])
    fov_right = np.arctan2(image_w - intr["cx"], intr["fx"])

    picks = list(range(0, len(frames), args.step))
    wanted = set(args.only) if args.only is not None else None
    gif_frames = []
    for gi, k in enumerate(picks):
        if wanted is not None and gi not in wanted:
            continue
        t = frames[k]
        elapsed = (t - frames[0]) * 1e-9
        canvas = Image.new("RGB", (width, height), CANVAS)
        draw = ImageDraw.Draw(canvas)

        # Header.
        pad = round(12 * u)
        draw.text((pad, round(8 * u)), "One camera + IMU: object size, distance and speed in metres",
                  font=f_title, fill=TEXT)
        draw.text((pad, round(30 * u)), note or "YOLOX boxes, mono + IMU VIO, objects placed on the ground",
                  font=f_sub, fill=TEXT2)
        right = f"{args.title}   {elapsed:4.1f} s"
        draw.text((width - pad - draw.textlength(right, font=f_sub), round(12 * u)), right, font=f_sub, fill=TEXT2)

        # Video with labels.
        image = Image.open(args.euroc / "mav0" / "cam0" / "data" / f"{t}.png").convert("RGB")
        image = image.crop((0, crop, image_w, image_h)).resize((width, video_h), Image.BILINEAR)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        labelled = []
        for r in boxes.get(t, []):
            box = [(float(r[key]) - (crop if key[0] == "y" else 0)) * sx for key in ("x_min", "y_min", "x_max", "y_max")]
            obj = objects.get(r["object_id"])
            if obj is None:
                od.rectangle(box, outline=GREY_BOX + (150,), width=1)
                continue
            plausible = obj["plausible"] == "yes"
            colour = CLASS_COLORS.get(obj["class"], ORANGE) if plausible else GREY_BOX
            od.rounded_rectangle(box, radius=3, outline=colour + (255,), width=2)
            foot = foot_at(r["object_id"], t)
            labelled.append((foot[3] if foot else float(obj["distance"]), box, obj, foot, colour, plausible))
        taken = []
        for distance, box, obj, foot, colour, plausible in sorted(labelled, key=lambda item: item[0]):
            head = f"{float(obj['height']):.2f} m" + ("" if plausible else " ?")
            if obj["moving"] == "true":
                motion = f"avg {float(obj['speed']) * 3.6:.0f} km/h"
            else:
                motion = "parked"
            sub = (f"{foot[3]:.0f} m · " if foot else "") + motion
            w = max(od.textlength(head, font=f_tag), od.textlength(sub, font=f_tag2)) + round(14 * u)
            h = round(34 * u)
            cx = min(max(2, (box[0] + box[2]) / 2 - w / 2), width - w - 2)
            for cy in (box[1] - h - 3, box[3] + 3):
                chip = (cx, cy, cx + w, cy + h)
                if cy < 2 or cy + h > video_h - 2:
                    continue
                if any(chip[0] < o[2] + 2 and o[0] < chip[2] + 2 and chip[1] < o[3] + 2 and o[1] < chip[3] + 2
                       for o in taken):
                    continue
                taken.append(chip)
                od.rounded_rectangle(chip, radius=round(6 * u), fill=(12, 12, 11, 215))
                od.rectangle((cx, cy + 4, cx + round(3 * u), cy + h - 4), fill=colour + (255,))
                od.text((cx + round(8 * u), cy + round(2 * u)), head, font=f_tag, fill=TEXT + (255,))
                od.text((cx + round(8 * u), cy + round(18 * u)), sub, font=f_tag2, fill=TEXT2 + (255,))
                break
        canvas.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), (0, header_h))

        # Bottom left: the top view that follows the camera.
        vi = nearest(vio_t, t, 5_000_000)
        local = Image.new("RGB", (panel_w * SS, panel_h * SS), SURFACE)
        ld = ImageDraw.Draw(local)
        if vi is not None:
            centre = vio_cam[vi, :2]
            forward = vio_axis[vi, :2] / np.linalg.norm(vio_axis[vi, :2])
            left = np.array([-forward[1], forward[0]])
            ox, oy = panel_w * SS / 2, (panel_h - round(30 * u)) * SS
            ppm = local_ppm * SS

            def to_local(p) -> tuple[float, float]:
                d = np.asarray(p[:2]) - centre
                return ox - float(d @ left) * ppm, oy - float(d @ forward) * ppm

            reach = 80 * ppm
            ld.polygon([(ox, oy), (ox - reach * np.sin(fov_left), oy - reach * np.cos(fov_left)),
                        (ox + reach * np.sin(fov_right), oy - reach * np.cos(fov_right))], fill=FOV)
            for ring in range(10, 80, 10):
                rr = ring * ppm
                ld.ellipse((ox - rr, oy - rr, ox + rr, oy + rr), outline=GRID, width=SS)
                # Outside the field of view, where the objects are not.
                angle = fov_right + 0.12
                ld.text((ox + rr * np.sin(angle) + SS * 3, oy - rr * np.cos(angle) - SS * 14), f"{ring} m",
                        font=f_map_small, fill=MUTED)
            past = (vio_t <= t) & (vio_t >= t - 20_000_000_000)
            path = [to_local(p) for p in vio_cam[past]]
            if len(path) > 1:
                ld.line(path, fill=BLUE, width=3 * SS, joint="curve")
            for oid, obj in objects.items():
                rows = feet.get(oid)
                if not rows or rows[0][0] > t:
                    continue
                plausible = obj["plausible"] == "yes"
                colour = CLASS_COLORS.get(obj["class"], ORANGE) if plausible else GREY_BOX
                side = max(1.6, min(3.0, float(obj["width"]))) * ppm / 2
                visible = any(b["object_id"] == oid for b in boxes.get(t, []))
                if obj["moving"] == "true":
                    recent = [r for r in rows if t - 2_000_000_000 <= r[0] <= t]
                    if not recent or t - recent[-1][0] > 300_000_000:
                        continue
                    trail = [to_local(r[1:3]) for r in recent]
                    if len(trail) > 1:
                        ld.line(trail, fill=colour, width=2 * SS)
                    x, y = trail[-1]
                    ld.rectangle((x - side, y - side, x + side, y + side), fill=colour)
                else:
                    x, y = to_local((float(obj["x"]), float(obj["y"])))
                    if visible:
                        ld.rectangle((x - side, y - side, x + side, y + side), fill=colour)
                    else:
                        ld.rectangle((x - side, y - side, x + side, y + side), outline=colour, width=SS)
            ld.polygon([(ox, oy - 9 * SS), (ox - 6 * SS, oy + 6 * SS), (ox + 6 * SS, oy + 6 * SS)], fill=BLUE)
        ld.rectangle((0, 0, panel_w * SS, round(44 * u) * SS), fill=SURFACE)
        ld.text((round(12 * u) * SS, round(9 * u) * SS), "Top view, metres (mono + IMU)", font=f_map_head, fill=TEXT)
        ld.text((round(12 * u) * SS, round(27 * u) * SS), "camera heading up, a ring every 10 m",
                font=f_map_small, fill=TEXT2)
        canvas.paste(local.resize((panel_w, panel_h), Image.LANCZOS), (0, header_h + video_h + gap))

        # Bottom right: the whole drive, and the same drive with the IMU off.
        whole = Image.new("RGB", (panel_w * SS, panel_h * SS), SURFACE)
        wd = ImageDraw.Draw(whole)
        wd.line([route_xy(p) for p in gt_cam], fill=GRID, width=3 * SS, joint="curve")
        done = gt_t <= t
        if done.sum() > 1:
            wd.line([route_xy(p) for p in gt_cam[done]], fill=TEXT2, width=2 * SS, joint="curve")
        done = vio_t <= t
        if done.sum() > 1:
            wd.line([route_xy(p) for p in vio_on_gt[done]], fill=BLUE, width=2 * SS, joint="curve")
            x, y = route_xy(vio_on_gt[done][-1])
            wd.ellipse((x - 4 * SS, y - 4 * SS, x + 4 * SS, y + 4 * SS), fill=BLUE)
        done = vo_t <= t
        if done.any():
            x0, y0 = route_xy(vo_on_gt[0])
            rr = 11 * SS
            wd.ellipse((x0 - rr, y0 - rr, x0 + rr, y0 + rr), outline=RED, width=SS)
            if done.sum() > 1:
                wd.line([route_xy(p) for p in vo_on_gt[done]], fill=RED, width=2 * SS)
            x, y = route_xy(vo_on_gt[done][-1])
            wd.ellipse((x - 3 * SS, y - 3 * SS, x + 3 * SS, y + 3 * SS), fill=RED)
            driven = path_length(gt_cam[(gt_t >= vo_t[0]) & (gt_t <= t)])
            lines = [f"visual only: {path_length(vo_p[done]):.1f} m", f"for {driven:.0f} m driven"]
            lw = max(wd.textlength(line, font=f_map_small) for line in lines)
            tx = x0 + rr + 4 * SS if x0 + rr + 4 * SS + lw < panel_w * SS - 6 * SS else x0 - rr - 4 * SS - lw
            for n, line in enumerate(lines):
                wd.text((tx, y0 - 15 * SS + n * 15 * SS), line, font=f_map_small, fill=RED if n == 0 else TEXT2)
        bar = 50 * ppm_route * SS
        bx, by = panel_w * SS - round(14 * u) * SS - bar, panel_h * SS - round(14 * u) * SS
        wd.line([(bx, by), (bx + bar, by)], fill=MUTED, width=2 * SS)
        wd.text((bx + bar / 2 - wd.textlength("50 m", font=f_map_small) / 2, by - 16 * SS), "50 m",
                font=f_map_small, fill=MUTED)
        wd.rectangle((0, 0, panel_w * SS, round(44 * u) * SS), fill=SURFACE)
        wd.text((round(12 * u) * SS, round(9 * u) * SS), "Whole drive, and with the IMU off", font=f_map_head, fill=TEXT)
        wd.text((round(12 * u) * SS, round(27 * u) * SS), "visual only has no metric scale: drawn as metres, not scaled",
                font=f_map_small, fill=TEXT2)
        x0, y0, x1, y1 = (v * SS for v in corner_box(readout_corner))
        wd.rounded_rectangle((x0, y0, x1, y1), radius=8 * SS, fill=CANVAS)
        wd.text((x0 + 10 * SS, y0 + 8 * SS), "camera's own speed", font=f_map_small, fill=TEXT2)
        rows = [("ground truth", TEXT2, speed_at(gt_t, gt_cam, t)),
                ("mono + IMU", BLUE, speed_at(vio_t, vio_cam, t)),
                ("visual only", RED, speed_at(vo_t, vo_p, t))]
        for n, (name, colour, v) in enumerate(rows):
            y = y0 + (26 + 18 * n) * SS * u
            wd.rectangle((x0 + 10 * SS, y + 4 * SS, x0 + 20 * SS, y + 14 * SS), fill=colour)
            wd.text((x0 + 26 * SS, y), name, font=f_map_small, fill=TEXT2)
            value = "starting" if v is None else (f"{v * 3.6:.1f} km/h" if v * 3.6 < 9.95 else f"{v * 3.6:.0f} km/h")
            wd.text((x1 - 10 * SS - wd.textlength(value, font=f_map_value), y - 1 * SS), value,
                    font=f_map_value, fill=TEXT)
        canvas.paste(whole.resize((panel_w, panel_h), Image.LANCZOS), (panel_w + gap, header_h + video_h + gap))

        if wanted is not None:
            png = args.out.with_name(f"{args.out.stem}_{gi:03d}.png")
            canvas.save(png)
            print(png)
            continue
        gif_frames.append(canvas.quantize(palette=palette, dither=Image.Dither.NONE))
        if gi % 25 == 0:
            print(f"frame {gi}/{len(picks)}")

    if wanted is not None:
        return 0
    hold = [gif_frames[-1]] * int(args.fps * args.hold)
    gif_frames[0].save(args.out, save_all=True, append_images=gif_frames[1:] + hold,
                       duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
