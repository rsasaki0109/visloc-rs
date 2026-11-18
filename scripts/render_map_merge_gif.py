#!/usr/bin/env python3
"""Render a multi-session map-merging GIF: separate SLAM maps snapping into one.

Three acts, all from files produced elsewhere in this repo:

1. Each session's SLAM map grows in its own panel, in its own world frame:
   the keyframe trajectory so far, and each landmark from the moment its
   first keyframe saw it. All sessions play at once, on one clock.
2. The sessions are merged in the order `basalt_map_merge` registered them:
   each one turns and slides from its panel by the transform the merge
   found (`map.json`'s `base_from_session`) and settles onto the map so far,
   with that registration's statistics.
3. The merged map orbits.

Inputs:

  --map DIR (repeated)  maps saved by `basalt_euroc_online_slam_demo --save-map`,
                        in the order they were passed to basalt_map_merge
                        (the first is the base)
  --merged DIR          the map `basalt_map_merge` wrote from them
  --label NAME (repeated, optional)  one per --map; the map directory's
                        name otherwise
  --cloud FILE (repeated, optional)  one per --map: a colour point cloud in
                        that map's frame (`rgbd_map_cloud.py`), drawn
                        in its own colours instead of the map's landmarks

Example:

  basalt_map_merge --base runs/mh01_map --add runs/mh02_map --add runs/mh03_map \\
      --add runs/mh04_map --add runs/mh05_map --out runs/mh_merged
  python scripts/render_map_merge_gif.py \\
      --map runs/mh01_map --map runs/mh02_map --map runs/mh03_map \\
      --map runs/mh04_map --map runs/mh05_map --merged runs/mh_merged \\
      --label MH_01 --label MH_02 --label MH_03 --label MH_04 --label MH_05 \\
      --out docs/assets/hero_map_merge.gif
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BACKGROUND = "#1e2126"
PANEL = "#272b31"
TEXT, DIM = "#f2f2f2", "#9aa0a8"
COLOURS = ["#4ea8de", "#ff9f1c", "#2ec4b6", "#f25f7a", "#b388ff", "#c0ca33", "#ff7043"]


# ----------------------------------------------------------------- loading


def quaternion_matrix(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_image_times(map_dir: Path) -> dict[int, tuple[float, int]]:
    """image_id -> (timestamp in s, camera index) from `images.csv`."""
    times = {}
    lines = (map_dir / "images.csv").read_text().splitlines()
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) >= 4 and fields[3]:
            times[int(fields[0])] = (int(fields[3]) * 1e-9, int(fields[2]))
    return times


def read_keyframe_path(map_dir: Path, times) -> tuple[np.ndarray, np.ndarray]:
    """Camera-0 keyframe centres in the map's world frame, in time order."""
    lines = [line for line in (map_dir / "sparse" / "images.txt").read_text().splitlines()
             if line and not line.startswith("#")]
    stamps, centres = [], []
    for line in lines[::2]:  # pose line, then the points2D line
        fields = line.split()
        image_id = int(fields[0])
        if image_id not in times or times[image_id][1] != 0:
            continue
        qw, qx, qy, qz, tx, ty, tz = (float(v) for v in fields[1:8])
        rotation = quaternion_matrix(qw, qx, qy, qz)  # camera from world
        centres.append(-rotation.T @ np.array([tx, ty, tz]))
        stamps.append(times[image_id][0])
    order = np.argsort(stamps)
    return np.asarray(stamps)[order], np.asarray(centres)[order]


def read_landmarks(map_dir: Path, times) -> tuple[np.ndarray, np.ndarray]:
    """Landmark positions and the time of the first keyframe that saw each."""
    points, first_seen = [], []
    for line in (map_dir / "sparse" / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        track = [int(v) for v in fields[8::2]]
        seen = [times[i][0] for i in track if i in times]
        points.append([float(v) for v in fields[1:4]])
        first_seen.append(min(seen) if seen else np.inf)
    return np.asarray(points), np.asarray(first_seen)


def se3_from_json(value) -> tuple[np.ndarray, np.ndarray]:
    return quaternion_matrix(*value["quaternion_wxyz"]), np.asarray(value["translation"])


class Session:
    def __init__(self, map_dir: Path, label: str, colour: str, session_json,
                 cloud: Path | None = None, max_points: int = 0):
        times = read_image_times(map_dir)
        self.label, self.colour = label, colour
        self.path_t, self.path = read_keyframe_path(map_dir, times)
        t0 = self.path_t[0]
        self.path_t = self.path_t - t0
        self.duration = float(self.path_t[-1])
        self.rgb = None
        if cloud is not None:
            # A colour cloud in the map's frame, timed from its first keyframe
            # (rgbd_map_cloud.py).
            data = np.load(cloud)
            self.points = data["points"].astype(np.float64)
            self.rgb = data["colours"].astype(np.float64)
            self.seen = data["first_seen"].astype(np.float64)
        else:
            self.points, self.seen = read_landmarks(map_dir, times)
            self.seen = self.seen - t0
            # Drop far-off landmarks (a few are triangulated at the edge of
            # the depth range): keep those within 8 m of the flight.
            near = np.full(len(self.points), np.inf)
            for chunk in np.array_split(np.arange(len(self.points)), max(1, len(self.points) // 4000)):
                d = np.linalg.norm(self.points[chunk, None, :] - self.path[None, ::5, :], axis=2)
                near[chunk] = d.min(axis=1)
            keep = near < 8.0
            self.points, self.seen = self.points[keep], self.seen[keep]
        if max_points and len(self.points) > max_points:
            keep = np.sort(np.random.default_rng(0).choice(len(self.points), max_points, replace=False))
            self.points, self.seen = self.points[keep], self.seen[keep]
            if self.rgb is not None:
                self.rgb = self.rgb[keep]
        self.rotation, self.translation = se3_from_json(session_json["base_from_session"])
        self.yaw = float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))
        self.registration = session_json.get("registration")
        self.centre = np.median(self.path, axis=0)
        self.keyframes = len(self.path)
        own = json.loads((map_dir / "map.json").read_text())
        self.map_images, self.map_landmarks = int(own["images"]), int(own["landmarks"])

    def placed(self, values: np.ndarray, merged: float) -> np.ndarray:
        """`values` (own frame) moved `merged` (0..1) of the way into the base frame:
        turned by that share of the merge's yaw about the session's centre, which
        slides to its place."""
        yaw = merged * self.yaw
        c, s = np.cos(yaw), np.sin(yaw)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        target = self.rotation @ self.centre + self.translation
        return (values - self.centre) @ turn.T + self.centre + merged * (target - self.centre)


# ----------------------------------------------------------------- drawing


def view_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Orthographic view: rows are the screen x and y axes in the world."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    toward = np.array([np.cos(a), np.sin(a), 0.0])
    up = np.cos(e) * np.array([0.0, 0.0, 1.0]) + np.sin(e) * toward
    return np.stack([right, up])


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def blend(colour: str, other: str, t: float):
    a = np.array(matplotlib.colors.to_rgb(colour))
    b = np.array(matplotlib.colors.to_rgb(other))
    return tuple(lerp(a, b, t))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, action="append", required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--cloud", type=Path, action="append",
                        help="per --map: a colour point cloud (.npz from rgbd_map_cloud.py) "
                             "drawn in its own colours instead of the map's landmarks")
    parser.add_argument("--note", action="append",
                        help="per --map: a panel note (default: its own frame and duration)")
    parser.add_argument("--max-points", type=int, default=0,
                        help="draw at most this many points per session (0: all)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="scale of the drawn points")
    parser.add_argument("--unit", default="flight",
                        help="what a session is, in the captions (flight, run, ...)")
    parser.add_argument("--place", default="hall", help="what was mapped, in the captions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", help="default: Multi-session mapping: N <unit>s, one map")
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--height", type=int, default=428)
    parser.add_argument("--fps", type=float, default=12.5)
    parser.add_argument("--grow-frames", type=int, default=50)
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument("--slide-frames", type=int, default=9)
    parser.add_argument("--align-frames", type=int, default=10)
    parser.add_argument("--orbit-frames", type=int, default=50)
    parser.add_argument("--orbit-degrees", type=float, default=30.0,
                        help="amplitude of the closing azimuth swing")
    parser.add_argument("--fill", type=float, default=0.68,
                        help="share of the height the merged map fills")
    parser.add_argument("--orbit-tilt", type=float, default=16.0,
                        help="degrees the orbit rises towards top-down halfway through")
    parser.add_argument("--azimuth", type=float,
                        help="view azimuth in degrees; default: the merged map's long axis "
                             "runs across the screen")
    parser.add_argument("--azimuth-offset", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=66.0)
    parser.add_argument("--frames-dir", type=Path, help="keep the PNG frames here")
    args = parser.parse_args()

    labels = args.label or [d.name for d in args.map]
    if len(labels) != len(args.map):
        parser.error("give one --label per --map")
    merged_json = json.loads((args.merged / "map.json").read_text())
    session_json = merged_json.get("sessions") or [{"base_from_session": {
        "translation": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}}]
    if len(session_json) != len(args.map):
        parser.error(f"{args.merged} holds {len(session_json)} sessions, not {len(args.map)}")
    notes = args.note or [None] * len(args.map)
    if len(notes) != len(args.map):
        parser.error("give one --note per --map")
    clouds = args.cloud or [None] * len(args.map)
    if len(clouds) != len(args.map):
        parser.error("give one --cloud per --map")
    sessions = [Session(d, label, COLOURS[i % len(COLOURS)], session_json[i], cloud,
                        args.max_points)
                for i, (d, label, cloud) in enumerate(zip(args.map, labels, clouds))]
    for s in sessions:
        print(f"{s.label}: {s.keyframes} keyframes, {len(s.points)} landmarks, "
              f"{s.duration:.0f} s, yaw {np.degrees(s.yaw):+.1f} deg")

    for name in ("Inter", "DejaVu Sans"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break

    W, H = args.width, args.height
    n = len(sessions)
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    args.title = args.title or f"Multi-session mapping: {words.get(n, n)} {args.unit}s, one map"
    all_merged = np.concatenate([s.placed(s.points, 1.0) for s in sessions])
    if args.azimuth is None:
        xy = all_merged[:, :2] - np.median(all_merged[:, :2], axis=0)
        _, vectors = np.linalg.eigh(np.cov(xy.T))
        long_axis = vectors[:, -1]
        args.azimuth = np.degrees(np.arctan2(long_axis[1], long_axis[0])) - 90.0
    args.azimuth += args.azimuth_offset
    view = view_matrix(args.azimuth, args.elevation)

    # Merged-view framing: the whole merged map in the area under the title
    # and above the legend, left of the queue of maps still to merge.
    all_paths = np.concatenate([s.placed(s.path, 1.0) for s in sessions])
    queue_w = 128
    area = np.array([W - queue_w - 40, H * args.fill])
    area_centre = np.array([(W - queue_w) / 2, H * (0.125 + args.fill / 2)])
    merged_centre_world = np.percentile(all_merged, 50, axis=0)

    def framing(view):
        """Scale and screen offset that fit the merged map (its landmarks bar
        outliers, and every trajectory) in the area. The offset is kept for
        the map's centre, so a turning view turns about it."""
        lo, hi = np.percentile(all_merged @ view.T, [0.5, 99.5], axis=0)
        paths = all_paths @ view.T
        lo, hi = np.minimum(lo, paths.min(axis=0)), np.maximum(hi, paths.max(axis=0))
        scale = float(np.min(area / (hi - lo)))
        return scale, (merged_centre_world @ view.T - (lo + hi) / 2) * scale

    fit = {}
    fit["scale"], fit["offset"] = framing(view)

    # Act 1 panels: a row of three over a row of the rest. Act 2 shrinks them
    # into a queue down the right edge.
    top = (n + 1) // 2
    rows = [list(range(top)), list(range(top, n))]
    grid_w, grid_h = W / top - 10, H * 0.355
    grid = {}
    for r, members in enumerate(rows):
        y = H * 0.66 - r * (grid_h + 8)
        offset = (top - len(members)) * (grid_w + 10) / 2
        for j, i in enumerate(members):
            grid[i] = np.array([5 + offset + j * (grid_w + 10) + grid_w / 2, y])
    queue_h = (H * 0.80 - 6 * (n - 1)) / n
    queue = {i: np.array([W - 6 - queue_w / 2, H * 0.875 - queue_h / 2 - i * (queue_h + 6)])
             for i in range(n)}
    # One scale for every panel, so the maps compare.
    spans = []
    for s in sessions:
        p = (s.points - s.centre) @ view.T
        lo_s, hi_s = np.percentile(p, [2, 98], axis=0)
        spans.append(np.maximum(np.abs(lo_s), np.abs(hi_s)) * 2)
    spans = np.max(spans, axis=0)
    grid_scale = float(np.min(np.array([grid_w * 0.92, grid_h * 0.78]) / spans))
    queue_scale = float(np.min(np.array([queue_w * 0.92, queue_h * 0.80]) / spans))
    queued = {"mix": 0.0}  # 0: act 1 grid, 1: act 2 queue

    def panel(i):
        """Centre, width, height and scale of session i's panel."""
        q = 1.0 if queued["mix"] >= 0.5 else 0.0
        return (lerp(grid[i], queue[i], q), lerp(grid_w, queue_w, q),
                lerp(grid_h, queue_h, q), lerp(grid_scale, queue_scale, q))

    def screen(s_index: int, values: np.ndarray, slide: float, align: float) -> np.ndarray:
        """Screen positions of a session's `values`: `align` of the way through
        its registration transform, and `slide` of the way from its own panel
        to the shared view."""
        s = sessions[s_index]
        world = s.placed(values, align)
        centre, _, _, scale = panel(s_index)
        own = (world - s.centre) @ view.T * scale + centre
        if slide <= 0:
            return own
        whole = (world - merged_centre_world) @ view.T * fit["scale"] + area_centre + fit["offset"]
        return lerp(own, whole, smooth(slide))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1])

    clock_end = max(s.duration for s in sessions)
    frames_dir = args.frames_dir or Path(tempfile.mkdtemp(prefix="map_merge_frames_"))
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    total_landmarks_merged = int(merged_json["landmarks"])
    total_images_merged = int(merged_json["images"])

    def draw(state, frame_index):
        """state["sessions"]: per session a dict with t (clock, s), slide and
        align (0..1), flash (0..1), panel (panel opacity) and merged (whether
        it is part of the map); plus caption and status."""
        ax.clear()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")
        ax.set_facecolor(BACKGROUND)
        states = state["sessions"]
        for i, s in enumerate(sessions):
            # Act 2 starts by fading the grid out and the queue in.
            alpha = states[i]["panel"] * min(1.0, abs(1 - 2 * queued["mix"]) * 1.4)
            if alpha > 0.01:
                (cx, cy), pw, ph, _ = panel(i)
                ax.add_patch(matplotlib.patches.FancyBboxPatch(
                    (cx - pw / 2, cy - ph / 2), pw, ph,
                    boxstyle="round,pad=0,rounding_size=6", facecolor=PANEL,
                    edgecolor="none", alpha=alpha, zorder=0))
                ax.text(cx - pw / 2 + 7, cy + ph / 2 - 6, s.label, color=s.colour,
                        fontsize=8.5 - 1.5 * queued["mix"], fontweight="bold", va="top",
                        alpha=alpha, zorder=5)
                ax.text(cx + pw / 2 - 8, cy + ph / 2 - 7,
                        notes[i] or f"its own frame · {s.duration:.0f} s {args.unit}",
                        color=DIM, fontsize=7,
                        va="top", ha="right", alpha=alpha if queued["mix"] < 0.5 else 0.0, zorder=5)
        # The map so far under the session being added.
        order = sorted(range(n), key=lambda i: (states[i].get("active", False), i))
        # Maps still in a panel fade with it while act 2 swaps grid for queue.
        swap = min(1.0, abs(1 - 2 * queued["mix"]) * 1.4)
        for i in order:
            s, st = sessions[i], states[i]
            slide, align, flash = st["slide"], st["align"], st["flash"]
            shown = s.seen <= st["t"]
            if shown.any():
                xy = screen(i, s.points[shown], slide, align)
                if s.rgb is None:
                    colour = [blend(s.colour, "#ffffff", 0.5 * flash)]
                else:
                    # True colours, tinted with the session's colour until it
                    # is registered.
                    tint = 0.0 if st["merged"] or slide <= 0 else 0.5 * (1 - align)
                    base = np.array(matplotlib.colors.to_rgb(s.colour))
                    colour = s.rgb[shown] * (1 - tint) + base * tint
                    colour = colour * (1 - 0.35 * flash) + 0.35 * flash
                size = (lerp(0.45, 0.8, smooth(slide)) * (1 + 1.2 * flash)
                        * args.point_size)
                fade = swap if slide <= 0 else 1.0
                ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colour,
                           alpha=(0.6 if not st.get("active") else 0.75) * fade,
                           linewidths=0, zorder=1 + i * 0.01 + (1 if st.get("active") else 0))
            k = np.searchsorted(s.path_t, st["t"], side="right")
            if k >= 2:
                xy = screen(i, s.path[:k], slide, align)
                ax.plot(xy[:, 0], xy[:, 1], color=blend(s.colour, "#ffffff", 0.3 + 0.4 * flash),
                        lw=lerp(0.9, 1.05, smooth(slide)), alpha=0.9 * (swap if slide <= 0 else 1.0), zorder=3,
                        solid_capstyle="round", solid_joinstyle="round")
                if st["t"] < s.duration:
                    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=14, c="white", zorder=4,
                               linewidths=0)
        ax.text(14, H - 14, args.title, color=TEXT, fontsize=11, va="top", zorder=6)
        ax.text(14, H - 34, state["caption"], color=DIM, fontsize=8.5, va="top", zorder=6)
        # Legend: one chip per session, with its state.
        x = 14
        for i, s in enumerate(sessions):
            done = states[i]["merged"]
            ax.scatter([x + 4], [16], s=30, c=[s.colour], marker="s", linewidths=0, zorder=6)
            text = s.label + ((" base" if i == 0 else " ✓") if done else "")
            ax.text(x + 12, 16, text, color=TEXT if done else DIM, fontsize=8,
                    va="center", zorder=6)
            x += 24 + 5.4 * len(text)
        if state.get("status"):
            ax.text(W - 14, 16, state["status"], color=TEXT, fontsize=8, va="center",
                    ha="right", zorder=6)
        path = frames_dir / f"frame_{len(frame_paths):04d}.png"
        fig.savefig(path, dpi=100, facecolor=BACKGROUND)
        frame_paths.append(path)

    def session_state(**kw):
        state = {"t": np.inf, "slide": 0.0, "align": 0.0, "flash": 0.0, "panel": 1.0,
                 "merged": False}
        state.update(kw)
        return state

    # Act 1: every session maps its own flight in its own frame, on one clock.
    for f in range(args.grow_frames):
        t = clock_end * f / (args.grow_frames - 1)
        draw({"sessions": [session_state(t=t) for _ in sessions],
              "caption": f"{n} separate SLAM sessions of one {args.place}, each mapping in its own "
                         f"world frame    t = {t:3.0f} s"}, f)
    for _ in range(args.hold_frames):
        draw({"sessions": [session_state() for _ in sessions],
              "caption": f"{n} maps of the same {args.place}, in {n} coordinate frames"}, 0)

    # Act 2: merge in order. The base slides into the shared view and stays;
    # each added map slides over it in its own frame (visibly off), then
    # turns and moves by the transform its registration found.
    states = [session_state() for _ in sessions]
    for f in range(args.slide_frames + 4):
        queued["mix"] = (f + 1) / (args.slide_frames + 4)
        draw({"sessions": states, "caption": "Merging them, one at a time"}, 0)
    for i, s in enumerate(sessions):
        reg = s.registration
        states[i]["active"] = True
        if i == 0:
            caption = f"{s.label} is the base: its world frame becomes the merged map's frame"
            status = f"{s.map_images:,} keyframe images, {s.map_landmarks:,} landmarks"
            for f in range(args.slide_frames):
                states[i].update(slide=(f + 1) / args.slide_frames, align=1.0,
                                 panel=1 - smooth((f + 1) / args.slide_frames))
                draw({"sessions": states, "caption": caption, "status": status}, 0)
            states[i].update(merged=True, panel=0.0)
            for _ in range(args.hold_frames // 2):
                draw({"sessions": states, "caption": caption, "status": status}, 0)
            states[i]["active"] = False
            continue
        caption = (f"+ {s.label}, in its own frame: off by {np.degrees(abs(s.yaw)):.1f}° and "
                   f"{np.linalg.norm(s.translation) * 100:.0f} cm")
        for f in range(args.slide_frames):
            u = (f + 1) / args.slide_frames
            states[i].update(slide=u, panel=1 - smooth(u))
            draw({"sessions": states, "caption": caption, "status": ""}, 0)
        states[i]["panel"] = 0.0
        for _ in range(args.hold_frames // 2):
            draw({"sessions": states, "caption": caption, "status": ""}, 0)
        caption = (f"+ {s.label}: {reg['localized_keyframes']} of {reg['keyframes']} keyframes "
                   f"localize in the map so far, {reg['keyframe_rms_m'] * 100:.1f} cm residual")
        status = f"{reg['fused_landmarks']:,} landmarks fused, {reg['added_landmarks']:,} added"
        for f in range(args.align_frames):
            states[i]["align"] = smooth((f + 1) / args.align_frames)
            draw({"sessions": states, "caption": caption, "status": ""}, 0)
        states[i]["merged"] = True
        for f in range(args.hold_frames):
            states[i]["flash"] = max(0.0, 1 - f / 3)
            draw({"sessions": states, "caption": caption, "status": status}, 0)
        states[i]["active"] = False

    # Act 3: the merged map, orbiting.
    summary = (f"One map in {sessions[0].label}'s frame: {total_images_merged:,} keyframe images, "
               f"{total_landmarks_merged:,} landmarks from {n} {args.unit}s")
    for f in range(args.orbit_frames):
        u = f / max(1, args.orbit_frames - 1)
        # A swing out and back, so the loop restarts where act 1 began.
        view = view_matrix(args.azimuth + args.orbit_degrees * np.sin(2 * np.pi * u),
                           args.elevation + args.orbit_tilt * np.sin(np.pi * u))
        fit["scale"], fit["offset"] = framing(view)
        draw({"sessions": [session_state(slide=1.0, align=1.0, panel=0.0, merged=True)
                           for _ in sessions],
              "caption": summary, "status": f"later {args.unit}s localize in it"}, 0)

    palette = frames_dir / "palette.png"
    pattern = str(frames_dir / "frame_%04d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", pattern, "-vf", "palettegen=max_colors=192:stats_mode=full",
                    str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", pattern, "-i", str(palette), "-lavfi",
                    "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    "-loop", "0", str(args.out)], check=True)
    if args.frames_dir is None:
        shutil.rmtree(frames_dir)
    print(f"wrote {args.out}: {len(frame_paths)} frames, {args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
