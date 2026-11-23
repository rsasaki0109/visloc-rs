#!/usr/bin/env python3
"""Third-person video of a gs_sim.py drive: a chase camera in the splat, the robot drawn in (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/chase_video.py --splat runs/street/scene.ply \
        --scene runs/street/sim/scene.json --run runs/street/sim/nav \
        --map runs/street/map --inliers runs/street/sim/inliers.csv --out runs/street/chase

`--run PREFIX` names a `nav_goals.py` run: PREFIX.csv (its record),
PREFIX.json (its result) and PREFIX.sim.csv (`gs_sim.py --record`, the true
poses). Each frame renders the splat from a camera `--back` metres behind
and `--up` metres over the robot (near the capture height, where the splat
looks right), draws the robot (`robot_model.py`) where it is in front of the
splat's depth, and overlays:

- the path driven (red) and where visloc puts the robot (a cyan ring), on the ground;
- the map's landmarks within 12 m (white dots);
- green rays from the robot's camera to the landmarks its latest localization
  matched (`visloc_map_localize_node --dump_inliers`, the `--inliers` CSV);
  when no image has localized for 0.8 s, the last fix's rays turn grey and
  the caption says how long the robot has run on wheel odometry;
- the people walking the route, when the run had `gs_sim.py --pedestrians`
  (its PREFIX.sim.csv.people.csv), in the view and in the inset;
- an inset with that localized camera image and its matched keypoints, a minimap
  and the goals.

Writes OUT/NNNNN.jpg; `ffmpeg -framerate 30 -i OUT/%05d.jpg` makes the video.
"""
import argparse, bisect, csv, json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_model  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument('--splat', required=True)
p.add_argument('--scene', required=True, help="gs_sim.py's scene.json (camera height, ground, grid)")
p.add_argument('--run', required=True, help='nav_goals.py run prefix: PREFIX.csv, PREFIX.json, PREFIX.sim.csv')
p.add_argument('--map', required=True, help='the localization map (sparse/points3D.txt)')
p.add_argument('--inliers', required=True, help='visloc_map_localize_node --dump_inliers CSV')
p.add_argument('--out', required=True)
p.add_argument('--step', type=float, default=0.5, help='simulated seconds per frame')
p.add_argument('--skip', type=float, default=8.0, help='simulated seconds left out at the start')
p.add_argument('--limit', type=int, default=10 ** 9, help='at most this many frames')
p.add_argument('--back', type=float, default=3.2)
p.add_argument('--up', type=float, default=2.75)
p.add_argument('--pitch-deg', type=float, default=16.0)
p.add_argument('--seed', type=int, default=0, help="gs_sim.py's --seed (the people's looks)")
p.add_argument('--people-style', dest='people_style', choices=['splat', 'mesh'], default='splat',
               help='splat: the people are gaussians rendered with the scene; mesh: drawn over it')
p.add_argument('--server', default=os.path.join(REPO, 'target', 'release', 'examples', 'gsplat_render_server'))
args = p.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
STEP, LIMIT, SERVER = args.step, args.limit, args.server
W, H, FX = 1280, 720, 560.0
BACK, UP, PITCH = args.back, args.up, math.radians(args.pitch_deg)
F = lambda n: ImageFont.load_default(size=n)  # noqa: E731
SCENE_DIR = os.path.dirname(os.path.abspath(args.scene))

sim = list(csv.DictReader(open(f'{args.run}.sim.csv'))); ST = [float(r['time_s']) for r in sim]
nav = list(csv.DictReader(open(f'{args.run}.csv'))); NT = [float(r['time_s']) for r in nav]
res = json.load(open(f'{args.run}.json')); scene = json.load(open(args.scene))
goals = [g['goal'] for g in res['goals']]; done_t, t0 = [], NT[0]
for g in res['goals']:
    t0 += g['sim_time_s']; done_t.append(t0 if g['reached'] else 1e9)
gr = scene['grid']
ground = np.load(os.path.join(SCENE_DIR, scene['ground'])) if scene.get('ground') else None
LM = {}
for line in open(os.path.join(args.map, 'sparse', 'points3D.txt')):
    if line.startswith('#') or not line.strip(): continue
    f = line.split(); LM[int(f[0])] = (float(f[1]), float(f[2]), float(f[3]))
LM_IDS = np.array(list(LM)); LM_XYZ = np.array([LM[i] for i in LM_IDS])
FIX = {}
for row in csv.DictReader(open(args.inliers)):
    FIX.setdefault(int(row['timestamp_ns']), []).append((float(row['x']), float(row['y']), int(row['landmark_id'])))
FT = sorted(FIX)
ROBOT = robot_model.Model(scene['camera_height'])
PEOPLE_CSV = f'{args.run}.sim.csv.people.csv'
CROWD = None
if os.path.exists(PEOPLE_CSV):
    from people import Crowd  # noqa: E402
    PEOPLE, KIND = {}, {}
    for row in csv.DictReader(open(PEOPLE_CSV)):
        PEOPLE.setdefault(float(row['time_s']), []).append(
            (float(row['x']), float(row['y']), float(row['yaw']), float(row['phase'])))
        KIND[int(row['id'])] = int(row.get('kind') or 0)
    PT = sorted(PEOPLE)
    class _Free:
        def hits(self, x, y, r): return False
    counts = [sum(1 for k in KIND.values() if k == kind) for kind in range(4)]
    CROWD = Crowd([scene['start']] + scene.get('goals', []), counts[0], _Free(), seed=args.seed,
                  cyclists=counts[1], dog_walkers=counts[2], strollers=counts[3])
    DYN = 2500 * counts[0] + 6000 * counts[1] + 4500 * counts[2] + 5500 * counts[3]

def people_at(t):
    """Put the crowd where gs_sim.py recorded it at time t."""
    rows = np.array(PEOPLE[PT[min(bisect.bisect_left(PT, t), len(PT) - 1)]])
    CROWD.xy, CROWD.yaw, CROWD.phase = rows[:, :2].copy(), rows[:, 2].copy(), rows[:, 3].copy()

CAM = scene['camera']
CW, CH = CAM['width'], CAM['height']
def gz(x, y):
    if ground is None: return 0.0
    i = min(max(int((x - gr['x0']) / gr['resolution']), 0), ground.shape[1] - 1)
    j = min(max(int((y - gr['y0']) / gr['resolution']), 0), ground.shape[0] - 1)
    return float(ground[j, i])

SPLAT = CROWD is not None and args.people_style == 'splat'
DYN_ARGS = ['--dynamic', str(DYN)] if SPLAT else []
srv = subprocess.Popen([SERVER, '--ply', args.splat, '--camera', f'PINHOLE {W} {H} {FX} {FX} {W / 2} {H / 2}', '--depth']
                       + DYN_ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
fpv = subprocess.Popen([SERVER, '--ply', args.splat, '--camera', CAM['model']] + (['--depth'] if CROWD else [])
                       + DYN_ARGS, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def ask(p, R_cw, t, n, splats=None):
    if splats is not None:
        p.stdin.write(f'dyn {len(splats)}\n'.encode() + np.ascontiguousarray(splats, '<f4').tobytes())
    p.stdin.write((' '.join(f'{v:.7f}' for v in [*R_cw.reshape(-1), *t]) + '\n').encode()); p.stdin.flush()
    return p.stdout.read(n)

def cam_axes(yaw, pitch):
    """World <- camera rotation (x right, y down, z forward) looking along yaw, pitched down."""
    c, s, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    fwd = np.array([c * cp, s * cp, -sp]); right = np.array([s, -c, 0.0]); down = np.cross(fwd, right)
    return np.stack([right, down, fwd], 1)

def project(pts, R_wc, c_w):
    cam = (np.asarray(pts) - c_w) @ R_wc
    ok = cam[:, 2] > 0.3
    return np.stack([cam[:, 0] / np.maximum(cam[:, 2], 1e-6) * FX + W / 2, cam[:, 1] / np.maximum(cam[:, 2], 1e-6) * FX + H / 2], 1), ok

# minimap
path = np.array([[float(r['true_x']), float(r['true_y'])] for r in sim[::20]])
MX0, MX1, MY0, MY1 = path[:, 0].min() - 3, path[:, 0].max() + 3, path[:, 1].min() - 3, path[:, 1].max() + 3
MW = 420; MPPM = MW / (MX1 - MX0); MH = int((MY1 - MY0) * MPPM)
mto = lambda x, y: ((x - MX0) * MPPM, (MY1 - y) * MPPM)

cam_yaw = None; trail = []; n = 0
for t in np.arange(NT[0] + args.skip, NT[-1] + 2.0, STEP):
    if n >= LIMIT: break
    for tt in np.arange(max(NT[0], t - STEP) if trail else NT[0], t, 0.25):  # trail from the start
        rr = sim[min(bisect.bisect_left(ST, tt), len(sim) - 1)]
        trail.append((float(rr['true_x']), float(rr['true_y']), gz(float(rr['true_x']), float(rr['true_y'])) + 0.03))
    a = min(bisect.bisect_left(ST, t), len(sim) - 1); r = sim[a]
    x, y, yaw = float(r['true_x']), float(r['true_y']), float(r['true_yaw'])
    b = min(bisect.bisect_left(NT, t), len(nav) - 1)
    est = (float(nav[b]['est_x']), float(nav[b]['est_y'])) if nav[b]['est_x'] else None
    z0 = gz(x, y)
    cam_yaw = yaw if cam_yaw is None else cam_yaw + 0.12 * math.atan2(math.sin(yaw - cam_yaw), math.cos(yaw - cam_yaw))
    c_w = np.array([x - BACK * math.cos(cam_yaw), y - BACK * math.sin(cam_yaw), gz(x, y) + UP])
    R_wc = cam_axes(cam_yaw, PITCH); R_cw = R_wc.T
    if SPLAT:
        people_at(t)
        # the people, except those brushing past the chase camera itself
        clear = np.hypot(CROWD.xy[:, 0] - c_w[0], CROWD.xy[:, 1] - c_w[1]) > 2.0
        buf = ask(srv, R_cw, -R_cw @ c_w, W * H * 7, CROWD.splats(c_w, gz, max_m=40.0, only=clear))
    else:
        buf = ask(srv, R_cw, -R_cw @ c_w, W * H * 7)
    rgb = np.frombuffer(buf[:W * H * 3], np.uint8).reshape(H, W, 3).astype(float)
    depth = np.frombuffer(buf[W * H * 3:], '<f4').reshape(H, W)
    # shadow
    rgb = robot_model.shadow(rgb, project, (x, y, yaw), z0, R_wc, c_w)
    # ground overlays first (the robot stands on them)
    img = Image.fromarray(rgb.astype(np.uint8)); d = ImageDraw.Draw(img)
    if len(trail) > 1:
        pts, ok = project(trail[-400:], R_wc, c_w)
        seg = [tuple(p) for p, k in zip(pts, ok) if k]
        if len(seg) > 1: d.line(seg, fill=(255, 70, 50), width=4)
    if est:
        ez = gz(*est); ring = [(est[0] + 0.7 * math.cos(a_), est[1] + 0.7 * math.sin(a_), ez + 0.02) for a_ in np.linspace(0, 2 * math.pi, 48)]
        pts, ok = project(ring, R_wc, c_w)
        if ok.all(): d.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=(0, 220, 255), width=4)
    # map landmarks (under the robot too) near the robot, where the splat does not hide them
    near = np.linalg.norm(LM_XYZ[:, :2] - [x, y], axis=1) < 12
    pts, ok = project(LM_XYZ[near], R_wc, c_w)
    zc = ((LM_XYZ[near] - c_w) @ R_wc)[:, 2]
    for (u, v), good, z in zip(pts, ok, zc):
        if good and 0 <= u < W and 0 <= v < H and (depth[int(v), int(u)] <= 0 or z < depth[int(v), int(u)] + 0.3):
            d.point((u, v), fill=(225, 225, 225))
    rgb = np.asarray(img, float)
    if CROWD:
        people_at(t)
        for px_, py_ in CROWD.xy[np.hypot(CROWD.xy[:, 0] - c_w[0], CROWD.xy[:, 1] - c_w[1]) > 2.0]:  # contact shadows
            ring = [(px_ + 0.28 * math.cos(a_), py_ + 0.28 * math.sin(a_), gz(px_, py_) + 0.01) for a_ in np.linspace(0, 2 * math.pi, 24)]
            pts, ok = project(ring, R_wc, c_w)
            if ok.all():
                sm = Image.new('L', (W, H), 0); ImageDraw.Draw(sm).polygon([tuple(q) for q in pts], fill=120)
                a = np.asarray(sm.filter(ImageFilter.GaussianBlur(5)), float) / 255
                if SPLAT:   # the person is in the render already: shade only the ground behind their feet
                    pz = ((np.array([px_, py_, gz(px_, py_)]) - c_w) @ R_wc)[2]
                    a = a * ((depth <= 0) | (depth > pz - 0.35))
                rgb = rgb * (1 - a[..., None])
        if SPLAT:
            rgb = robot_model.draw(ROBOT, rgb, depth, (x, y, yaw), z0, R_wc, c_w, FX)
        else:
            dist = np.hypot(CROWD.xy[:, 0] - c_w[0], CROWD.xy[:, 1] - c_w[1])
            far = dist > math.hypot(x - c_w[0], y - c_w[1])
            near = ~far & (dist > 2.0)               # not those brushing past the chase camera itself
            rgb = CROWD.draw(rgb, depth, R_wc, c_w, FX, gz, only=far)
            rgb = robot_model.draw(ROBOT, rgb, depth, (x, y, yaw), z0, R_wc, c_w, FX)
            rgb = CROWD.draw(rgb, depth, R_wc, c_w, FX, gz, only=near)
    else:
        rgb = robot_model.draw(ROBOT, rgb, depth, (x, y, yaw), z0, R_wc, c_w, FX)
    img = Image.fromarray(rgb.astype(np.uint8)); d = ImageDraw.Draw(img)
    # the latest localization (its PnP inliers), if recent
    k = bisect.bisect_right(FT, int(round(t * 1e9))) - 1
    since = t - FT[k] * 1e-9 if k >= 0 else None
    fix = FIX[FT[k]] if since is not None and since < 0.8 else None
    # without a fresh fix, the last one's landmarks stay, greyed out
    shown = fix if fix else (FIX[FT[k]] if since is not None else None)
    line_col, dot_col, dot_edge = ((70, 255, 130), (70, 255, 130), (0, 90, 30)) if fix else \
        ((150, 150, 150), (170, 170, 170), (70, 70, 70))
    cam_top = np.array([x, y, z0 + scene["camera_height"]])
    if shown:
        ids = [i for _, _, i in shown if i in LM]
        P = np.array([LM[i] for i in ids]); pts, ok = project(np.vstack([cam_top, P]), R_wc, c_w)
        if ok[0]:
            for (u, v), good in zip(pts[1:], ok[1:]):
                if good: d.line((tuple(pts[0]), (u, v)), fill=line_col, width=1)
            for (u, v), good in zip(pts[1:], ok[1:]):
                if good: d.ellipse((u - 4, v - 4, u + 4, v + 4), fill=dot_col, outline=dot_edge)
    # inset: the robot's own camera, at the localized image's time, with its matched features
    tf = FT[k] * 1e-9 if fix else t
    rr = sim[min(bisect.bisect_left(ST, tf), len(sim) - 1)]
    fx_, fy_, fyaw = float(rr['true_x']), float(rr['true_y']), float(rr['true_yaw'])
    fc, fs = math.cos(fyaw), math.sin(fyaw)
    R_fw = np.array([[fs, 0, fc], [-fc, 0, fs], [0, -1, 0]], float).T  # camera <- world
    cf = np.array([fx_ + math.cos(fyaw) * scene.get('camera_forward', 0.0), fy_ + math.sin(fyaw) * scene.get('camera_forward', 0.0), gz(fx_, fy_) + scene['camera_height']])
    if SPLAT:
        people_at(tf)
        fb = ask(fpv, R_fw, -R_fw @ cf, CW * CH * 7, CROWD.splats(cf, gz))
        inset = Image.fromarray(np.frombuffer(fb[:CW * CH * 3], np.uint8).reshape(CH, CW, 3)).convert('L').convert('RGB')
    elif CROWD:
        fb = ask(fpv, R_fw, -R_fw @ cf, CW * CH * 7)
        frgb = np.frombuffer(fb[:CW * CH * 3], np.uint8).reshape(CH, CW, 3).astype(float)
        people_at(tf)
        frgb = CROWD.draw(frgb, np.frombuffer(fb[CW * CH * 3:], '<f4').reshape(CH, CW), R_fw.T, cf,
                          float(CAM["model"].split()[3]), gz, fast=True)
        inset = Image.fromarray(np.clip(frgb, 0, 255).astype(np.uint8)).convert('L').convert('RGB')
    else:
        inset = Image.frombytes('RGB', (CW, CH), ask(fpv, R_fw, -R_fw @ cf, CW * CH * 3)).convert('L').convert('RGB')
    sx = sy = 320 / CW
    inset = inset.resize((320, int(320 * CH / CW)))
    idr = ImageDraw.Draw(inset)
    if fix:
        for px_, py_, _ in fix:
            u, v = px_ * sx, py_ * sy
            idr.ellipse((u - 3, v - 3, u + 3, v + 3), outline=(70, 255, 130), width=2)
    img.paste(inset, (W - 336, 16)); d.rectangle((W - 337, 15, W - 15, 17 + inset.height), outline=(255, 255, 255), width=2)
    if fix:
        label, col = f"camera: {len(fix)} features matched", (255, 255, 255)
    elif since is not None:
        label, col = f"no map match for {since:.1f} s:\non wheel odometry until the next", (255, 170, 40)
    else:
        label, col = "robot's camera", (255, 255, 255)
    d.text((W - 336, 22 + inset.height), label, font=F(18), fill=col, stroke_width=2, stroke_fill=(0, 0, 0))
    # minimap
    m = Image.new('RGB', (MW, MH), (25, 28, 36)); md = ImageDraw.Draw(m)
    md.line([mto(*p) for p in path], fill=(90, 96, 110), width=8)
    if len(trail) > 1: md.line([mto(p[0], p[1]) for p in trail], fill=(255, 70, 50), width=3)
    for gi, (gx_, gy_, _) in enumerate(goals):
        px, py = mto(gx_, gy_); col = (70, 200, 100) if t >= done_t[gi] else (250, 170, 30)
        md.ellipse((px - 8, py - 8, px + 8, py + 8), fill=col, outline=(255, 255, 255), width=2)
    px, py = mto(x, y); md.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 70, 50), outline=(255, 255, 255))
    img.paste(m, (16, H - MH - 16)); d.rectangle((15, H - MH - 17, 16 + MW, H - 15), outline=(255, 255, 255), width=2)
    reached = sum(t >= dt for dt in done_t)
    d.text((24, 16), f'goal {min(reached + 1, len(goals))}/{len(goals)}' if reached < len(goals) else f'all {len(goals)} goals reached',
           font=F(30), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    d.text((24, 54), 'green: map features visloc matched (grey: last match)   cyan ring: visloc estimate   red: path', font=F(20), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(f'{OUT}/{n:05d}.jpg', quality=90); n += 1
srv.stdin.close(); fpv.stdin.close()
print(n, 'frames')
