#!/usr/bin/env python3
"""Generate the README hero GIF: one continuous orbit camera circling a
`gsplat_photos` reconstruction from above/outside, showing (1) the sparse SfM
point cloud + recovered camera frustums popping in, (2) a dissolve into the
photoreal 3D Gaussian splat rendered with visloc-rs's own Rust/wgpu renderer,
and (3) a dissolve into the extracted, shaded mesh, looping back smoothly.

Why this exists: a flythrough at the original (ground-level) photo poses was
rejected because it "just looks like normal photos" -- nothing in frame told
the viewer this was a 3D reconstruction rather than a photo slideshow. This
script instead orbits the camera around a viewpoint NO input photo has (an
elevated ring around the whole scene), and visibly grows/dissolves through
the three pipeline stages, so the 3D-ness is obvious at a glance.

Inputs: the `--src` work directory written by `gsplat_photos`:
  <src>/{scene.ply, mesh.ply, sparse/0/{cameras.txt,images.txt,points3D.txt}}

Pipeline:
  - Points + camera frustums (phase 1) and the shaded mesh (phase 3) are
    rasterized with a small headless OpenGL renderer (moderngl; EGL when no
    display is available), reusing the exact same per-frame camera
    intrinsics/poses as the splat phase so the crossfades line up.
  - The Gaussian splat (phase 2) -- the project's actual accuracy/speed claim
    -- is rendered by the project's own Rust `gsplat_eval` binary, which is
    the same renderer used elsewhere in the README. A copy of scene.ply is
    used, filtered with a KD-tree so only gaussians within a couple of the
    SfM point cloud's own nearest-neighbour spacings of some real triangulated
    point survive (plus a roofline height cap, a size cap relative to the
    scene and a large+low-opacity drop);
    this is what keeps the elevated orbit view -- which the ground-level
    training photos never saw -- from washing out in sky/ground floaters. The
    mesh is similarly cropped to the SfM point-cloud bounding box.
  - The orbit is fitted to the training cameras: "up" is the normal of the
    plane the camera centres lie in, the orbit radius is a multiple of the
    camera ring radius, and every world-space margin is a fraction of that
    radius, so the same settings work at any (arbitrary) SfM scale.
  - Frames are alpha-crossfaded at the phase boundaries, labelled, and
    assembled into a palette-quantized GIF with ffmpeg.

Tanks and Temples Courthouse (a whole town square; GPU run on Google Colab):
scripts/colab/readme_hero_courthouse.ipynb runs every step below.

    pip install numpy scipy pillow moderngl plyfile
    cargo build --release -p visloc-gsplat-train --features gpu,euroc \
        --example gsplat_photos --example gsplat_eval
    target/release/examples/gsplat_photos --images <Courthouse every 2nd frame> \
        --out runs/courthouse --max-size 1280 --steps 30000 --normal-weight 0.005
    python scripts/make_readme_hero.py --src runs/courthouse \
        --work runs/courthouse/hero --scene-label "Tanks and Temples Courthouse" \
        --elev 40 --radius-mult 2.6 --target-height 0.25 --zoom 1.15 \
        --out docs/assets/hero_reconstruction.gif

The previous south-building hero was made the same way from
`gsplat_photos --images <south-building>/images --out runs/sb --max-size 1024
--steps 30000 --normal-weight 0.005` with `--src runs/sb --elev 18
--radius-mult 1.35 --width 560`.
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

# ---------------------------------------------------------------------------
# Config (timeline; scene-dependent settings are command-line flags)
# ---------------------------------------------------------------------------

FPS = 10
N_FRAMES = 80             # one full revolution, 8s at 10fps (native frame rate;
                          # no frame-drop/stride trick -- 10fps was chosen over
                          # decimating a 12fps render, which looked choppy)

P1_END = 24               # 0..P1_END-1: pure points+frustums (2.4s)
XF1 = 6                   # crossfade length at the phase1->2 boundary (0.6s)
P2_START = P1_END + XF1   # 30
P2_END = 58                # pure splat through here, then crossfade (2.8s pure)
XF2 = 6                    # 0.6s
P3_START = P2_END + XF2   # 64 .. pure mesh 64..79 (1.6s)

REVEAL_FRAMES = 20.0      # points/frustums finish popping in by this frame
REVEAL_START_FRAC = 0.18  # fraction of points/frustums already visible at frame 0
                          # (no empty black opening frames)

# Gaussian-splat floater filtering: keep a gaussian only if it is within
# DIST_MULT x the SfM point cloud's own median nearest-neighbour spacing of
# some SfM point (kills anything not actually near real triangulated
# geometry -- this is what removes the big sky/ground floater smears), is
# not above the highest SfM point (+ a margin) along the scene's up axis,
# and is not simultaneously large *and* low-opacity.
FILTER_DIST_MULT = 2.5
FILTER_MAX_SCALE_FRAC = 0.05   # drop any gaussian whose largest std exceeds this x ring radius
FILTER_LARGE_SCALE_FRAC = 0.015  # ... and "large" ones (std above this x ring radius)
FILTER_LOW_OPACITY_RAW = 0.0     # ... that are also low-opacity (raw logit < 0, i.e. alpha < 0.5)

# World-space margins as fractions of the training-camera ring radius.
UP_MARGIN_FRAC = 0.035     # splat roofline cap / mesh crop above the top SfM point
DOWN_MARGIN_FRAC = 0.055   # mesh crop below the lowest SfM point
SIDE_MARGIN_FRAC = 0.04    # mesh crop beyond the SfM point cloud's horizontal extent
FRUSTUM_FRAC = 0.016       # camera-frustum glyph size
NEAR_FRAC, FAR_FRAC = 0.005, 30.0

FONT_BOLD = ['C:/Windows/Fonts/segoeuib.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf']
FONT_REG = ['C:/Windows/Fonts/segoeui.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']


# ---------------------------------------------------------------------------
# Geometry: COLMAP text I/O, building frame fit, orbit path
# ---------------------------------------------------------------------------

def load_images_txt(path):
    lines = [l.split() for l in open(path) if l.strip() and not l.startswith('#')]
    lines = [l for l in lines if len(l) == 10]
    lines.sort(key=lambda l: int(l[0]))
    q = np.array([[float(x) for x in l[1:5]] for l in lines])
    t = np.array([[float(x) for x in l[5:8]] for l in lines])
    return q, t


def load_camera_txt(path):
    """(width, height, fx, fy, cx, cy) of the first camera in a COLMAP
    cameras.txt (PINHOLE, or SIMPLE_PINHOLE / *_RADIAL with distortion
    ignored)."""
    for line in open(path):
        if not line.strip() or line.startswith('#'):
            continue
        p = line.split()
        model, w, h = p[1], int(p[2]), int(p[3])
        v = [float(x) for x in p[4:]]
        if model == 'PINHOLE' or model == 'OPENCV':
            return w, h, v[0], v[1], v[2], v[3]
        return w, h, v[0], v[0], v[1], v[2]
    raise ValueError(f'no camera in {path}')


def quat_to_R(q):
    from scipy.spatial.transform import Rotation as Rt
    return Rt.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def R_to_quat(R):
    from scipy.spatial.transform import Rotation as Rt
    q = Rt.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def camera_centers(q, t):
    R = quat_to_R(q)
    C = -np.einsum('nji,nj->ni', R, t)
    return C, R


def load_points3d(path):
    pos, col, first_img = [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
            col.append([int(parts[4]), int(parts[5]), int(parts[6])])
            track = parts[8:]
            img_ids = [int(track[i]) for i in range(0, len(track), 2)]
            first_img.append(min(img_ids) if img_ids else 10 ** 9)
    return np.array(pos), np.array(col, dtype=np.uint8), np.array(first_img)


def build_basis(up):
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, up)) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    e1 = a - up * np.dot(a, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2


def fit_building_frame(q, t):
    C, R = camera_centers(q, t)
    up_cam_est = -R[:, 1, :]
    centroid = C.mean(axis=0)
    Cc = C - centroid
    cov = Cc.T @ Cc
    w, v = np.linalg.eigh(cov)
    up = v[:, 0]
    if np.dot(up, up_cam_est.mean(axis=0)) < 0:
        up = -up
    up /= np.linalg.norm(up)
    rel = C - centroid
    rel_h = rel - np.outer(rel @ up, up)
    ring_radius = np.linalg.norm(rel_h, axis=1).mean()
    return centroid, up, ring_radius, C


def orbit_path(center, up, radius, elevation_deg, n_frames, phase0_deg, target):
    e1, e2 = build_basis(up)
    height = radius * np.tan(np.radians(elevation_deg))
    angles = np.radians(phase0_deg) + np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    Cs = np.array([center + radius * (np.cos(a) * e1 + np.sin(a) * e2) + height * up for a in angles])
    targets = np.tile(target, (n_frames, 1))
    return Cs, targets, angles


def poses_from_lookat(Cs, targets, up):
    n = len(Cs)
    Rs = np.zeros((n, 3, 3))
    ts = np.zeros((n, 3))
    for i in range(n):
        z = targets[i] - Cs[i]
        z /= np.linalg.norm(z)
        y = -up
        y = y - z * np.dot(y, z)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        Rcw = np.stack([x, y, z])
        Rs[i] = Rcw
        ts[i] = -Rcw @ Cs[i]
    return Rs, ts


def write_colmap_pose_dir(out_dir, Rs, ts, indices, cam_line, dummy_image):
    os.makedirs(f'{out_dir}/sparse/0', exist_ok=True)
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    with open(f'{out_dir}/sparse/0/cameras.txt', 'w') as f:
        f.write('# Camera list\n' + cam_line + '\n')
    open(f'{out_dir}/sparse/0/points3D.txt', 'w').write('# 3D point list\n0 0 0 0 128 128 128 0\n')
    with open(f'{out_dir}/sparse/0/images.txt', 'w') as fo:
        fo.write('# Image list\n')
        for j, i in enumerate(indices):
            qq = R_to_quat(Rs[i])
            name = f'f{i:04d}.png'
            fo.write(f'{j+1} {qq[0]} {qq[1]} {qq[2]} {qq[3]} {ts[i][0]} {ts[i][1]} {ts[i][2]} 1 {name}\n\n')
            p = f'{out_dir}/images/{name}'
            if not os.path.exists(p):
                try:
                    os.link(dummy_image, p)
                except OSError:
                    shutil.copy(dummy_image, p)


# ---------------------------------------------------------------------------
# Headless OpenGL rendering (points+frustums, mesh) via moderngl
# ---------------------------------------------------------------------------

def cv_to_gl_Rt(Rcw, tcw):
    flip = np.diag([1.0, -1.0, -1.0])
    return flip @ Rcw, flip @ tcw


def gl_view_matrix(Rcw, tcw):
    Rgl, tgl = cv_to_gl_Rt(Rcw, tcw)
    V = np.eye(4)
    V[:3, :3] = Rgl
    V[:3, 3] = tgl
    return V


def gl_projection_matrix(fx, fy, cx, cy, w, h, near, far):
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[0, 2] = 1.0 - 2.0 * cx / w
    P[1, 1] = 2.0 * fy / h
    P[1, 2] = 2.0 * cy / h - 1.0
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


class GLRenderer:
    def __init__(self, w, h, bg=(0.043, 0.047, 0.06, 1.0)):
        import moderngl
        self.moderngl = moderngl
        self.w, self.h = w, h
        try:
            self.ctx = moderngl.create_context(standalone=True, require=330)
        except Exception:
            # Headless Linux (Colab, CI): no X display, use EGL instead.
            self.ctx = moderngl.create_context(standalone=True, require=330, backend='egl')
        print(f'OpenGL: {self.ctx.info.get("GL_RENDERER")}', flush=True)
        self.bg = bg
        self.fbo = self.ctx.simple_framebuffer((w, h), components=4, samples=0)
        self.fbo.use()
        self.pt_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 mvp; uniform float point_size;
                in vec3 in_pos; in vec3 in_color; out vec3 v_color;
                void main() { gl_Position = mvp*vec4(in_pos,1.0); gl_PointSize = point_size; v_color = in_color; }
            ''',
            fragment_shader='''
                #version 330
                uniform float alpha_mult;
                in vec3 v_color; out vec4 f_color;
                void main() {
                    vec2 d = gl_PointCoord - vec2(0.5);
                    float r2 = dot(d, d);
                    if (r2 > 0.25) discard;
                    float a = 1.0 - smoothstep(0.0, 0.25, r2);
                    f_color = vec4(v_color, a * alpha_mult);
                }
            ''')
        self.line_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 mvp; in vec3 in_pos; in vec3 in_color; out vec3 v_color;
                void main() { gl_Position = mvp*vec4(in_pos,1.0); v_color = in_color; }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color; out vec4 f_color; void main() { f_color = vec4(v_color,1.0); }
            ''')
        self.mesh_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 mvp; uniform vec3 light_dir;
                in vec3 in_pos; in vec3 in_normal; in vec3 in_color; out vec3 v_color;
                void main() {
                    gl_Position = mvp*vec4(in_pos,1.0);
                    float ndl = max(dot(normalize(in_normal), -light_dir), 0.0);
                    float amb = 0.42;
                    v_color = in_color * (amb + (1.0-amb)*ndl);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color; out vec4 f_color; void main() { f_color = vec4(v_color,1.0); }
            ''')

    def clear(self):
        self.fbo.use()
        self.ctx.enable(self.moderngl.DEPTH_TEST)
        self.ctx.depth_func = '<='  # so a same-depth glow+core point pair both pass
        self.ctx.disable(self.moderngl.BLEND)
        self.fbo.clear(*self.bg)

    def read(self):
        data = self.fbo.read(components=4, alignment=1)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(self.h, self.w, 4)
        return np.flipud(arr)

    def draw_points(self, pos, color01, mvp, point_size=7.0, alpha=1.0, additive=False):
        mgl = self.moderngl
        vbo = self.ctx.buffer(np.hstack([pos, color01]).astype('f4').tobytes())
        vao = self.ctx.vertex_array(self.pt_prog, [(vbo, '3f 3f', 'in_pos', 'in_color')])
        self.ctx.enable(mgl.PROGRAM_POINT_SIZE)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE) if additive else (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        self.pt_prog['mvp'].write(np.ascontiguousarray(mvp.T).astype('f4').tobytes())
        self.pt_prog['point_size'].value = point_size
        self.pt_prog['alpha_mult'].value = alpha
        vao.render(mgl.POINTS)
        self.ctx.disable(mgl.BLEND)
        vao.release(); vbo.release()

    def draw_points_glow(self, pos, color01, mvp, core_size=7.0, glow_size=20.0, glow_alpha=0.22):
        """Bright core disc + a larger, dim, additively-blended glow halo, so
        the sparse point cloud reads clearly against the dark background."""
        self.draw_points(pos, color01, mvp, point_size=glow_size, alpha=glow_alpha, additive=True)
        self.draw_points(pos, color01, mvp, point_size=core_size, alpha=1.0, additive=False)

    def draw_lines(self, pos, color01, mvp):
        mgl = self.moderngl
        vbo = self.ctx.buffer(np.hstack([pos, color01]).astype('f4').tobytes())
        vao = self.ctx.vertex_array(self.line_prog, [(vbo, '3f 3f', 'in_pos', 'in_color')])
        self.line_prog['mvp'].write(np.ascontiguousarray(mvp.T).astype('f4').tobytes())
        vao.render(mgl.LINES)
        vao.release(); vbo.release()

    def draw_mesh(self, pos, normal, color01, faces_idx, mvp, light_dir):
        mgl = self.moderngl
        vbo = self.ctx.buffer(np.hstack([pos, normal, color01]).astype('f4').tobytes())
        ibo = self.ctx.buffer(faces_idx.astype('i4').tobytes())
        vao = self.ctx.vertex_array(self.mesh_prog, [(vbo, '3f 3f 3f', 'in_pos', 'in_normal', 'in_color')], ibo)
        self.mesh_prog['mvp'].write(np.ascontiguousarray(mvp.T).astype('f4').tobytes())
        ld = np.array(light_dir, dtype='f4'); ld /= np.linalg.norm(ld)
        self.mesh_prog['light_dir'].value = tuple(ld.tolist())
        self.ctx.disable(mgl.PROGRAM_POINT_SIZE)
        vao.render(mgl.TRIANGLES)
        vao.release(); vbo.release(); ibo.release()


def frustum_segments(Ctr, Rtr, indices, scale=0.28, aspect=0.75):
    segs = []
    edges = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    a = aspect
    corners_cam = np.array([[-1, -a, 1.4], [1, -a, 1.4], [1, a, 1.4], [-1, a, 1.4]]) * scale
    for i in indices:
        Rcw = Rtr[i].T
        corners_world = Ctr[i] + corners_cam @ Rcw.T
        pts = np.vstack([Ctr[i][None, :], corners_world])
        for a, b in edges:
            segs.append(pts[a]); segs.append(pts[b])
    return np.array(segs) if segs else np.zeros((0, 3))


# ---------------------------------------------------------------------------
# Gaussian-splat PLY filtering (drop likely-floater gaussians for the
# elevated orbit view the ground-level training photos never covered)
# ---------------------------------------------------------------------------

def read_ply_header(f):
    header = b''
    while True:
        line = f.readline()
        header += line
        if line.strip() == b'end_header':
            break
    return header


def ply_float_columns(header):
    """Column index of every `property float <name>` of a 3DGS PLY header."""
    names = [l.split()[2].decode() for l in header.split(b'\n')
             if l.startswith(b'property float')]
    return {n: i for i, n in enumerate(names)}


def filter_scene_ply(src_ply, out_ply, pos_pts, center, up, dist_mult, up_margin,
                      max_scale, large_scale, low_opacity_raw):
    """Keep a gaussian only if it is close to some real SfM point (kills the
    sky/ground floater smears an elevated, never-photographed viewpoint would
    otherwise expose), is not above the reconstructed roofline, is not huge
    (world-space std above `max_scale`), and is not simultaneously large
    (above `large_scale`) *and* low-opacity."""
    from scipy.spatial import cKDTree

    with open(src_ply, 'rb') as f:
        header = read_ply_header(f)
        data = np.fromfile(f, dtype='<f4')
    col = ply_float_columns(header)
    arr = data.reshape(-1, len(col))
    pos = arr[:, 0:3]

    # Isolated SfM points (sky, reflections, far-away mismatches) are not
    # support: drop points whose 8 nearest neighbours are unusually far.
    tree = cKDTree(pos_pts)
    d_knn, _ = tree.query(pos_pts, k=9, workers=-1)
    spread = d_knn[:, 1:].mean(1)
    support = pos_pts[spread < 3.0 * np.median(spread)]
    tree = cKDTree(support)
    med_spacing = np.median(d_knn[:, 1])
    dist_thr = dist_mult * med_spacing
    d_near, _ = tree.query(pos, k=1, workers=-1)
    keep = d_near < dist_thr

    top = np.percentile((support - center) @ up, 99.5)
    keep &= ((pos - center) @ up) < (top + up_margin)

    scale_cols = [col['scale_0'], col['scale_1'], col['scale_2']]
    max_std = np.exp(arr[:, scale_cols].max(1))
    opacity_raw = arr[:, col['opacity']]
    keep &= max_std < max_scale
    keep &= ~((max_std > large_scale) & (opacity_raw < low_opacity_raw))

    kept = arr[keep]
    lines = header.split(b'\n')
    out_lines = [f'element vertex {kept.shape[0]}'.encode() if l.startswith(b'element vertex') else l
                 for l in lines]
    with open(out_ply, 'wb') as f:
        f.write(b'\n'.join(out_lines))
        kept.astype('<f4').tofile(f)
    return kept.shape[0], arr.shape[0], dist_thr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def default_gsplat_eval():
    exe = 'gsplat_eval.exe' if os.name == 'nt' else 'gsplat_eval'
    return os.path.join('target', 'release', 'examples', exe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True,
                    help='gsplat_photos work dir (scene.ply, mesh.ply, sparse/0/)')
    ap.add_argument('--work', default=None, help='frame scratch dir (default: <src>/hero)')
    ap.add_argument('--gsplat-eval', default=default_gsplat_eval())
    ap.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    ap.add_argument('--out', default='docs/assets/hero_reconstruction.gif')
    ap.add_argument('--scene-label', default='',
                    help='dataset name shown in the footer, e.g. "Tanks and Temples Courthouse"')
    ap.add_argument('--input-images', type=int, default=0,
                    help='number of input photos, if more than were registered (label only)')
    ap.add_argument('--render-width', type=int, default=1024,
                    help='render resolution (long side follows the training camera aspect)')
    ap.add_argument('--zoom', type=float, default=1.0,
                    help='focal-length multiplier on the training camera for the orbit view')
    ap.add_argument('--elev', type=float, default=18.0,
                    help='orbit elevation (deg) above the horizontal camera ring')
    ap.add_argument('--radius-mult', type=float, default=1.35,
                    help='orbit radius as a multiple of the training-camera ring radius')
    ap.add_argument('--center', choices=['points', 'cameras'], default='points',
                    help='orbit centre: SfM points inside the camera ring, or the camera centroid')
    ap.add_argument('--phase0', type=float, default=45.0, help='orbit start angle (deg)')
    ap.add_argument('--target-height', type=float, default=0.5,
                    help='look-at height as a fraction between the low and high SfM point percentiles')
    ap.add_argument('--point-size', type=float, default=8.0, help='SfM point core size (px)')
    ap.add_argument('--max-points', type=int, default=400000,
                    help='randomly subsample the SfM cloud above this many points (phase 1 only)')
    ap.add_argument('--skip-render', action='store_true', help='reuse frames already in --work')
    ap.add_argument('--width', type=int, default=640, help='final GIF width in pixels')
    ap.add_argument('--colors', type=int, default=100)
    ap.add_argument('--dither', default='none')
    ap.add_argument('--stride', type=int, default=1,
                     help='keep every Nth composited frame (GIF size vs. smoothness); '
                          'output fps is FPS/stride so total duration is unchanged. '
                          'Prefer lowering --width over raising this -- dropping frames '
                          'from an already-smooth render looks choppier than it needs to.')
    ap.add_argument('--skip-composite', action='store_true', help='reuse final_*.png, only re-run ffmpeg')
    ap.add_argument('--stages', default='points,mesh,splat',
                     help='comma list of render stages to (re)run, e.g. --stages points '
                          'to only re-render phase 1 after tweaking its look')
    args = ap.parse_args()
    stages = set(args.stages.split(','))

    src = args.src
    work = args.work or os.path.join(src, 'hero')
    os.makedirs(work, exist_ok=True)

    tw, th, tfx, tfy, tcx, tcy = load_camera_txt(f'{src}/sparse/0/cameras.txt')
    s = args.render_width / tw
    W, H = args.render_width, int(round(th * s))
    FX, FY, CX, CY = tfx * s * args.zoom, tfy * s * args.zoom, tcx * s, tcy * s

    q, t = load_images_txt(f'{src}/sparse/0/images.txt')
    center, up, ring_radius, Ctr = fit_building_frame(q, t)
    _, Rtr = camera_centers(q, t)
    pos_pts, col_pts, first_img = load_points3d(f'{src}/sparse/0/points3D.txt')
    if (first_img > len(q)).all():
        # gsplat_photos's points3D.txt carries no per-point track/image list
        # (unlike a full COLMAP export), so there is no real "first image
        # that saw this point" to grow the cloud by. Approximate it with the
        # nearest training camera instead, so points still pop in roughly in
        # image/capture order together with that camera's frustum.
        from scipy.spatial import cKDTree
        _, nearest_cam = cKDTree(Ctr).query(pos_pts, k=1, workers=-1)
        first_img = nearest_cam + 1
    e1, e2 = build_basis(up)
    L = ring_radius
    if args.center == 'points':
        # Captures that walk unevenly around the subject bias the camera
        # centroid; centre the orbit on the SfM points inside the camera ring
        # instead (horizontally; the height stays on the camera plane).
        rel = pos_pts - center
        rel_h = rel - np.outer(rel @ up, up)
        inner = np.linalg.norm(rel_h, axis=1) < L
        if inner.sum() > 100:
            center = center + np.median(rel_h[inner], axis=0)
    print(f'{len(q)} cameras, {len(pos_pts)} points, ring radius {L:.3f}, render {W}x{H}', flush=True)

    hh = (pos_pts - center) @ up
    h1 = (pos_pts - center) @ e1
    h2 = (pos_pts - center) @ e2
    lo_h, hi_h = np.percentile(hh, 1), np.percentile(hh, 99)
    lo_1, hi_1 = np.percentile(h1, 1), np.percentile(h1, 99)
    lo_2, hi_2 = np.percentile(h2, 1), np.percentile(h2, 99)
    bounds = (lo_h - DOWN_MARGIN_FRAC * L, hi_h + UP_MARGIN_FRAC * L, lo_1, hi_1, lo_2, hi_2)
    target = center + up * (lo_h + (hi_h - lo_h) * args.target_height)

    radius = L * args.radius_mult
    Cs, targets, angles = orbit_path(center, up, radius, args.elev, N_FRAMES, args.phase0, target)
    Rs, ts = poses_from_lookat(Cs, targets, up)
    P = gl_projection_matrix(FX, FY, CX, CY, W, H, NEAR_FRAC * L, FAR_FRAC * L)

    n_train_images = len(q)
    n_points = len(pos_pts)

    if not args.skip_render and ({'points', 'mesh'} & stages):
        renderer = GLRenderer(W, H)

    if not args.skip_render and 'points' in stages:
        # ---- phase 1: points + frustums, frames 0..P2_START-1 (through the
        # crossfade-in, so the blend at the boundary has real content) ----
        sel = np.arange(n_points)
        if n_points > args.max_points:
            sel = np.random.default_rng(0).choice(n_points, args.max_points, replace=False)
        ppos, pcol, pfirst = pos_pts[sel].astype(np.float32), col_pts[sel].astype(np.float32) / 255.0, first_img[sel]
        for k in range(0, P2_START):
            reveal = REVEAL_START_FRAC + (1.0 - REVEAL_START_FRAC) * min(1.0, k / REVEAL_FRAMES)
            cutoff = reveal * n_train_images
            pmask = pfirst <= cutoff
            fmask_idx = [i for i in range(n_train_images) if i + 1 <= cutoff]
            V = gl_view_matrix(Rs[k], ts[k]); mvp = P @ V
            renderer.clear()
            if pmask.any():
                renderer.draw_points_glow(ppos[pmask], pcol[pmask], mvp,
                                           core_size=args.point_size, glow_size=args.point_size * 1.6,
                                           glow_alpha=0.16)
            if fmask_idx:
                segs = frustum_segments(Ctr, Rtr, fmask_idx, scale=FRUSTUM_FRAC * L,
                                        aspect=H / W).astype(np.float32)
                if len(segs):
                    accent = np.tile(np.array([1.0, 0.55, 0.12], dtype=np.float32), (segs.shape[0], 1))
                    renderer.draw_lines(segs, accent, mvp)
            img = renderer.read()
            Image.fromarray(img, 'RGBA').convert('RGB').save(f'{work}/pts_{k:04d}.png')
        print('points+frustums done', flush=True)

    if not args.skip_render and 'mesh' in stages:
        # ---- phase 3: mesh, frames P2_END..N_FRAMES-1 (from the crossfade-out
        # of phase 2 through the end) ----
        mpos = np.load(f'{work}/mesh_pos.npy') if os.path.exists(f'{work}/mesh_pos.npy') else None
        if mpos is None:
            from plyfile import PlyData
            ply = PlyData.read(f'{src}/mesh.ply')
            v = ply['vertex']
            mpos = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
            mcol = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.float32) / 255.0
            mfaces = np.vstack(ply['face']['vertex_indices']).astype(np.int32)
            v0, v1, v2 = mpos[mfaces[:, 0]], mpos[mfaces[:, 1]], mpos[mfaces[:, 2]]
            fn = np.cross(v1 - v0, v2 - v0)
            mnorm = np.zeros_like(mpos)
            np.add.at(mnorm, mfaces[:, 0], fn); np.add.at(mnorm, mfaces[:, 1], fn); np.add.at(mnorm, mfaces[:, 2], fn)
            norm_len = np.linalg.norm(mnorm, axis=1, keepdims=True); norm_len[norm_len < 1e-12] = 1.0
            mnorm = (mnorm / norm_len).astype(np.float32)
            np.save(f'{work}/mesh_pos.npy', mpos); np.save(f'{work}/mesh_col.npy', mcol)
            np.save(f'{work}/mesh_normal.npy', mnorm); np.save(f'{work}/mesh_faces.npy', mfaces)
        else:
            mcol = np.load(f'{work}/mesh_col.npy'); mnorm = np.load(f'{work}/mesh_normal.npy')
            mfaces = np.load(f'{work}/mesh_faces.npy')

        side = SIDE_MARGIN_FRAC * L
        mh = (mpos - center) @ up; m1 = (mpos - center) @ e1; m2 = (mpos - center) @ e2
        inside = ((mh > bounds[0]) & (mh < bounds[1]) & (m1 > bounds[2] - side) & (m1 < bounds[3] + side) &
                  (m2 > bounds[4] - side) & (m2 < bounds[5] + side))
        face_ok = inside[mfaces[:, 0]] & inside[mfaces[:, 1]] & inside[mfaces[:, 2]]
        mfaces_c = mfaces[face_ok]

        for k in range(P2_END, N_FRAMES):
            V = gl_view_matrix(Rs[k], ts[k]); mvp = P @ V
            view_dir = targets[k] - Cs[k]; view_dir = view_dir / np.linalg.norm(view_dir)
            renderer.clear()
            renderer.draw_mesh(mpos, mnorm, mcol, mfaces_c, mvp, light_dir=tuple(view_dir))
            img = renderer.read()
            Image.fromarray(img, 'RGBA').convert('RGB').save(f'{work}/mesh_{k:04d}.png')
        print('mesh done', flush=True)

    if not args.skip_render and 'splat' in stages:
        # ---- phase 2: the Gaussian splat, rendered with the project's own
        # Rust wgpu renderer, frames P1_END..P2_END+XF2-1 ----
        filtered_ply = f'{work}/scene_filtered.ply'
        kept, total, dist_thr = filter_scene_ply(f'{src}/scene.ply', filtered_ply, pos_pts, center, up,
                                                  FILTER_DIST_MULT, UP_MARGIN_FRAC * L,
                                                  FILTER_MAX_SCALE_FRAC * L, FILTER_LARGE_SCALE_FRAC * L,
                                                  FILTER_LOW_OPACITY_RAW)
        print(f'splat filter: kept {kept}/{total} gaussians (dist_thr={dist_thr:.4f})', flush=True)

        splat_dir = f'{work}/splat_poses'
        splat_indices = list(range(P1_END, P2_END + XF2))
        cam_line = f'1 PINHOLE {W} {H} {FX} {FY} {CX} {CY}'
        # gsplat_eval scores each render against an image of the same size;
        # the score is meaningless here, so feed it a black placeholder.
        dummy = f'{work}/placeholder_{W}x{H}.png'
        if not os.path.exists(dummy):
            Image.new('RGB', (W, H)).save(dummy)
        if os.path.isdir(f'{splat_dir}/images'):
            shutil.rmtree(f'{splat_dir}/images')
        write_colmap_pose_dir(splat_dir, Rs, ts, splat_indices, cam_line, dummy)
        splat_out = f'{work}/splat_out'
        subprocess.run([args.gsplat_eval, '--ply', filtered_ply, '--data', splat_dir,
                         '--eval-every', '1', '--save-dir', splat_out], check=True)
        for j, k in enumerate(splat_indices):
            shutil.copy(f'{splat_out}/f{k:04d}.png', f'{work}/splat_{k:04d}.png')
        print('splat done', flush=True)

    # ---- composite + encode ----
    if not args.skip_composite:
        composite(work, args, W, H, n_train_images, n_points)
    encode_gif(work, args)


def load_font(candidates, size):
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def composite(work, args, W, H, n_train_images, n_points):
    font_title = load_font(FONT_BOLD, 26)
    font_sub = load_font(FONT_REG, 16)
    font_foot = load_font(FONT_REG, 14)

    if args.input_images > n_train_images:
        sfm_sub = f'{args.input_images:,} photos, {n_train_images:,} cameras, {n_points:,} points'
    else:
        sfm_sub = f'{n_train_images:,} photos, {n_train_images:,} cameras, {n_points:,} points'
    titles = {
        1: (f'1 · Structure from Motion', sfm_sub),
        2: ('2 · 3D Gaussian Splatting', 'trained scene, rendered with our Rust + wgpu rasterizer'),
        3: ('3 · Mesh', 'extracted from the splat (TSDF + surface nets)'),
    }

    def draw_label(img, phase, alpha):
        if alpha <= 0.01:
            return img
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        title, sub = titles[phase]
        pad = 14
        tw = max(d.textlength(title, font=font_title), d.textlength(sub, font=font_sub))
        box_w, box_h = int(tw + pad * 2), 64
        d.rounded_rectangle([20, 18, 20 + box_w, 18 + box_h], radius=10,
                             fill=(8, 9, 14, int(150 * alpha)))
        d.text((20 + pad, 18 + 10), title, font=font_title, fill=(255, 255, 255, int(255 * alpha)))
        d.text((20 + pad, 18 + 40), sub, font=font_sub, fill=(205, 210, 222, int(230 * alpha)))
        return Image.alpha_composite(img.convert('RGBA'), overlay)

    def draw_footer(img):
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        text = 'visloc-rs · one command · pure Rust + wgpu'
        if args.scene_label:
            text = f'{args.scene_label} · {text}'
        tw = d.textlength(text, font=font_foot)
        x = (img.size[0] - tw) / 2
        y = img.size[1] - 30
        d.rounded_rectangle([x - 10, y - 5, x + tw + 10, y + 20], radius=8, fill=(6, 7, 11, 130))
        d.text((x, y), text, font=font_foot, fill=(225, 227, 235, 235))
        return Image.alpha_composite(img.convert('RGBA'), overlay)

    def load(prefix, k):
        return Image.open(f'{work}/{prefix}_{k:04d}.png').convert('RGBA')

    for k in range(N_FRAMES):
        if k < P1_END:
            base = load('pts', k); label = (1, 1.0)
        elif k < P2_START:
            a = (k - P1_END + 1) / (XF1 + 1)
            base = Image.blend(load('pts', k), load('splat', k), a)
            label = (1, 1 - a) if a < 0.5 else (2, (a - 0.5) * 2)
        elif k < P2_END:
            base = load('splat', k); label = (2, 1.0)
        elif k < P2_END + XF2:
            a = (k - P2_END + 1) / (XF2 + 1)
            base = Image.blend(load('splat', k), load('mesh', k), a)
            label = (2, 1 - a) if a < 0.5 else (3, (a - 0.5) * 2)
        else:
            base = load('mesh', k); label = (3, 1.0)

        base = draw_label(base, label[0], label[1])
        base = draw_footer(base)
        out_w = args.width
        out_h = int(round(H * out_w / W))
        base = base.convert('RGB').resize((out_w, out_h), Image.LANCZOS)
        base.save(f'{work}/final_{k:04d}.png')
    print('composite done', flush=True)


def encode_gif(work, args):
    ffmpeg = args.ffmpeg
    stride = max(1, args.stride)
    out_fps = FPS / stride

    frame_dir = f'{work}/final_%04d.png'
    if stride > 1:
        # GIF size is dominated by frame count far more than by color/dither
        # settings for this kind of high-frequency photoreal content; keep
        # every Nth composited frame and play back proportionally slower so
        # total duration (and the real-time orbit speed) is unchanged.
        dec_dir = f'{work}/decimated'
        os.makedirs(dec_dir, exist_ok=True)
        for f in os.listdir(dec_dir):
            os.remove(os.path.join(dec_dir, f))
        j = 0
        for k in range(0, N_FRAMES, stride):
            shutil.copy(f'{work}/final_{k:04d}.png', f'{dec_dir}/final_{j:04d}.png')
            j += 1
        frame_dir = f'{dec_dir}/final_%04d.png'

    palette = f'{work}/palette.png'
    subprocess.run([ffmpeg, '-y', '-framerate', str(out_fps), '-i', frame_dir,
                     '-vf', f'palettegen=max_colors={args.colors}', '-update', '1', palette], check=True)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    subprocess.run([ffmpeg, '-y', '-framerate', str(out_fps), '-i', frame_dir,
                     '-i', palette, '-lavfi', f'paletteuse=dither={args.dither}', '-loop', '0',
                     out_path], check=True)
    size = os.path.getsize(out_path)
    print(f'wrote {out_path} ({size/1e6:.2f} MB, {out_fps:.1f} fps)', flush=True)


if __name__ == '__main__':
    main()
