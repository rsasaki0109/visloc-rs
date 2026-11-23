#!/usr/bin/env python3
"""A 360-degree video in, a robot simulator out: SfM, 3DGS, map and Nav2 scene (docs/navigation.md).

    python3 ros2/visloc-ros2/gs/real2sim.py --insv walk.insv --out runs/street [--frames 300] [--nav]

From an Insta360 `.insv` (both lenses and the IMU in one file), every step
of docs/navigation.md's street simulator, each skipped when its output
exists (so a run can be resumed or redone from any step by deleting its
directory):

1. `frames/`: the first `--frames` frames, both lenses side by side (ffmpeg).
2. `sfm/`: `gsplat_photos --insv --steps 0 --cpu` (without `--cpu` given
   `--gpu`), the lens faces in metres from the camera's IMU.
3. `level/`: `level_sfm.py`, z up and the ground at 0; its `level.json`
   holds the capture height.
4. `train/`: the faces with their blacked-out pixels (blurred faces, the
   camera's pole) made transparent, then `scene.ply` from
   `gsplat_train --alpha-mask`.
5. `sim/scene.json`: `make_gs_scene.py --terrain`, the robot's camera
   `--camera-height` over the ground, start and goals along the walk.
6. `map/`: the 6 horizontal faces of every frame rendered at the robot's
   camera height (`render_tum_seq.py`), `tum_rgbd_save_map` on them and
   `visual_map_to_occupancy`; `lane_map.py` then writes Nav2's map free only
   within `--lane-half-width` (1 m) of the walk, so the robot keeps to the
   pavement the video was walked on.
7. With `--nav`: `gs_nav.launch.py` and `nav_goals.py` (the tuned
   localization settings, `--dump_inliers`), results in `sim/nav.json`.
   `--pedestrians`, `--cyclists`, `--dog-walkers` and `--strollers` put
   people on the street (`gs_sim.py`, `people.py`).
8. With `--video` (and `--nav`): `chase_video.py` and ffmpeg make
   `chase.mp4`, the drive seen from behind the robot.

Needs the release examples `gsplat_photos`, `gsplat_train`,
`gsplat_render_server`, `tum_rgbd_save_map` and `visual_map_to_occupancy`,
ffmpeg, python3 with numpy, scipy and Pillow, and for `--nav` ROS 2 Jazzy
with Nav2 and `ros2/visloc-ros2`'s release build (ROS_PYTHON=... names the
interpreter that has rclpy, when `python3` does not).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
EX = os.path.join(REPO, 'target', 'release', 'examples')
sys.path.insert(0, HERE)
from align_sfm_tum import qvec2rot, read_images, rot2qvec  # noqa: E402

FACE = 'PINHOLE 640 640 320 320 320 320'
HORIZONTAL = ('front lens front', 'front lens right', 'front lens left',
              'back lens front', 'back lens right', 'back lens left')
TUNED = ['--projection_fresh_s', '60', '--projection_fresh_m', '15', '--projection_radius', '40',
         '--max_hamming', '90', '--reprojection_threshold', '5', '--min_inliers', '20']


def run(cmd, log=None, **kw):
    t0 = time.time()
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    if log:
        with open(log, 'w', encoding='utf-8') as f:
            subprocess.run([str(c) for c in cmd], check=True, stdout=f, stderr=subprocess.STDOUT, **kw)
    else:
        subprocess.run([str(c) for c in cmd], check=True, **kw)
    return time.time() - t0


def done(path):
    return os.path.exists(path)


def mask_faces(src_images, src_model, out):
    """Copy the model and make each face's near-black pixels transparent (RGBA)."""
    from PIL import Image  # noqa: PLC0415
    from scipy import ndimage  # noqa: PLC0415
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'sparse', '0'), exist_ok=True)
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        with open(os.path.join(src_model, name), 'rb') as f, open(os.path.join(out, 'sparse', '0', name), 'wb') as g:
            g.write(f.read())
    for name in sorted(os.listdir(src_images)):
        a = np.array(Image.open(os.path.join(src_images, name)).convert('RGB'))
        black = ndimage.binary_dilation(a.max(2) <= 12, iterations=2)
        Image.fromarray(np.dstack([a, np.where(black, 0, 255).astype(np.uint8)])).save(
            os.path.join(out, 'images', name))


def face_frames(names_path, model_path):
    """{frame number: [(face name, image record)]} for the model's faces."""
    images = {v['name']: v for v in read_images(os.path.join(model_path, 'images.txt')).values()}
    frames = {}
    for line in open(names_path, encoding='utf-8'):
        face, rest = line.split(maxsplit=1)
        parts = rest.split()
        if face in images:
            frames.setdefault(int(os.path.basename(parts[0])[:6]), []).append((' '.join(parts[1:]), images[face]))
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--insv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=300, help='video frames to use, from the start')
    p.add_argument('--imu-frame-period', default='',
                   help="seconds per frame of the .insv, when its trailer times another rate (OverMaps360: 0.333683)")
    p.add_argument('--steps', type=int, default=7000)
    p.add_argument('--max-gaussians', type=int, default=600000)
    p.add_argument('--camera-height', type=float, default=1.2, help="the robot's camera over the ground, m")
    p.add_argument('--goals', type=int, default=4)
    p.add_argument('--fence-m', type=float, default=3.0)
    p.add_argument('--gpu', action='store_true',
                   help="run SfM's matching on the GPU (wgpu) too; 3DGS training uses a GPU whenever wgpu finds one")
    p.add_argument('--nav', action='store_true', help='also drive the goals with Nav2')
    p.add_argument('--lane-half-width', dest='lane_half_width', type=float, default=1.0,
                   help="Nav2's map is free only this far either side of the walk (lane_map.py), so the robot "
                        'keeps to the pavement; 0: the whole occupancy map')
    p.add_argument('--pedestrians', type=int, default=0, help='with --nav: people walking the street')
    p.add_argument('--cyclists', type=int, default=0, help='with --nav: people riding bicycles')
    p.add_argument('--dog-walkers', dest='dog_walkers', type=int, default=0, help='with --nav: people with a dog')
    p.add_argument('--strollers', type=int, default=0, help='with --nav: people pushing a pushchair')
    p.add_argument('--video', action='store_true', help='with --nav: a chase-camera video of the drive (chase.mp4)')
    args = p.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    times = {}
    timing = os.path.join(out, 'timing.json')
    if os.path.exists(timing):
        times = json.load(open(timing))

    def step(name, fn):
        if name in times:
            print(f'[{name}] done before ({times[name]:.0f} s)')
            return
        t0 = time.time()
        fn()
        times[name] = time.time() - t0
        json.dump(times, open(timing, 'w'), indent=1)
        print(f'[{name}] {times[name]:.0f} s', flush=True)

    frames_dir = os.path.join(out, 'frames')

    def cut():
        os.makedirs(frames_dir, exist_ok=True)
        run(['ffmpeg', '-v', 'error', '-i', args.insv, '-filter_complex',
             f"[0:v:0][0:v:1]hstack=inputs=2,select='lt(n\\,{args.frames})'", '-fps_mode', 'passthrough',
             '-frame_pts', '1', '-frames:v', str(args.frames), '-q:v', '2', os.path.join(frames_dir, '%06d.jpg')])
    step('frames', cut)

    sfm = os.path.join(out, 'sfm')

    def do_sfm():
        cmd = [os.path.join(EX, 'gsplat_photos'), '--insv', args.insv, '--images', frames_dir, '--out', sfm,
               '--steps', '0', '--max-size', '640', '--window', '2'] + ([] if args.gpu else ['--cpu'])
        if args.imu_frame_period:
            cmd += ['--imu-frame-period', args.imu_frame_period]
        run(cmd, log=os.path.join(out, 'sfm.log'))
    step('sfm', do_sfm)

    level = os.path.join(out, 'level')
    step('level', lambda: run([sys.executable, os.path.join(HERE, 'level_sfm.py'), '--model',
                               os.path.join(sfm, 'sparse', '0'), '--images', os.path.join(sfm, 'images'),
                               '--out', level]))

    train = os.path.join(out, 'train')
    step('mask', lambda: mask_faces(os.path.join(sfm, 'images'), os.path.join(level, 'sparse', '0'), train))
    splat = os.path.join(out, 'scene.ply')
    step('3dgs', lambda: run([os.path.join(EX, 'gsplat_train'), '--data', train, '--alpha-mask', '--steps',
                              args.steps, '--max-gaussians', args.max_gaussians, '--out', splat],
                             log=os.path.join(out, 'train.log')))

    frames = face_frames(os.path.join(sfm, 'image_names.txt'), os.path.join(level, 'sparse', '0'))
    order = sorted(frames)

    def centre(fr):
        v = frames[fr][0][1]
        return -qvec2rot(v['q']).T @ v['t']

    path = np.array([centre(fr) for fr in order])
    sim = os.path.join(out, 'sim')
    scene_json = os.path.join(sim, 'scene.json')

    def scene():
        d = path[min(5, len(path) - 1), :2] - path[0, :2]
        start = f'{path[0, 0]:.3f},{path[0, 1]:.3f},{math.atan2(d[1], d[0]):.3f}'
        run([sys.executable, os.path.join(HERE, 'make_gs_scene.py'), '--splat', splat, '--model',
             os.path.join(level, 'sparse', '0'), '--out', scene_json, '--camera', FACE,
             '--camera-height', args.camera_height, '--camera-forward', 0, f'--start={start}', '--terrain',
             '--fence-m', args.fence_m, '--min-opacity', 0.3, '--max-scale', 0.3])
        # goals spread along the walk, the last near its end, facing along it
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
        goals = []
        for k in range(1, args.goals + 1):
            i = int(np.searchsorted(s, s[-1] * k / args.goals - (2.0 if k == args.goals else 0)))
            i = min(i, len(path) - 1)
            a, b = path[max(i - 3, 0), :2], path[min(i + 3, len(path) - 1), :2]
            goals.append([round(float(path[i, 0]), 3), round(float(path[i, 1]), 3),
                          round(math.atan2(b[1] - a[1], b[0] - a[0]), 3)])
        sc = json.load(open(scene_json))
        sc['goals'] = goals
        json.dump(sc, open(scene_json, 'w'), indent=1)
        print(f'walk {s[-1]:.0f} m, goals {goals}')
    step('scene', scene)

    mapd = os.path.join(out, 'map')

    def build_map():
        sc = json.load(open(scene_json))
        ground = np.load(os.path.join(sim, sc['ground']))
        gr = sc['grid']

        def gz(x, y):
            i = min(max(int((x - gr['x0']) / gr['resolution']), 0), ground.shape[1] - 1)
            j = min(max(int((y - gr['y0']) / gr['resolution']), 0), ground.shape[0] - 1)
            return float(ground[j, i])
        traj = os.path.join(out, 'faces.tum')
        rows = []
        for fr in order:
            for face, v in frames[fr]:
                if face not in HORIZONTAL:
                    continue
                R = qvec2rot(v['q'])
                c = -R.T @ v['t']
                c[2] = gz(c[0], c[1]) + args.camera_height
                q = rot2qvec(R.T)
                ts = fr * 0.333677 + HORIZONTAL.index(face) * 0.01
                rows.append((ts, f'{ts:.6f} {c[0]:.5f} {c[1]:.5f} {c[2]:.5f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {q[0]:.8f}\n'))
        with open(traj, 'w', encoding='utf-8') as f:
            f.writelines(r for _, r in sorted(rows))
        seq = os.path.join(out, 'render_faces')
        run([sys.executable, os.path.join(HERE, 'render_tum_seq.py'), '--splat', splat, '--trajectory', traj,
             '--out', seq, '--camera', FACE, '--max-depth', 25])
        run([os.path.join(EX, 'tum_rgbd_save_map'), '--seq-dir', seq, '--trajectory', traj, '--out', mapd,
             '--fx', 320, '--fy', 320, '--cx', 320, '--cy', 320, '--k1', 0, '--k2', 0, '--p1', 0, '--p2', 0, '--k3', 0,
             '--match-radius-px', 12, '--max-depth', 20, '--keyframe-m', 0.3, '--keyframe-deg', 10,
             '--max-hamming', 70, '--keep-world'], log=os.path.join(out, 'map.log'))
        run([os.path.join(EX, 'visual_map_to_occupancy'), '--map-dir', mapd, '--camera-height', args.camera_height,
             '--max-range', 20, '--min-height', 0.3, '--out', os.path.join(mapd, 'occupancy')])
        if args.lane_half_width > 0:            # keep the robot to the pavement the video was walked on
            run([sys.executable, os.path.join(HERE, 'lane_map.py'), '--occupancy', os.path.join(mapd, 'occupancy.yaml'),
                 '--walk', traj, '--out', os.path.join(mapd, 'occupancy_lane'), '--half-width', args.lane_half_width])
    step('map', build_map)

    if args.nav:
        def nav():
            ident = os.path.join(sim, 'identity.json')
            json.dump({'x': 0.0, 'y': 0.0, 'yaw': 0.0}, open(ident, 'w'))
            ros2 = os.path.join(REPO, 'ros2', 'visloc-ros2')
            loc = ' '.join(TUNED + ['--dump_inliers', os.path.join(sim, 'inliers.csv')])
            lane = os.path.join(mapd, 'occupancy_lane.yaml')
            occ = f'occupancy:={lane} ' if args.lane_half_width > 0 and os.path.exists(lane) else ''
            sim_args = ' '.join(['--mono', '--pedestrians', str(args.pedestrians), '--cyclists', str(args.cyclists),
                                 '--dog-walkers', str(args.dog_walkers), '--strollers', str(args.strollers)])
            # only this scene's processes (other splats may be rendering on the machine)
            kill = (f'pkill -9 -f "gs_sim.py --scene {scene_json}"; pkill -9 -f "gsplat_render_server --ply {sim}"; '
                    'pkill -9 -f visloc_map_localize; pkill -9 -f "ros-args"')
            script = f'''
source /opt/ros/jazzy/setup.bash
export RCUTILS_COLORIZED_OUTPUT=0
for attempt in 1 2 3; do
  ros2 launch {HERE}/gs_nav.launch.py scene:={scene_json} map:={mapd} {occ}record:={sim}/nav.sim.csv \\
    "sim_args:={sim_args}" "localize_args:={loc}" > {sim}/nav.launch.log 2>&1 &
  LP=$!; ok=0
  for t in $(seq 1 60); do sleep 5; grep -q "lifecycle_manager_navigation\\]: .*Managed nodes are active" {sim}/nav.launch.log && {{ ok=1; break; }}; done
  [ $ok = 1 ] && break
  echo "Nav2 bringup stalled (attempt $attempt), relaunching"; kill -INT $LP; sleep 8
  {kill}; sleep 3
done
${{ROS_PYTHON:-python3}} -u {ros2}/gz/nav_goals.py --world-from-map {ident} --scene {scene_json} --timeout 900 \\
  --out {sim}/nav.json --label real2sim --record {sim}/nav.csv --retries 3 --spin-after 5 > {sim}/nav.goals.log 2>&1
kill -INT $LP; sleep 8
{kill}
'''
            run(['bash', '-c', script])
            res = json.load(open(os.path.join(sim, 'nav.json')))
            print(f"Nav2: {res['reached']}/{res['goal_count']} goals, {res['contacts']} contacts, "
                  f"/map_pose RMSE {res['pose_error_m']['rmse'] * 100:.1f} cm")
        step('nav', nav)

        if args.video:
            def video():
                frames = os.path.join(out, 'chase')
                run([sys.executable, os.path.join(HERE, 'chase_video.py'), '--splat', splat, '--scene', scene_json,
                     '--run', os.path.join(sim, 'nav'), '--map', mapd, '--inliers', os.path.join(sim, 'inliers.csv'),
                     '--out', frames], log=os.path.join(out, 'video.log'))
                run(['ffmpeg', '-v', 'error', '-y', '-framerate', 30, '-i', os.path.join(frames, '%05d.jpg'),
                     '-vf', 'scale=960:-2', '-pix_fmt', 'yuv420p', '-crf', 26, os.path.join(out, 'chase.mp4')])
            step('video', video)

    print(f'done: splat {splat}, scene {scene_json}, map {mapd}')
    print('timing (s):', {k: round(v) for k, v in times.items()})
    if not args.nav:
        print(f'drive it: ros2 launch {HERE}/gs_nav.launch.py scene:={scene_json} map:={mapd} sim_args:=--mono '
              f'"localize_args:={" ".join(TUNED)}"')


if __name__ == '__main__':
    main()
