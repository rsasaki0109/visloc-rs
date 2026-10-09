#!/usr/bin/env python3
"""README hero on a Tanks and Temples scene, end to end on a Colab GPU VM.

Vulkan for the NVIDIA GPU -> build visloc-rs -> download the photos ->
`gsplat_photos` (SfM, 3DGS, mesh) -> `scripts/make_readme_hero.py`. Every
step is skipped when its output already exists, so a rerun only redoes what
is missing (e.g. just the GIF after changing --hero-flags).

Driven from a terminal with the Colab CLI (github.com/googlecolab/google-colab-cli):

    colab new --gpu T4 -s hero
    colab upload -s hero scripts/colab/hero_pipeline.py /content/hero_pipeline.py
    colab exec -s hero --timeout 60 <<< "import subprocess; subprocess.Popen(
        'nohup python3 /content/hero_pipeline.py > /content/hero.log 2>&1 &', shell=True)"
    colab exec -s hero <<< "print(open('/content/hero.log').read()[-3000:])"
    colab download -s hero /content/out/hero_reconstruction.gif docs/assets/hero_reconstruction.gif

or from scripts/colab/readme_hero_courthouse.ipynb.

The current README hero (Hierarchical 3DGS SmallCity, A100 high-mem):

    hero_pipeline.py --dataset h3dgs --scene small_city --frame-stride 1 --max-size 1024 \
        --exhaustive-max 600 --window 20 --retrieval 30 --steps 50000 \
        --photos-args=--no-refine-intrinsics \
        "--hero-flags=--elev 45 --radius-mult 1.4 --target-height 0.2 --zoom 0.9 \
         --width 560 --colors 72 --filter-dist-mult 4 --filter-isolated 6 --point-size 2.5"
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time


def sh(cmd, check=True, quiet=False):
    print('$', cmd, flush=True)
    p = subprocess.Popen(['bash', '-o', 'pipefail', '-c', cmd], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = []
    for line in p.stdout:
        out.append(line)
        if not quiet:
            print(line, end='', flush=True)
    p.wait()
    if check and p.returncode != 0:
        if quiet:
            print(''.join(out[-60:]), flush=True)
        raise RuntimeError(f'exit {p.returncode}: {cmd}')
    return ''.join(out)


# ---------------------------------------------------------------------------
# Vulkan: Colab ships the NVIDIA compute driver but not always its Vulkan/EGL
# user-space part. Add the part that matches the running kernel driver
# exactly: already-present libraries first, then the apt package of the same
# version, then the libraries from NVIDIA's installer of that version
# (extracted only; no kernel module is touched).
# ---------------------------------------------------------------------------

LIBDIR = '/usr/lib/x86_64-linux-gnu'


def vulkan_gpu():
    r = subprocess.run('vulkaninfo --summary', shell=True, text=True, capture_output=True)
    names = re.findall(r'deviceName\s*=\s*(.+)', r.stdout)
    hw = [n for n in names if 'llvmpipe' not in n and 'SwiftShader' not in n]
    return hw[0].strip() if hw else None


def write_icds(glx='libGLX_nvidia.so.0', egl='libEGL_nvidia.so.0'):
    os.makedirs('/usr/share/vulkan/icd.d', exist_ok=True)
    json.dump({'file_format_version': '1.0.0',
               'ICD': {'library_path': glx, 'api_version': '1.3.242'}},
              open('/usr/share/vulkan/icd.d/nvidia_icd.json', 'w'))
    os.makedirs('/usr/share/glvnd/egl_vendor.d', exist_ok=True)
    json.dump({'file_format_version': '1.0.0', 'ICD': {'library_path': egl}},
              open('/usr/share/glvnd/egl_vendor.d/10_nvidia.json', 'w'))


def setup_vulkan():
    sh('apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y '
       'vulkan-tools libvulkan1 libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri ffmpeg unzip > /dev/null',
       quiet=True)
    drv = sh('nvidia-smi --query-gpu=driver_version --format=csv,noheader',
             quiet=True).strip().splitlines()[0]
    major = drv.split('.')[0]
    print('NVIDIA driver', drv, flush=True)

    def present():
        for d in ['/usr/lib64-nvidia', LIBDIR, '/usr/local/nvidia/lib64']:
            hits = sorted(glob.glob(f'{d}/libGLX_nvidia.so.{drv}'))
            if hits:
                egl = glob.glob(f'{d}/libEGL_nvidia.so.{drv}')
                write_icds(hits[0], egl[0] if egl else 'libEGL_nvidia.so.0')
                return True
        return False

    def apt():
        out = subprocess.run(f'apt-cache madison libnvidia-gl-{major}', shell=True, text=True,
                             capture_output=True).stdout
        vers = [l.split('|')[1].strip() for l in out.splitlines() if '|' in l]
        match = [v for v in vers if v.startswith(drv)]
        print('apt libnvidia-gl versions:', vers[:5], '-> match', match[:1], flush=True)
        if not match:
            return False
        sh(f'DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends '
           f'libnvidia-gl-{major}={match[0]} > /dev/null', check=False, quiet=True)
        write_icds()
        return True

    def runfile():
        run = f'/content/NVIDIA-Linux-x86_64-{drv}.run'
        if not os.path.exists(run):
            for url in [f'https://us.download.nvidia.com/tesla/{drv}/NVIDIA-Linux-x86_64-{drv}.run',
                        f'https://us.download.nvidia.com/XFree86/Linux-x86_64/{drv}/NVIDIA-Linux-x86_64-{drv}.run']:
                if subprocess.run(f'wget -q -O {run} {url}', shell=True).returncode == 0:
                    break
                if os.path.exists(run):
                    os.remove(run)
            else:
                return False
        ex = f'/content/nvidia-{drv}'
        if not os.path.isdir(ex):
            sh(f'sh {run} --extract-only --target {ex} > /dev/null', quiet=True)
        for n in ['libGLX_nvidia', 'libnvidia-glcore', 'libnvidia-glvkspirv', 'libnvidia-tls',
                  'libnvidia-gpucomp', 'libEGL_nvidia', 'libnvidia-eglcore', 'libnvidia-glsi']:
            src = f'{ex}/{n}.so.{drv}'
            if os.path.exists(src):
                shutil.copy(src, LIBDIR)
                if n in ('libGLX_nvidia', 'libEGL_nvidia'):
                    link = f'{LIBDIR}/{n}.so.0'
                    if os.path.lexists(link):
                        os.remove(link)
                    os.symlink(f'{n}.so.{drv}', link)
        sh('ldconfig', quiet=True)
        write_icds()
        return True

    gpu = vulkan_gpu()
    for name, step in [('present', present), ('apt', apt), ('runfile', runfile)]:
        if gpu:
            break
        print(f'-- trying {name}', flush=True)
        if step():
            gpu = vulkan_gpu()
    print('Vulkan GPU:', gpu, flush=True)
    if not gpu:
        sh('vulkaninfo --summary', check=False)
        raise SystemExit('no hardware Vulkan device')
    os.environ['WGPU_BACKEND'] = 'vulkan'


def undistort_one(job):
    """Downscale one photo and undistort it with the dataset calibration."""
    import cv2
    import numpy as np
    src, dst, max_size, calib_w, (fx, fy, cx, cy), dist = job
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    k_scale = max_size / max(img.shape[:2]) * img.shape[1] / calib_w
    img = cv2.resize(img, None, fx=max_size / max(img.shape[:2]), fy=max_size / max(img.shape[:2]),
                     interpolation=cv2.INTER_AREA)
    if any(abs(d) > 0 for d in dist):
        k = np.array([[fx * k_scale, 0, cx * k_scale], [0, fy * k_scale, cy * k_scale], [0, 0, 1]])
        img = cv2.undistort(img, k, np.array(dist))
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='https://github.com/rsasaki0109/visloc-rs.git')
    ap.add_argument('--branch', default='main')
    ap.add_argument('--no-fetch', action='store_true',
                    help='build an existing /content/visloc-rs checkout as it is (e.g. with local patches)')
    ap.add_argument('--dataset', choices=['tnt', 'mill19', 'h3dgs'], default='tnt')
    ap.add_argument('--scene', default='Courthouse',
                    help='tnt: zip name in hf.co/datasets/hongliu6/tanks_and_temples; '
                         'mill19: building or rubble; h3dgs: small_city')
    ap.add_argument('--frame-stride', type=int, default=2)
    ap.add_argument('--max-size', type=int, default=1280)
    ap.add_argument('--exhaustive-max', type=int, default=600)
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--retrieval', type=int, default=0)
    ap.add_argument('--photos-args', default='',
                    help='extra gsplat_photos flags, e.g. --no-refine-intrinsics')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--normal-weight', type=float, default=0.005)
    ap.add_argument('--run', default=None, help='run dir (default /content/runs/<scene>)')
    ap.add_argument('--hero-flags',
                    default=('--elev 40 --radius-mult 2.6 --target-height 0.25 --zoom 1.15 --width 640 '
                             '--filter-dist-mult 4 --filter-isolated 6 --point-size 3.5'))
    args = ap.parse_args()

    t_all = time.time()
    run = args.run or f'/content/runs/{args.scene.lower()}'
    os.makedirs(run, exist_ok=True)
    os.makedirs('/content/out', exist_ok=True)
    sh('nvidia-smi')
    setup_vulkan()

    # ---- build ----
    os.environ['PATH'] = '/root/.cargo/bin:' + os.environ['PATH']
    if not os.path.exists('/root/.cargo/bin/cargo'):
        sh('curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null', quiet=True)
    repo = '/content/visloc-rs'
    if not os.path.isdir(repo):
        sh(f'git clone --depth 1 -b {args.branch} {args.repo} {repo}')
    elif not args.no_fetch:
        sh(f'cd {repo} && git fetch --depth 1 origin {args.branch} && git reset --hard FETCH_HEAD')
    t = time.time()
    sh(f'cd {repo} && cargo build --release -p visloc-gsplat-train --features gpu,euroc '
       '--example gsplat_photos --example gsplat_eval 2>&1 | tail -3')
    print(f'build {time.time() - t:.0f}s', flush=True)
    bin_dir = f'{repo}/target/release/examples'

    # ---- photos ----
    focal_flag = ''
    # Dataset-specific gsplat_photos flags.
    dataset_flags = ''
    if args.dataset == 'tnt':
        images = f'/content/data/{args.scene}_stride{args.frame_stride}'
        label = f'Tanks and Temples {args.scene}'
        if not os.path.isdir(images) or not os.listdir(images):
            z = f'/content/data/{args.scene}.zip'
            os.makedirs('/content/data', exist_ok=True)
            if not os.path.exists(z):
                sh(f'wget -q -O {z} "https://huggingface.co/datasets/hongliu6/tanks_and_temples/'
                   f'resolve/main/{args.scene}.zip"')
            sh(f'cd /content/data && rm -rf {args.scene} && unzip -q {z}')
            frames = sorted(glob.glob(f'/content/data/{args.scene}/*.jpg'))
            os.makedirs(images, exist_ok=True)
            for f in frames[::args.frame_stride]:
                shutil.move(f, images)
            shutil.rmtree(f'/content/data/{args.scene}')
    elif args.dataset == 'h3dgs':
        # Hierarchical 3DGS SmallCity: a city block filmed from a bicycle
        # helmet with 6 GoPros in 4 passes. The release only ships the
        # photos undistorted to its shared pinhole camera (people and cars
        # blacked out); take that camera's focal length as the prior.
        name = args.scene.lower()
        images = f'/content/data/{name}_stride{args.frame_stride}'
        label = 'Hierarchical 3DGS SmallCity' if name == 'small_city' else f'Hierarchical 3DGS {name}'
        root = f'/content/data/{name}'
        if not os.path.isdir(f'{root}/camera_calibration/rectified/images'):
            os.makedirs('/content/data', exist_ok=True)
            z = f'/content/data/{name}.zip'
            if not os.path.exists(z):
                sh(f'wget -q -O {z} https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/'
                   f'datasets/full_scenes/{name}.zip')
            sh(f'cd /content/data && unzip -q -o {z} "{name}/camera_calibration/rectified/images/*" '
               f'"{name}/camera_calibration/aligned/sparse/0/cameras.bin"')
        if not os.path.isdir(images) or not os.listdir(images):
            frames = sorted(glob.glob(f'{root}/camera_calibration/rectified/images/*'))
            os.makedirs(images, exist_ok=True)
            for f in frames[::args.frame_stride]:
                os.link(f, f'{images}/{os.path.basename(f)}')
        import struct
        b = open(f'{root}/camera_calibration/aligned/sparse/0/cameras.bin', 'rb').read()
        _, model, w, h = struct.unpack_from('<iiQQ', b, 8)
        fx = struct.unpack_from('<d', b, 32)[0]
        print(f'release camera: model {model} {w}x{h} fx {fx:.1f}', flush=True)
        focal_flag = f'--focal {fx * min(1.0, args.max_size / max(w, h)):.2f}'
        # People and cars are blacked out in these photos: train around them.
        dataset_flags = '--mask-black 12'
    else:
        # Mill-19 (Mega-NeRF): drone surveys, one camera; train + val photos.
        name = args.scene.lower()
        images = f'/content/data/mill19_{name}_stride{args.frame_stride}'
        label = f'Mill-19 {name}'
        root = f'/content/data/{name}-pixsfm'
        if not os.path.isdir(root):
            os.makedirs('/content/data', exist_ok=True)
            sh(f'cd /content/data && wget -q -O - https://storage.cmusatyalab.org/mega-nerf-data/'
               f'{name}-pixsfm.tgz | tar xz')
        # The photos carry no EXIF and are not undistorted. Take the camera
        # calibration (intrinsics + distortion only, no poses) from the
        # dataset, downscale to --max-size and undistort with it; the final
        # bundle adjustment still refines the focal length as usual.
        import torch
        meta = torch.load(sorted(glob.glob(f'{root}/train/metadata/*'))[0], map_location='cpu')
        fx, fy, cx, cy = [float(v) for v in meta['intrinsics']]
        w, h = int(meta['W']), int(meta['H'])
        dist = [float(v) for v in meta.get('distortion', [])]
        scale = args.max_size / max(w, h)
        print(f'calibration {w}x{h} f=({fx:.1f},{fy:.1f}) c=({cx:.1f},{cy:.1f}) dist={dist}', flush=True)
        if not os.path.isdir(images) or not os.listdir(images):
            frames = sorted(glob.glob(f'{root}/train/rgbs/*') + glob.glob(f'{root}/val/rgbs/*'),
                            key=os.path.basename)[::args.frame_stride]
            os.makedirs(images, exist_ok=True)
            jobs = [(f, f'{images}/{"" if f.split("/")[-3] == "train" else "val_"}'
                        f'{os.path.splitext(os.path.basename(f))[0]}.jpg',
                     args.max_size, w, (fx, fy, cx, cy), dist) for f in frames]
            from multiprocessing import Pool
            with Pool(os.cpu_count()) as pool:
                pool.map(undistort_one, jobs, chunksize=8)
        focal_flag = f'--focal {fx * scale:.2f}'
        print(f'focal prior {fx * scale:.1f} px at --max-size {args.max_size}', flush=True)
    n_input = len(os.listdir(images))
    print(n_input, 'photos in', images, flush=True)

    # ---- photos -> SfM -> 3DGS -> mesh ----
    if not os.path.exists(f'{run}/mesh.ply'):
        t = time.time()
        sh(f'{bin_dir}/gsplat_photos --images {images} --out {run} --max-size {args.max_size} '
           f'--steps {args.steps} --normal-weight {args.normal_weight} '
           f'--exhaustive-max {args.exhaustive_max} --window {args.window} --retrieval {args.retrieval} {focal_flag} '
           f'{dataset_flags} '
           f'{args.photos_args} 2>&1 '
           f'| grep --line-buffered -v -E "^(BA_|INIT_PAIR|BA global|BA local|TIMING)" '
           f'| tee {run}/gsplat_photos.log')
        print(f'gsplat_photos {time.time() - t:.0f}s', flush=True)

    # ---- hero GIF ----
    sh('pip -q install moderngl plyfile scipy pillow', quiet=True)
    sh(f'cd {repo} && python3 scripts/make_readme_hero.py --src {run} --work /content/hero '
       f'--gsplat-eval {bin_dir}/gsplat_eval --scene-label "{label}" '
       f'--input-images {n_input} {args.hero_flags} --out /content/out/hero_reconstruction.gif '
       f'2>&1 | grep -v -E "^(  |frame=|\\[|ffmpeg version|Input|Output|Stream|Press)"')
    shutil.copy(f'{run}/gsplat_photos.log', '/content/out/gsplat_photos.log')
    print(f'ALL DONE in {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
