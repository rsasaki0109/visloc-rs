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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='https://github.com/rsasaki0109/visloc-rs.git')
    ap.add_argument('--branch', default='main')
    ap.add_argument('--scene', default='Courthouse',
                    help='zip name in hf.co/datasets/hongliu6/tanks_and_temples')
    ap.add_argument('--frame-stride', type=int, default=2)
    ap.add_argument('--max-size', type=int, default=1280)
    ap.add_argument('--exhaustive-max', type=int, default=600)
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
    else:
        sh(f'cd {repo} && git fetch --depth 1 origin {args.branch} && git reset --hard FETCH_HEAD')
    t = time.time()
    sh(f'cd {repo} && cargo build --release -p visloc-gsplat-train --features gpu,euroc '
       '--example gsplat_photos --example gsplat_eval 2>&1 | tail -3')
    print(f'build {time.time() - t:.0f}s', flush=True)
    bin_dir = f'{repo}/target/release/examples'

    # ---- photos ----
    images = f'/content/data/{args.scene}_stride{args.frame_stride}'
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
    n_input = len(os.listdir(images))
    print(n_input, 'photos in', images, flush=True)

    # ---- photos -> SfM -> 3DGS -> mesh ----
    if not os.path.exists(f'{run}/mesh.ply'):
        t = time.time()
        sh(f'{bin_dir}/gsplat_photos --images {images} --out {run} --max-size {args.max_size} '
           f'--steps {args.steps} --normal-weight {args.normal_weight} '
           f'--exhaustive-max {args.exhaustive_max} 2>&1 '
           f'| grep --line-buffered -v -E "^(BA_|INIT_PAIR|BA global|BA local|TIMING|REGISTER)" '
           f'| tee {run}/gsplat_photos.log')
        print(f'gsplat_photos {time.time() - t:.0f}s', flush=True)

    # ---- hero GIF ----
    sh('pip -q install moderngl plyfile scipy pillow', quiet=True)
    sh(f'cd {repo} && python3 scripts/make_readme_hero.py --src {run} --work /content/hero '
       f'--gsplat-eval {bin_dir}/gsplat_eval --scene-label "Tanks and Temples {args.scene}" '
       f'--input-images {n_input} {args.hero_flags} --out /content/out/hero_reconstruction.gif '
       f'2>&1 | grep -v -E "^(  |frame=|\\[|ffmpeg version|Input|Output|Stream|Press)"')
    shutil.copy(f'{run}/gsplat_photos.log', '/content/out/gsplat_photos.log')
    print(f'ALL DONE in {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
