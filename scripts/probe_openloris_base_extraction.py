#!/usr/bin/env python3
"""Probe the recorded base SIFT recipe on four spread-out raw images."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor

from benchmark_electro import parse_gnu_time
from replay_native_candidates import sha


def partition_images(images, workers):
    if not 1 <= workers <= 6 or workers > len(images):
        raise ValueError('Require 1..6 workers and at least one image per worker')
    if len({p.stem for p in images}) != len(images):
        raise ValueError('Image stems must be unique')
    return [images[index::workers] for index in range(workers)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-images', action='store_true',
                        help='Run the full frozen 10k base bank instead of four spread-out images')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    base = Path('/home/sasaki/datasets/openloris')
    source = base / 'corridor1-1-m5'
    binary = base / 'corridor1-1-m8-extract-resume-pilot-v1/extract-3ae253a'
    binary_sha = sha(binary)
    if binary_sha != '8cfa9c53fcaea5d6305dbdd3018b8381ae214806751e6ee3dc85bb8692a8da34':
        raise RuntimeError('Frozen extractor changed')
    recipe = source / 'feature-extract/timing/shard-0.time.txt'
    command = shlex.split(shlex.split(recipe.read_text().splitlines()[0].split(
        'Command being timed: ', 1)[1])[0])
    images = sorted((source / 'images').glob('*.png'))
    if len(images) < 4:
        raise RuntimeError('Raw image bank missing')
    if args.all_images and len(images) != 10000:
        raise RuntimeError('Full base replay requires exactly 10000 raw images')
    selected = images if args.all_images else [images[index * (len(images) - 1) // 3] for index in range(4)]
    partitions = partition_images(selected, args.workers)
    reference = source / 'tiers/tier-10000/features256'
    expected = {image.stem + '_features.txt': sha(reference / (image.stem + '_features.txt'))
                for image in selected}
    output = args.output.resolve()
    required = sum((reference / name).stat().st_size for name in expected) + 1024**3
    if shutil.disk_usage(output.parent).free < required:
        raise RuntimeError('Insufficient disk for expected base bank plus 1 GiB reserve')
    output.mkdir()  # Never overwrite an existing probe.
    inputs = output / 'images'
    command[0] = str(binary)
    for flag, path in [('--images-dir', inputs),
                       ('--export-features-dir', output / 'features'),
                       ('--out-colmap', output / 'unused-model')]:
        command[command.index(flag) + 1] = str(path)
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                  '21600s' if args.all_images else '300s', *command]
    invocations = []
    for index, partition in enumerate(partitions):
        worker_inputs = output / f'images-{index}'
        worker_inputs.mkdir()
        for image in partition:
            (worker_inputs / image.name).symlink_to(image.resolve(strict=True))
        worker = list(invocation)
        worker[worker.index('--images-dir') + 1] = str(worker_inputs)
        worker[worker.index('-o') + 1] = str(output / f'time-{index}.txt')
        worker[worker.index('--out-colmap') + 1] = str(output / f'unused-model-{index}')
        invocations.append(worker)
    threads = '8' if args.workers == 1 else '1'
    report = {'status': 'running', 'binary_sha256': binary_sha,
              'recipe_sha256': sha(recipe), 'reference_feature_sha256': expected,
              'image_sha256': {image.name: sha(image) for image in selected},
              'rayon_num_threads': int(threads), 'malloc_arena_max': '1',
              'worker_commands': invocations, 'workers': args.workers,
              'image_count': len(selected),
              'scope': 'Base extraction parity only; full10k when --all-images is explicit. Not dense extraction, continuous E2E or a speedup claim.'}
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    def run_worker(item):
        index, worker = item
        with (output / f'run-{index}.log').open('w') as log:
            return subprocess.run(worker, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, RAYON_NUM_THREADS=threads, MALLOC_ARENA_MAX='1')).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run_worker, enumerate(invocations)))
    actual = {path.name: sha(path) for path in (output / 'features').glob('*_features.txt')}
    report.update(exit_code=0 if all(code == 0 for code in codes) else 1,
                  worker_exit_codes=codes, output_feature_sha256=actual,
                  worker_measurements=[parse_gnu_time(output / f'time-{index}.txt') for index in range(args.workers)],
                  status='pass' if all(code == 0 for code in codes) and actual == expected else 'fail')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('image_sha256', 'reference_feature_sha256', 'output_feature_sha256', 'worker_commands')}, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
