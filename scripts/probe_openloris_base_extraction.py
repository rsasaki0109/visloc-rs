#!/usr/bin/env python3
"""Probe the recorded base SIFT recipe on four spread-out raw images."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from benchmark_electro import parse_gnu_time
from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
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
    selected = [images[index * (len(images) - 1) // 3] for index in range(4)]
    reference = source / 'tiers/tier-10000/features256'
    expected = {image.stem + '_features.txt': sha(reference / (image.stem + '_features.txt'))
                for image in selected}
    output = args.output.resolve()
    output.mkdir()  # Never overwrite an existing probe.
    inputs = output / 'images'
    inputs.mkdir()
    for image in selected:
        (inputs / image.name).symlink_to(image.resolve(strict=True))
    command[0] = str(binary)
    for flag, path in [('--images-dir', inputs),
                       ('--export-features-dir', output / 'features'),
                       ('--out-colmap', output / 'unused-model')]:
        command[command.index(flag) + 1] = str(path)
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '300s', *command]
    report = {'status': 'running', 'command': invocation, 'binary_sha256': binary_sha,
              'recipe_sha256': sha(recipe), 'reference_feature_sha256': expected,
              'image_sha256': {image.name: sha(image) for image in selected},
              'rayon_num_threads': 8, 'malloc_arena_max': '1',
              'scope': 'Four-image base extraction parity only; not full10k or E2E.'}
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, RAYON_NUM_THREADS='8', MALLOC_ARENA_MAX='1'))
    actual = {path.name: sha(path) for path in (output / 'features').glob('*_features.txt')}
    report.update(exit_code=result.returncode, output_feature_sha256=actual,
                  measurement=parse_gnu_time(output / 'time.txt'),
                  status='pass' if result.returncode == 0 and actual == expected else 'fail')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
