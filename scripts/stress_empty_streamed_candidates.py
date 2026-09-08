#!/usr/bin/env python3
"""Linux CLI stress: empty feature banks must reject before exhaustive pairing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_electro import parse_gnu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--images', type=int, default=100000)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('This resource-limited fixture requires Linux')
    if not 2 <= args.images <= 100000:
        parser.error('--images must be between 2 and 100000')
    import resource

    def limit_address_space():
        ceiling = 2 * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    binary = args.binary.resolve(strict=True)
    with binary.open('rb') as stream:
        binary_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    root = args.root.resolve()
    root.mkdir()  # No reuse or replacement of an existing fixture.
    features = root / 'features'
    calibration = root / 'calibration'
    features.mkdir()
    calibration.mkdir()
    (calibration / 'cameras.txt').write_text('1 PINHOLE 100 100 50 50 50 50\n')
    with (calibration / 'images.txt').open('w') as images:
        for index in range(args.images):
            stem = f'image_{index:06d}'
            (features / f'{stem}_features.txt').touch(exist_ok=False)
            images.write(f'{index + 1} 1 0 0 0 0 0 0 1 {stem}.png\n\n')
    destination = root / 'candidates.txt'
    command = [str(binary), '--feature-extractor', 'files', '--features-dir', str(features),
               '--feature-suffix', '_features.txt', '--image-suffix', '.png',
               '--input-colmap-calibration', str(calibration),
               '--pair-source', 'temporal-pyramid', '--retrieval-topk', '32',
               '--candidate-budget', str(args.images * 7), '--stream-candidate-features',
               '--retrieval-backend', 'lsh', '--ann-tables', '8', '--ann-bits', '9',
               '--ann-probes', '6', '--export-candidate-manifest', str(destination),
               '--out-colmap', str(root / 'unused-model')]
    invocation = ['/usr/bin/time', '-v', '-o', str(root / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=5s', '120s', *command]
    started = time.monotonic()
    with (root / 'run.log').open('w') as log:
        result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, RAYON_NUM_THREADS='8', MALLOC_ARENA_MAX='1'),
                                preexec_fn=limit_address_space)
    elapsed = time.monotonic() - started
    diagnostic = (root / 'run.log').read_text()
    report = {'status': 'pass' if result.returncode == 1
              and 'refusing exhaustive fallback' in diagnostic and not destination.exists() else 'fail',
              'images': args.images, 'candidate_budget': args.images * 7,
              'command': invocation, 'binary_sha256': binary_sha,
              'expected_exit_status': 1, 'actual_exit_status': result.returncode,
              'address_space_limit_bytes': 2 * 1024**3, 'wall_seconds': elapsed,
              'candidate_manifest_exists': destination.exists(),
              'measurement': parse_gnu_time(root / 'time.txt'),
              'scope': 'empty-feature candidate rejection only; not full100k SfM or quality'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
