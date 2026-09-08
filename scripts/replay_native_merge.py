#!/usr/bin/env python3
"""Merge only fully reproduced native match shards and check reference bytes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_electro import build_merge_command, parse_gnu_time
from replay_native_candidates import sha
from replay_native_matching import snapshot_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['native', 'adaptive', 'targeted7', 'dense'], default='native')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    args = parser.parse_args()
    base = Path('/home/sasaki/datasets/openloris')
    reference = (base / 'corridor1-1-m8-native-rig-runner-10k-v1' if args.variant == 'native'
                 else base / 'corridor1-1-m8-adaptive32-halo8-10k-v1/pipeline')
    if args.variant == 'targeted7':
        reference = base / 'corridor1-1-m8-targeted7-dense256-v1'
    elif args.variant == 'dense':
        reference = base / 'corridor1-1-m8-dense256x2-10k-ann-gap128-local32-8n-v1'
    matching = base / f'corridor1-1-m8-{args.variant}-matching-replay-v1'
    report_path = matching / 'report.json'
    report = json.loads(report_path.read_text())
    accepted_status = 'complete-awaiting-merge-comparison' if args.variant == 'adaptive' else 'pass'
    if report['status'] != accepted_status or report['exit_code'] != 0 or sha(args.binary) != args.binary_sha256:
        raise RuntimeError('Matching replay or merge binary prerequisite failed')
    expected_count = 448 if args.variant == 'targeted7' else 2188
    if args.variant == 'dense':
        expected_count = 2500
    expected_names = {f'verified-{index:06d}.vps' for index in range(expected_count)}
    actual_names = {path.name for path in (matching / 'matches').glob('*.vps')}
    if actual_names != expected_names:
        raise RuntimeError('Snapshot membership differs')
    if args.variant == 'adaptive' and snapshot_inventory(matching / 'matches', expected_names) != report.get('snapshot_inventory_sha256'):
        raise RuntimeError('Adaptive snapshot bytes changed after matching')
    snapshots = []
    for name in sorted(expected_names):
        path = matching / 'matches' / name
        if args.variant != 'adaptive' and sha(path) != sha(reference / 'matches' / name):
            raise RuntimeError(f'Snapshot changed: {name}')
        snapshots.append(path)
    output = base / f'corridor1-1-m8-{args.variant}-merge-replay-v1'
    output.mkdir()
    destination = output / 'verified-merged.vps'
    command = build_merge_command(args.binary.resolve(), destination, snapshots)
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '600s', *command]
    started = time.monotonic()
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, env=dict(os.environ, RAYON_NUM_THREADS='1', MALLOC_ARENA_MAX='1'),
                                stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    expected = sha(reference / 'mapping/verified-merged.vps')
    actual = sha(destination) if destination.exists() else None
    report = {'status': 'pass' if result.returncode == 0 and actual == expected else 'fail',
              'variant': args.variant,
              'malloc_arena_max': '1',
              'exit_code': result.returncode, 'wall_seconds': elapsed,
              'binary': str(args.binary.resolve()), 'binary_sha256': args.binary_sha256,
              'matching_report_sha256': sha(report_path), 'snapshot_count': len(snapshots),
              'output_sha256': actual, 'reference_sha256': expected,
              'measurement': parse_gnu_time(output / 'time.txt'),
              'scope': 'merge of freshly generated matching shards only; not E2E'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
