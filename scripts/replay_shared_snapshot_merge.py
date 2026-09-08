#!/usr/bin/env python3
"""Merge shared-envelope shards and compare the full legacy reference bytes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from benchmark_electro import parse_gnu_time
from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    paths = sorted(args.shards.resolve(strict=True).glob('*.vps'))
    if not paths:
        parser.error('No shards')
    reference_sha = sha(args.reference)
    output = args.output.resolve()
    output.mkdir()
    if any(any(char.isspace() for char in str(path)) for path in paths):
        parser.error('Snapshot-list format does not support whitespace in paths')
    listing = output / 'snapshots.txt'
    listing.write_text(''.join(str(path) + '\n' for path in paths))
    command = [str(binary), '--output', str(output / 'merged.vps'), '--snapshot-list', str(listing)]
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '180s', *command]
    report = {'status': 'running', 'binary_sha256': sha(binary),
              'shards': len(paths), 'reference_sha256': reference_sha,
              'command': invocation, 'rayon_num_threads': 1, 'malloc_arena_max': '1'}
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, RAYON_NUM_THREADS='1', MALLOC_ARENA_MAX='1'))
    actual = sha(output / 'merged.vps') if (output / 'merged.vps').exists() else None
    report.update(exit_code=result.returncode, output_sha256=actual,
                  measurement=parse_gnu_time(output / 'time.txt'),
                  status='pass' if result.returncode == 0 and actual == reference_sha else 'fail')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'command'}, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
