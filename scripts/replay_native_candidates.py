#!/usr/bin/env python3
"""Replay the retained OpenLORIS native candidate recipe without overwriting it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    base = Path('/home/sasaki/datasets/openloris')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=base / 'corridor1-1-m8-extract-resume-pilot-v1/extract-3ae253a')
    parser.add_argument('--binary-sha256', default='8cfa9c53fcaea5d6305dbdd3018b8381ae214806751e6ee3dc85bb8692a8da34')
    parser.add_argument('--output-name', default='corridor1-1-m8-native-candidates-replay-v1')
    parser.add_argument('--source-commit', default='3ae253a0af909042fe0cfbdb08e8d50733b79375')
    args = parser.parse_args()
    if not args.output_name.startswith('corridor1-1-m8-native-candidates-') or Path(args.output_name).name != args.output_name:
        parser.error('--output-name must be a simple native-candidate replay directory name')
    reference = base / 'corridor1-1-m8-native-rig-runner-10k-v1'
    output = base / args.output_name
    binary = args.binary.resolve(strict=True)
    expected_binary = args.binary_sha256
    if sha(binary) != expected_binary:
        raise RuntimeError('Frozen binary changed')
    timing = reference / 'timing/candidate-generation.time.txt'
    first_line = timing.read_text().splitlines()[0]
    recorded = shlex.split(first_line.split('Command being timed: ', 1)[1])[0]
    command = shlex.split(recorded)
    command[0] = str(binary)
    for flag, name in [('--export-candidate-manifest', 'candidates.txt'),
                       ('--out-colmap', 'unused-model-candidates')]:
        command[command.index(flag) + 1] = str(output / name)
    output.mkdir()  # Fail closed on an existing replay directory.
    env = dict(os.environ, RAYON_NUM_THREADS='8')
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '1800s', *command]
    report = {'command': invocation, 'binary_sha256': expected_binary,
              'source_commit': args.source_commit,
              'reference_recipe_sha256': sha(timing),
              'reference_sha256': sha(reference / 'candidates.txt'),
              'rayon_num_threads': 8, 'status': 'running',
              'scope': 'candidate generation only; retained base features'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, env=env, stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode, wall_seconds=time.monotonic() - started)
    candidate = output / 'candidates.txt'
    report['output_sha256'] = sha(candidate) if candidate.exists() else None
    report['byte_identical'] = report['reference_sha256'] == report['output_sha256']
    report['status'] = 'pass' if result.returncode == 0 and report['byte_identical'] else 'fail'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
