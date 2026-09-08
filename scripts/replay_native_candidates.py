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


def replace_path(command, flag, path):
    if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
        raise ValueError(f'Missing or duplicate path flag: {flag}')
    result = list(command)
    result[result.index(flag) + 1] = str(path)
    return result


def bind_feature_input(command, manifest_path, override=None):
    from benchmark_electro import validate_feature_manifest
    if command.count('--features-dir') != 1:
        raise ValueError('Missing or duplicate feature directory')
    position = command.index('--features-dir') + 1
    if position >= len(command):
        raise ValueError('Missing feature directory value')
    features = Path(override if override is not None else command[position]).resolve(strict=True)
    validate_feature_manifest(manifest_path, features)
    manifest = json.loads(manifest_path.read_text())
    expected = {entry['feature'] for entry in manifest['images']}
    actual = {path.name for path in features.iterdir()
              if path.name.endswith(manifest['feature_suffix'])}
    if actual != expected:
        raise ValueError('Feature bank membership differs from frozen manifest')
    return replace_path(command, '--features-dir', features), features


def main():
    base = Path('/home/sasaki/datasets/openloris')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['native', 'dense'], default='native')
    parser.add_argument('--binary', type=Path, default=base / 'corridor1-1-m8-extract-resume-pilot-v1/extract-3ae253a')
    parser.add_argument('--binary-sha256', default='8cfa9c53fcaea5d6305dbdd3018b8381ae214806751e6ee3dc85bb8692a8da34')
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--output-name')
    outputs.add_argument('--output', type=Path, help='Fresh output directory for an enclosing run')
    parser.add_argument('--features-dir', type=Path, help='Explicit bank; must match the frozen feature manifest')
    parser.add_argument('--source-commit', default='3ae253a0af909042fe0cfbdb08e8d50733b79375')
    args = parser.parse_args()
    if args.output_name is None:
        args.output_name = f'corridor1-1-m8-{args.variant}-candidates-replay-v1'
    if not args.output_name.startswith(f'corridor1-1-m8-{args.variant}-candidates-') or Path(args.output_name).name != args.output_name:
        parser.error('--output-name must be a simple variant-specific candidate replay directory name')
    reference = (base / 'corridor1-1-m8-native-rig-runner-10k-v1' if args.variant == 'native'
                 else base / 'corridor1-1-m8-dense256x2-10k-ann-gap128-local32-8n-v1')
    output = args.output.resolve() if args.output is not None else base / args.output_name
    binary = args.binary.resolve(strict=True)
    expected_binary = args.binary_sha256
    if sha(binary) != expected_binary:
        raise RuntimeError('Frozen binary changed')
    timing = reference / 'timing/candidate-generation.time.txt'
    first_line = timing.read_text().splitlines()[0]
    recorded = shlex.split(first_line.split('Command being timed: ', 1)[1])[0]
    command = shlex.split(recorded)
    command[0] = str(binary)
    # Validate either explicit or historical input before publishing any output.
    command, features = bind_feature_input(command, reference / 'features.json', args.features_dir)
    for flag, name in [('--export-candidate-manifest', 'candidates.txt'),
                       ('--out-colmap', 'unused-model-candidates')]:
        command = replace_path(command, flag, output / name)
    output.mkdir()  # Fail closed on an existing replay directory.
    env = dict(os.environ, RAYON_NUM_THREADS='8')
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '1800s', *command]
    report = {'command': invocation, 'binary_sha256': expected_binary,
              'variant': args.variant,
              'source_commit': args.source_commit,
              'reference_recipe_sha256': sha(timing),
              'reference_sha256': sha(reference / 'candidates.txt'),
              'features_resolved': str(features.resolve()),
              'feature_manifest_sha256': sha(reference / 'features.json'),
              'rayon_num_threads': 8, 'status': 'running',
              'scope': 'Candidate generation only; manifest-validated bank. An explicit path does not by itself establish fresh extraction or E2E provenance.'}
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
