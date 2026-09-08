#!/usr/bin/env python3
"""Replay spread-out dense matching shards through the prepared shared writer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from benchmark_electro import parse_gnu_time, validate_feature_manifest
from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--compare-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    base = Path('/home/sasaki/datasets/openloris')
    reference = base / 'corridor1-1-m8-dense-matching-replay-v1'
    original = base / 'corridor1-1-m8-dense256x2-10k-ann-gap128-local32-8n-v1'
    features = base / 'corridor1-1-m8-dense256x2-full10k-v2/features'
    binary = args.binary.resolve(strict=True)
    compare = args.compare_binary.resolve(strict=True)
    lines = (reference / 'match-worker.plan').read_text().splitlines()
    rows = [line for line in lines if line.startswith('shard ')]
    if not 2 <= args.shards <= len(rows):
        parser.error('--shards must be 2..2500')
    selected = [rows[index * (len(rows) - 1) // (args.shards - 1)] for index in range(args.shards)]
    validate_feature_manifest(original / 'features.json', features)
    output = args.output.resolve()
    output.mkdir()
    (output / 'candidates').mkdir()
    (output / 'matches').mkdir()
    for row in selected:
        fields = row.split()
        if len(fields) != 5 or fields[2] != f'candidates/candidate-{int(fields[1]):06d}.txt' or fields[3] != f'matches/verified-{int(fields[1]):06d}.vps':
            raise RuntimeError('Unsafe or unexpected shard paths')
        source = reference / fields[2]
        if sha(source) != fields[4]:
            raise RuntimeError('Retained candidate shard changed')
        if 'pairs 32' not in source.read_text().splitlines():
            raise RuntimeError('Expected 32-pair shards')
        shutil.copyfile(source, output / fields[2])
    plan = [f'pairs {32 * args.shards}' if line.startswith('pairs ') else
            f'shards {args.shards}' if line.startswith('shards ') else line
            for line in lines if not line.startswith('shard ')] + selected
    (output / 'match-worker.plan').write_text('\n'.join(plan) + '\n')
    recipe = original / 'timing/persistent-match.time.txt'
    command = shlex.split(shlex.split(recipe.read_text().splitlines()[0].split('Command being timed: ', 1)[1])[0])
    command[0] = str(binary)
    if '--stream-match-features' not in command:
        raise RuntimeError('Missing streaming matching recipe')
    for flag, path in [('--persistent-match-worker-plan', output / 'match-worker.plan'),
                       ('--features-dir', features), ('--out-colmap', output / 'unused-model')]:
        command[command.index(flag) + 1] = str(path)
    command.append('--shared-snapshot-envelope')
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '1800s', *command]
    report = {'status': 'running', 'command': invocation, 'binary_sha256': sha(binary),
              'compare_binary_sha256': sha(compare), 'shard_ids': [int(row.split()[1]) for row in selected],
              'plan_sha256': sha(output / 'match-worker.plan'),
              'reference_plan_sha256': sha(reference / 'match-worker.plan'),
              'feature_manifest_sha256': sha(original / 'features.json'),
              'scope': 'Selected real matching shards; not full10k matching unless all2500 requested, not E2E.'}
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, RAYON_NUM_THREADS='4', MALLOC_ARENA_MAX='1'))
    expected = {Path(row.split()[3]).name for row in selected}
    actual = {path.name for path in (output / 'matches').glob('*.vps')}
    comparison = None
    if result.returncode == 0 and expected == actual:
        comparison = subprocess.run([str(compare), str(output / 'matches'), str(reference / 'matches')],
                                    capture_output=True, text=True, timeout=180)
    report.update(exit_code=result.returncode, membership_matches=expected == actual,
                  comparison_exit_code=comparison.returncode if comparison else None,
                  comparison_output=(comparison.stdout + comparison.stderr) if comparison else None,
                  measurement=parse_gnu_time(output / 'time.txt'),
                  status='pass' if comparison and comparison.returncode == 0 else 'fail')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    summary = {key: value for key, value in report.items() if key not in ('command', 'shard_ids')}
    summary['shards'] = len(report['shard_ids'])
    print(json.dumps(summary, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
