#!/usr/bin/env python3
"""Reproduce native matching shards from independently replayed candidates."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from benchmark_electro import (
    parse_candidate_manifest_with_metadata,
    validate_feature_manifest,
    write_candidate_manifest,
)
from replay_native_candidates import sha


def main():
    base = Path('/home/sasaki/datasets/openloris')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path,
                        default=base / 'corridor1-1-m8-native-candidates-replay-v1')
    parser.add_argument('--binary', type=Path,
                        default=base / 'corridor1-1-m8-extract-resume-pilot-v1/extract-3ae253a')
    parser.add_argument('--binary-sha256',
                        default='8cfa9c53fcaea5d6305dbdd3018b8381ae214806751e6ee3dc85bb8692a8da34')
    args = parser.parse_args()
    reference = base / 'corridor1-1-m8-native-rig-runner-10k-v1'
    candidate_root = args.candidate_root.resolve(strict=True)
    output = base / 'corridor1-1-m8-native-matching-replay-v1'
    binary = args.binary.resolve(strict=True)
    expected_binary = args.binary_sha256
    candidate_report = json.loads((candidate_root / 'report.json').read_text())
    if candidate_report['status'] != 'pass' or sha(binary) != expected_binary:
        raise RuntimeError('Candidate replay or frozen binary prerequisite failed')
    candidates = candidate_root / 'candidates.txt'
    if sha(candidates) != sha(reference / 'candidates.txt'):
        raise RuntimeError('Candidate contents changed')
    names, pairs, metadata = parse_candidate_manifest_with_metadata(candidates)
    if len(names) != 10000 or len(pairs) != 70000:
        raise RuntimeError('Unexpected candidate envelope')
    output.mkdir()
    (output / 'candidates').mkdir()
    (output / 'matches').mkdir()
    # Use the recorded plan as a frozen schedule, not retained match outputs.
    plan = (reference / 'match-worker.plan').read_text()
    shard_rows = [line.split() for line in plan.splitlines() if line.startswith('shard ')]
    if len(shard_rows) != 2188:
        raise RuntimeError('Unexpected shard count')
    for index, fields in enumerate(shard_rows):
        expected_candidate = f'candidates/candidate-{index:06d}.txt'
        expected_snapshot = f'matches/verified-{index:06d}.vps'
        if len(fields) != 5 or fields[1:4] != [str(index), expected_candidate, expected_snapshot]:
            raise RuntimeError('Unexpected shard order or unsafe paths')
        destination = output / expected_candidate
        write_candidate_manifest(destination, names, pairs[index * 32:(index + 1) * 32],
                                 metadata=metadata)
        if sha(destination) != fields[4]:
            raise RuntimeError(f'Regenerated shard differs: {index}')
    features = base / 'corridor1-1-m5/tiers/tier-10000/features256'
    validate_feature_manifest(reference / 'features.json', features)
    shutil.copy2(reference / 'match-worker.plan', output / 'match-worker.plan')
    timing = reference / 'timing/persistent-match.time.txt'
    first_line = timing.read_text().splitlines()[0]
    command = shlex.split(shlex.split(first_line.split('Command being timed: ', 1)[1])[0])
    command[0] = str(binary)
    for flag, path in [('--persistent-match-worker-plan', output / 'match-worker.plan'),
                       ('--out-colmap', output / 'matches/unused-model-persistent')]:
        command[command.index(flag) + 1] = str(path)
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '1800s', *command]
    report = {'status': 'running', 'command': invocation, 'rayon_num_threads': 4,
              'binary_sha256': expected_binary, 'candidate_sha256': sha(candidates),
              'plan_sha256': sha(output / 'match-worker.plan'),
              'feature_manifest_sha256': sha(reference / 'features.json'),
              'regenerated_candidate_shards': len(shard_rows),
              'scope': 'matching shards only; retained base features and schedule; no merge'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, env=dict(os.environ, RAYON_NUM_THREADS='4'),
                                stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode, wall_seconds=time.monotonic() - started)
    differences = []
    for fields in shard_rows:
        relative = fields[3]
        actual = output / relative
        if not actual.is_file() or sha(actual) != sha(reference / relative):
            differences.append(relative)
    expected_names = {Path(fields[3]).name for fields in shard_rows}
    actual_names = {path.name for path in (output / 'matches').glob('*.vps')}
    report['extra_snapshots'] = sorted(actual_names - expected_names)
    report['different_or_missing_snapshots'] = differences
    report['status'] = 'pass' if result.returncode == 0 and not differences and not report['extra_snapshots'] else 'fail'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
