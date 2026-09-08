#!/usr/bin/env python3
"""Linux real runner process-group SIGKILL/restart versus uninterrupted control."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark_electro import write_candidate_manifest
from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--merge-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('This process-group interruption fixture requires Linux')
    binary = args.binary.resolve(strict=True)
    merger = args.merge_binary.resolve(strict=True)
    root = args.output.resolve()
    root.mkdir()
    features = root / 'features'
    features.mkdir()
    base = Path('/home/sasaki/datasets/openloris')
    source = base / 'corridor1-1-m8-dense256x2-full10k-v2/features'
    selected = sorted(source.glob('*_features.txt'))[:128]
    if len(selected) != 128:
        raise RuntimeError('Expected 128 real feature files')
    for path in selected:
        (features / path.name).symlink_to(path.resolve(strict=True))
    names = [path.name.removesuffix('_features.txt') + '.png' for path in selected]
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, min(i + 9, len(names)))]
    runner = Path(__file__).resolve().with_name('benchmark_electro.py')
    common = [sys.executable, str(runner), '--features-dir', str(features),
              '--calibration-dir', str(base / 'corridor1-1-m5/tiers/tier-10000/calibration'),
              '--binary', str(binary), '--merge-binary', str(merger), '--pairs-per-shard', '8']

    def run(command, log):
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError(f'Runner failed ({result.returncode}); see {log}')

    commands = {}
    for name in ('control', 'resumed'):
        output = root / name
        output.mkdir()
        write_candidate_manifest(output / 'candidates.txt', names, pairs)
        command = [*common, '--artifact-root', str(output)]
        run([*command, '--prepare'], root / f'{name}-prepare.log')
        commands[name] = [*command, '--match', '--persistent-matcher', '--stream-match-features',
                          '--shared-snapshot-envelope', '--resume']
    run(commands['control'], root / 'control.log')
    interrupted = root / 'resumed'
    worker_log = interrupted / 'matches/persistent-match.log'
    killed = False
    with (root / 'interrupted.log').open('w') as stream:
        process = subprocess.Popen(commands['resumed'], stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while process.poll() is None and time.monotonic() < deadline:
                if worker_log.exists() and worker_log.read_text().count('persistent-match-complete ') >= 8:
                    os.killpg(process.pid, signal.SIGKILL)  # Our isolated child group only.
                    killed = True
                    break
                time.sleep(0.02)
            if process.poll() is None and not killed:
                os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    prior = {path.name: sha(path) for path in (interrupted / 'matches').glob('*.vps')}
    expected = {path.name: sha(path) for path in (root / 'control/matches').glob('*.vps')}
    if not killed or exit_code != -signal.SIGKILL or not 0 < len(prior) < len(expected):
        raise RuntimeError('Did not confirm a partial runner SIGKILL')
    run(commands['resumed'], root / 'resume.log')
    actual = {path.name: sha(path) for path in (interrupted / 'matches').glob('*.vps')}
    control_merged = sha(root / 'control/mapping/verified-merged.vps')
    resumed_merged = sha(interrupted / 'mapping/verified-merged.vps')
    unchanged = all(actual.get(name) == digest for name, digest in prior.items())
    report = {'status': 'pass' if actual == expected and unchanged and control_merged == resumed_merged else 'fail',
              'binary_sha256': sha(binary), 'merge_binary_sha256': sha(merger),
              'images': len(names), 'candidate_pairs': len(pairs), 'shards': len(expected),
              'interrupted_exit_code': exit_code, 'completed_before_resume': len(prior),
              'prior_chunks_unchanged': unchanged, 'all_shards_byte_equal': actual == expected,
              'control_merged_sha256': control_merged, 'resumed_merged_sha256': resumed_merged,
              'scope': '128-image real-feature matching+merge runner restart, not feature extraction, mapper restart, full10k or continuous E2E.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
