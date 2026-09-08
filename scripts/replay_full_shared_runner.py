#!/usr/bin/env python3
"""Full retained dense schedule through prepare/match/merge/completed resume."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--merge-binary', type=Path, required=True)
    parser.add_argument('--compare-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('Linux process-group timeout handling required')
    root = args.output.resolve()
    if shutil.disk_usage(root.parent).free < 3 * 1024**3:
        parser.error('Need at least 3 GiB free for the run and temporary merge files')
    root.mkdir()
    binaries = {}
    for name, source in [('sfm', args.binary), ('merge', args.merge_binary), ('compare', args.compare_binary)]:
        digest = sha(source.resolve(strict=True))
        target = root / name
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise RuntimeError('Binary copy mismatch')
        binaries[name] = {'path': str(target), 'sha256': digest}
    base = Path('/home/sasaki/datasets/openloris')
    candidates = base / 'corridor1-1-m8-dense-candidates-replay-v1/candidates.txt'
    reference = base / 'corridor1-1-m8-dense-matching-replay-v1/matches'
    reference_merged = base / 'corridor1-1-m8-dense-merge-replay-v1/verified-merged.vps'
    runner = Path(__file__).resolve().with_name('benchmark_electro.py')
    runner_hash = sha(runner)
    common = [sys.executable, str(runner), '--features-dir', str(base / 'corridor1-1-m8-dense256x2-full10k-v2/features'),
              '--calibration-dir', str(base / 'corridor1-1-m5/tiers/tier-10000/calibration'),
              '--artifact-root', str(root / 'run'), '--candidate-manifest', str(candidates),
              '--binary', binaries['sfm']['path'], '--merge-binary', binaries['merge']['path'],
              '--pairs-per-shard', '32', '--pair-source', 'temporal-pyramid',
              '--temporal-pyramid-max-offset', '32', '--retrieval-topk', '128',
              '--retrieval-min-frame-gap', '128', '--candidate-budget', '80000']
    match = [*common, '--match', '--persistent-matcher', '--stream-match-features',
             '--shared-snapshot-envelope', '--resume']
    report = {'status': 'running', 'binaries': binaries, 'runner_sha256': runner_hash,
              'candidate_sha256': sha(candidates), 'reference_merged_sha256': sha(reference_merged),
              'phases': {}, 'scope': 'Retained10k features/candidates through Python matching+merge runner. No extraction, candidate generation, mapping, quality or native E2E claim.'}

    def save():
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(name, command):
        if sha(runner) != runner_hash:
            raise RuntimeError('Runner source changed during experiment')
        print(f'Running {name}', flush=True)
        started = time.monotonic()
        with (root / f'{name}.log').open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=2400)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise
        report['phases'][name] = {'exit_code': code, 'wall_seconds': time.monotonic() - started, 'command': command}
        save()
        if code:
            raise RuntimeError(f'{name} failed; see {root / (name + ".log")}')

    save()
    try:
        run('prepare', [*common, '--prepare'])
        run('match', match)
        outputs = root / 'run/matches'
        index = json.loads((outputs / 'index.json').read_text())
        if len(index['shards']) != 2500 or any(entry['status'] != 'complete' or not entry.get('snapshot_envelope') for entry in index['shards']):
            raise RuntimeError('Incomplete or unbound match index')
        expected = {path.name for path in reference.glob('*.vps')}
        paths = sorted(outputs.glob('*.vps'))
        if {path.name for path in paths} != expected:
            raise RuntimeError('Output membership mismatch')
        run('compare', [binaries['compare']['path'], str(outputs), str(reference)])
        merged = root / 'run/mapping/verified-merged.vps'
        if sha(merged) != report['reference_merged_sha256']:
            raise RuntimeError('Merged snapshot differs')
        log = outputs / 'persistent-match.log'
        prior = {path.name: (sha(path), path.stat().st_mtime_ns) for path in paths}
        log_before = (sha(log), log.stat().st_mtime_ns)
        run('completed-resume', match)
        unchanged = all((sha(path), path.stat().st_mtime_ns) == prior[path.name] for path in paths)
        no_worker = (sha(log), log.stat().st_mtime_ns) == log_before
        report.update(prior_shards_unchanged=unchanged, worker_log_unchanged=no_worker,
                      merged_sha256=sha(merged), shard_count=len(paths))
        if not unchanged or not no_worker or sha(merged) != report['reference_merged_sha256']:
            raise RuntimeError('Completed resume changed output or reran worker')
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='fail', error=str(error))
        save()
        raise
    save()
    print(json.dumps({key: value for key, value in report.items() if key != 'phases'}, indent=2))


if __name__ == '__main__':
    main()
