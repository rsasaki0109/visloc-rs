#!/usr/bin/env python3
"""Kill our own partial converter process, then validate resume and prior bytes."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    source = args.source.resolve(strict=True)
    expected = {path.name for path in source.glob('*.vps')}
    if len(expected) < 500:
        parser.error('Use at least 500 shards for a partial-batch interruption')
    root = args.output.resolve()
    root.mkdir()
    destination = root / 'matches'
    command = [str(binary), str(source), str(destination)]
    killed = False
    with (root / 'interrupted.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None and time.monotonic() < deadline:
                completed = len(list(destination.glob('*.vps')))
                if 250 <= completed < len(expected):
                    process.kill()  # Only this script's own child.
                    killed = True
                    break
                time.sleep(0.02)
            if process.poll() is None and not killed:
                process.kill()
            exit_code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    prior = {path.name: sha(path) for path in destination.glob('*.vps')}
    if not killed or exit_code != -signal.SIGKILL or not 0 < len(prior) < len(expected):
        raise RuntimeError('Did not confirm SIGKILL during a partial conversion')
    with (root / 'resume.log').open('w') as log:
        result = subprocess.run([*command, '--resume'], stdout=log, stderr=subprocess.STDOUT, timeout=180)
    actual = {path.name for path in destination.glob('*.vps')}
    unchanged = all(sha(destination / name) == digest for name, digest in prior.items())
    report = {'status': 'pass' if result.returncode == 0 and actual == expected and unchanged else 'fail',
              'binary_sha256': sha(binary), 'source': str(source),
              'interrupted_exit_code': exit_code, 'completed_before_resume': len(prior),
              'expected_shards': len(expected), 'final_shards': len(actual),
              'resume_exit_code': result.returncode, 'prior_chunks_byte_unchanged': unchanged,
              'scope': 'SIGKILL during partial real-bank conversion, not mapper or full native pipeline restart. Converter readback compares every output record to its source.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
