#!/usr/bin/env python3
"""Launch detached measured work as a persistent systemd user service."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from replay_native_candidates import sha


def launch_argv(unit, root, cwd, timeout, command):
    if not re.fullmatch(r'visloc-[a-z0-9-]+', unit):
        raise ValueError('Unit name must start with visloc- and contain lowercase letters/digits/hyphens')
    if not command or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Require a command and finite positive timeout')
    return ['systemd-run', '--user', '--unit=' + unit, '--expand-environment=no',
            '--property=MemoryMax=2G', '--property=MemorySwapMax=0',
            '--property=RemainAfterExit=yes', '--property=KillMode=control-group',
            '--property=RuntimeMaxSec=' + str(math.ceil(timeout) + 30),
            '--working-directory=' + str(cwd),
            '--property=StandardOutput=append:' + str(root / 'service.log'),
            '--property=StandardError=append:' + str(root / 'service.log'),
            sys.executable, str(root / 'measure_native_scope.py'),
            '--report', str(root / 'measurement.json'), '--timeout', str(timeout), '--', *command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    root = args.output.resolve()
    argv = launch_argv(args.unit, root, Path.cwd(), args.timeout, command)
    root.mkdir()  # Never overwrite prior measurements.
    monitor = Path(__file__).with_name('measure_native_scope.py')
    shutil.copy2(monitor, root / monitor.name)
    digest = sha(monitor)
    if sha(root / monitor.name) != digest:
        raise RuntimeError('Monitor copy changed')
    result = subprocess.run(argv, capture_output=True, text=True)
    report = {'unit': args.unit + '.service', 'command': argv, 'monitor_sha256': digest,
              'launch_exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'status': 'launched-not-completed' if result.returncode == 0 else 'launch-failed'}
    (root / 'launch.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
