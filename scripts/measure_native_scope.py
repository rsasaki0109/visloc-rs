#!/usr/bin/env python3
"""Measure a command inside a dedicated, already memory-capped cgroup v2 scope."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def validate_limits(root):
    maximum = (root / 'memory.max').read_text().strip()
    swap = (root / 'memory.swap.max').read_text().strip()
    if maximum != str(2 * 1024**3) or swap != '0':
        raise ValueError('Require dedicated MemoryMax=2G MemorySwapMax=0 scope')


def resident_kib(status):
    for line in status.splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])
    return 0


def scope_pids(root):
    # Traverse descendant cgroups too; do not count the same PID twice.
    pids = set()
    for file in [root / 'cgroup.procs', *root.glob('**/cgroup.procs')]:
        try:
            pids.update(file.read_text().split())
        except FileNotFoundError:
            pass
    return pids


def sample_rss(root):
    total = 0
    pids = scope_pids(root)
    for pid in pids:
        try:
            total += resident_kib((Path('/proc') / pid / 'status').read_text())
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.timeout < float('inf'):
        parser.error('Require command and finite positive timeout')
    entry = Path('/proc/self/cgroup').read_text().strip()
    if not entry.startswith('0::/'):
        raise ValueError('Require unified cgroup v2')
    root = Path('/sys/fs/cgroup') / entry[3:].lstrip('/')
    validate_limits(root)
    if scope_pids(root) != {str(os.getpid())}:
        raise ValueError('Scope must initially contain only the measurement process')
    # Exclusive creation protects previous measurements.
    with args.report.open('x') as output:
        report = {'status': 'running', 'command': command, 'cgroup': str(root),
                  'rss_sampling_interval_seconds': 0.05,
                  'scope': 'Sampled aggregate RSS includes monitor; shared pages may be counted multiple times. cgroup peak includes charged cache/kernel memory and is not RSS.'}
        def save():
            output.seek(0)
            json.dump(report, output, indent=2)
            output.write('\n')
            output.truncate()
            output.flush()
        save()
        before = (root / 'memory.events').read_text()
        start = time.monotonic()
        process = None
        peak = sample_rss(root)
        try:
            process = subprocess.Popen(command, start_new_session=True)
            while process.poll() is None:
                peak = max(peak, sample_rss(root))
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError('Command deadline exceeded')
                time.sleep(0.05)
            if scope_pids(root) != {str(os.getpid())}:
                raise RuntimeError('Command left descendants running')
            events_before = dict(line.split() for line in before.splitlines())
            events_after = dict(line.split() for line in (root / 'memory.events').read_text().splitlines())
            if any(int(events_after.get(k, 0)) > int(events_before.get(k, 0))
                   for k in ('oom', 'oom_kill', 'oom_group_kill')):
                raise RuntimeError('Memory OOM event during command')
            report.update(exit_code=process.returncode,
                          status='pass' if process.returncode == 0 else 'fail')
        except BaseException as error:
            report.update(status='fail', error=repr(error))
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
                report['exit_code'] = process.returncode
            raise
        finally:
            report.update(wall_seconds=time.monotonic() - start,
                          sampled_aggregate_peak_rss_kib=peak,
                          cgroup_memory_peak_bytes=int((root / 'memory.peak').read_text()),
                          memory_events_before=before,
                          memory_events_after=(root / 'memory.events').read_text())
            save()
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
