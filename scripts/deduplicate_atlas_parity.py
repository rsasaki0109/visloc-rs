#!/usr/bin/env python3
"""Deduplicate only frozen, hash-identical Schur diagnostic model files.

Preserves paths and bytes, but shares inode metadata and future writes. These
completed output roots must never be reused or overwritten by another run.
No raw images, feature banks, logs, measurements or ON geometry are targeted.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

from benchmark_electro import atomic_json
from replay_native_candidates import sha


def checked_stat(path, digest):
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError('Symlink in model path: ' + str(path))
    stat = path.stat()
    if not path.is_file() or sha(path) != digest:
        raise ValueError('Model changed: ' + str(path))
    return stat


def merge_file(source, target, digest):
    before = checked_stat(target, digest)
    canonical = checked_stat(source, digest)
    if (before.st_dev, before.st_ino) == (canonical.st_dev, canonical.st_ino):
        return 0
    if before.st_dev != canonical.st_dev:
        raise ValueError('Hardlinks require one filesystem')
    temporary = target.with_name(target.name + '.dedup-' + uuid.uuid4().hex)
    os.link(source, temporary)
    try:
        current = checked_stat(target, digest)
        if (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size) != (
                before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size):
            raise ValueError('Target changed during deduplication')
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    checked_stat(target, digest)
    return before.st_blocks * 512 if before.st_nlink == 1 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.audit.exists():
        raise FileExistsError(args.audit)
    dataset = Path('/home/sasaki/datasets/openloris')
    names = ('schur-parity-off-v1', 'schur-parity-on-v1', 'schur-feasibility-v1')
    canonical = dataset / 'corridor1-1-m8-schur-feasibility-v2'
    evidence = Path(__file__).resolve().parents[1] / 'benchmarks/electro/m8-schur-feasibility-v2.json'
    expected = json.loads(evidence.read_text())['reference_file_sha256']
    if len(expected) != 14 or any(Path(name).is_absolute() or '..' in Path(name).parts for name in expected):
        raise ValueError('Invalid frozen model manifest')
    for name in (*names, 'schur-feasibility-v2'):
        state = subprocess.check_output(['systemctl', '--user', 'show', 'visloc-' + name + '.service',
                                         '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'], text=True)
        if set(state.splitlines()) != {'MainPID=0', 'SubState=exited', 'Result=success'}:
            raise ValueError('Require completed successful source and targets: ' + name)
    rows = []
    for name in names:
        for relative, digest in expected.items():
            source = canonical / relative
            target = dataset / ('corridor1-1-m8-' + name) / relative
            source_stat = checked_stat(source, digest)
            target_stat = checked_stat(target, digest)
            if source_stat.st_dev != target_stat.st_dev:
                raise ValueError('Different filesystems')
            rows.append(dict(source=str(source), target=str(target), sha256=digest,
                             original_inode=target_stat.st_ino, original_links=target_stat.st_nlink,
                             allocated_bytes=target_stat.st_blocks * 512))
    report = dict(status='validated', applied=args.apply, rows=rows,
                  caveat='Shared inode/metadata: never overwrite these frozen model paths',
                  freed_allocated_bytes=0)
    atomic_json(args.audit, report)
    if args.apply:
        try:
            for row in rows:
                freed = merge_file(Path(row['source']), Path(row['target']), row['sha256'])
                row['merged'] = True
                report['freed_allocated_bytes'] += freed
                atomic_json(args.audit, report)
            report['status'] = 'completed'
        except BaseException as error:
            report.update(status='failed-partial-preserved', error=str(error))
            raise
        finally:
            atomic_json(args.audit, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
