#!/usr/bin/env python3
"""Share audited base-v3 feature bytes; preserve sidecars, paths and evidence.

Both feature banks become frozen shared-inode data: never overwrite either.
"""
import argparse
import json
from pathlib import Path
import subprocess

from benchmark_electro import atomic_json
from deduplicate_atlas_parity import checked_stat, merge_file
from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.audit.exists():
        raise FileExistsError(args.audit)
    evidence = json.loads((Path(__file__).resolve().parents[1] /
                           'benchmarks/electro/m8-full-base-extraction-v3-audit.json').read_text())
    dataset = Path(evidence['dataset_root'])
    report_path = dataset / evidence['report']
    if sha(report_path) != evidence['report_sha256']:
        raise ValueError('Report changed')
    state = subprocess.check_output(['systemctl', '--user', 'show', evidence['unit'],
                                    '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'], text=True)
    if set(state.splitlines()) != {'MainPID=0', 'SubState=exited', 'Result=success'}:
        raise ValueError('Require successful terminal extraction')
    report = json.loads(report_path.read_text())
    expected = report['output_feature_sha256']
    if report['status'] != 'pass' or len(expected) != 10000 or expected != report['reference_feature_sha256']:
        raise ValueError('Require full exact parity')
    reference = dataset / evidence['independent_audit']['reference']
    source = dataset / 'corridor1-1-m5/feature-extract/features'
    target = report_path.parent / 'features'
    if any(Path(n).name != n or not n.endswith('_features.txt') for n in expected):
        raise ValueError('Unsafe feature name')
    for root in (reference, target):
        if {p.name for p in root.glob('*_features.txt')} != set(expected):
            raise ValueError('Feature membership changed')
    estimated = 0
    for name, digest in expected.items():
        if (reference / name).resolve(strict=True) != source / name:
            raise ValueError('Unexpected reference backing file')
        a, b = checked_stat(source / name, digest), checked_stat(target / name, digest)
        if a.st_dev != b.st_dev:
            raise ValueError('Different filesystems')
        if a.st_ino != b.st_ino and b.st_nlink == 1:
            estimated += b.st_blocks * 512
    audit = dict(status='validated', applied=args.apply, source=str(source), target=str(target),
                 report_sha256=sha(report_path), files=len(expected), processed_files=0,
                 estimated_freed_allocated_bytes=estimated, freed_allocated_bytes=0,
                 caveat='Shared inode metadata; never overwrite either bank. Sidecars untouched.')
    atomic_json(args.audit, audit)
    if args.apply:
        try:
            for name, digest in expected.items():
                audit['freed_allocated_bytes'] += merge_file(source / name, target / name, digest)
                audit['processed_files'] += 1
                if audit['processed_files'] % 500 == 0:
                    atomic_json(args.audit, audit)
            for name, digest in expected.items():
                checked_stat(target / name, digest)
                if not (source / name).samefile(target / name):
                    raise ValueError('Missing shared inode')
            audit['status'] = 'completed'
        except BaseException as error:
            audit.update(status='failed-partial-preserved', error=str(error))
            raise
        finally:
            atomic_json(args.audit, audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
