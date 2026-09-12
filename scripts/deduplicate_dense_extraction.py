#!/usr/bin/env python3
"""Share verified dense extraction bytes without removing feature paths.

Only the completed full-dense-extraction-v1 output is replaced with hardlinks
to its reference bank. Never write into either frozen bank after this action.
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
    dataset = Path('/home/sasaki/datasets/openloris')
    output = dataset / 'corridor1-1-m8-full-dense-extraction-v1/features'
    source = dataset / 'corridor1-1-m8-dense256x2-full10k-v2/features'
    evidence_path = Path(__file__).resolve().parents[1] / 'benchmarks/electro/m8-full-dense-extraction-v1.json'
    evidence = json.loads(evidence_path.read_text())
    report_path = Path(evidence['report'])
    if sha(report_path) != evidence['report_sha256']:
        raise ValueError('Extraction report changed')
    report = json.loads(report_path.read_text())
    expected = report['reference_feature_sha256']
    if report['status'] != 'pass' or len(expected) != 20000 or expected != report['output_feature_sha256']:
        raise ValueError('Require completed full exact extraction')
    if any(Path(name).name != name or not name.endswith(('_features.txt', '_loci.txt')) for name in expected):
        raise ValueError('Unsafe feature name')
    state = subprocess.check_output(['systemctl', '--user', 'show', evidence['service'],
                                     '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'], text=True)
    if set(state.splitlines()) != {'MainPID=0', 'SubState=exited', 'Result=success'}:
        raise ValueError('Require terminal successful extraction')
    estimated = 0
    for name, digest in expected.items():
        original = checked_stat(output / name, digest)
        canonical = checked_stat(source / name, digest)
        if original.st_dev != canonical.st_dev:
            raise ValueError('Different filesystems')
        if original.st_ino != canonical.st_ino and original.st_nlink == 1:
            estimated += original.st_blocks * 512
    audit = dict(status='validated', applied=args.apply, files=len(expected),
                 source=str(source), output=str(output),
                 extraction_report=str(report_path), extraction_report_sha256=sha(report_path),
                 estimated_freed_allocated_bytes=estimated, freed_allocated_bytes=0,
                 processed_files=0,
                 caveat='Paths and bytes preserved; shared inode/metadata, frozen banks must never be overwritten')
    atomic_json(args.audit, audit)
    if args.apply:
        try:
            for name, digest in expected.items():
                audit['freed_allocated_bytes'] += merge_file(source / name, output / name, digest)
                audit['processed_files'] += 1
                if audit['processed_files'] % 500 == 0:
                    atomic_json(args.audit, audit)
            # Independently re-read every resulting path before claiming completion.
            for name, digest in expected.items():
                checked_stat(output / name, digest)
                if not (source / name).samefile(output / name):
                    raise ValueError('Missing shared inode: ' + name)
            audit['status'] = 'completed'
        except BaseException as error:
            audit.update(status='failed-partial-preserved', error=str(error))
            raise
        finally:
            atomic_json(args.audit, audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
