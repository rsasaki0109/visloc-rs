#!/usr/bin/env python3
"""Validate immutable hardlink publication against the retained real 10k bank."""
import argparse
import json
import os
from pathlib import Path
import time

from merge_sift_supplements import write_bank
from replay_native_candidates import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir()
    dataset = Path('/home/sasaki/datasets/openloris')
    base = dataset / 'corridor1-1-m5/feature-extract/features'
    tier = dataset / 'corridor1-1-m5/tiers/tier-10000/features256'
    dense = dataset / 'corridor1-1-m8-dense256x2-full10k-v2/features'
    reference = dataset / 'corridor1-1-m8-adaptive-bank-publication-v1/features'
    selection_path = dataset / 'corridor1-1-m8-adaptive32-halo8-10k-v1/selection.json'
    selection = json.loads(selection_path.read_text())
    selected = {Path(name).stem + '_features.txt' for name in selection['image_names']}
    report = {'status': 'running', 'selection_sha256': sha(selection_path),
              'scope': 'Retained-bank publication parity and incremental file allocation, not extraction, E2E, RSS or independent backup.'}

    def save():
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        assert {p.name for p in tier.glob('*_features.txt')} == {p.name for p in base.glob('*_features.txt')}
        assert all(p.resolve() == base / p.name for p in tier.glob('*_features.txt'))
        before = {p.name: sha(p) for p in base.glob('*_features.txt')}
        expected = {p.name: sha(p) for p in reference.glob('*_features.txt')}
        assert len(before) == len(expected) == 10000 and len(selected) == 692
        started = time.monotonic()
        report['publication'] = write_bank(base, dense, selection, root / 'features', hardlink_unselected=True)
        report['publication_wall_seconds'] = time.monotonic() - started
        actual = {p.name: sha(p) for p in (root / 'features').iterdir()}
        assert actual == expected
        linked = allocated = shared_bytes = 0
        for name in actual:
            path = root / 'features' / name
            shared = os.path.samefile(path, base / name)
            assert shared == (name not in selected)
            linked += shared
            if shared:
                shared_bytes += path.stat().st_size
            else:
                allocated += path.stat().st_blocks * 512
        report['resume'] = write_bank(base, dense, selection, root / 'features',
                                      resume=True, hardlink_unselected=True)
        assert report['resume']['reused'] == 10000 and report['resume']['written'] == 0
        assert before == {p.name: sha(p) for p in base.glob('*_features.txt')}
        assert actual == {p.name: sha(p) for p in (root / 'features').iterdir()}
        report.update(status='pass', files=10000, selected_independent_files=692,
                      linked_files=linked, shared_logical_bytes=shared_bytes,
                      new_regular_file_allocated_bytes=allocated,
                      allocation_excludes='directory entries, metadata, logs and report',
                      all_reference_bytes_equal=True, base_bytes_unchanged=True)
    except BaseException as error:
        report.update(status='fail', error=repr(error))
        save()
        raise
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
