#!/usr/bin/env python3
"""Read-only M8 model/match duplicate inventory; never applies replacements."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import stat

from benchmark_electro import atomic_json
from replay_native_candidates import sha


def inventory(root):
    sizes = defaultdict(list)
    seen = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
        for name in sorted(files):
            path = Path(base) / name
            relative = path.relative_to(root)
            if not relative.parts[0].startswith('corridor1-1-m8-'):
                continue
            if name not in ('images.txt', 'points3D.txt') and path.suffix != '.vps':
                continue
            before = path.lstat()
            key = (before.st_dev, before.st_ino)
            if not stat.S_ISREG(before.st_mode) or before.st_size < 1024 * 1024 or key in seen:
                continue
            seen.add(key)
            sizes[before.st_size].append((path, before))
    groups = []
    for rows in sizes.values():
        if len(rows) < 2:
            continue
        hashes = defaultdict(list)
        for path, before in rows:
            digest = sha(path)
            after = path.lstat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('File changed during inventory: ' + str(path))
            hashes[digest].append(dict(path=str(path.relative_to(root)), inode=after.st_ino,
                                       device=after.st_dev, links=after.st_nlink,
                                       allocated_bytes=after.st_blocks * 512))
        for digest, files in hashes.items():
            if len(files) < 2:
                continue
            # Prefer retaining an already shared inode; never replace multi-link targets.
            files.sort(key=lambda row: (-row['links'], row['path']))
            savings = sum(row['allocated_bytes'] for row in files[1:] if row['links'] == 1)
            groups.append(dict(sha256=digest, files=files, potential_freed_bytes=savings))
    return dict(status='inventory-only-not-approved-for-application', root=str(root),
                potential_freed_bytes=sum(g['potential_freed_bytes'] for g in groups),
                groups=sorted(groups, key=lambda g: -g['potential_freed_bytes']),
                caveat='Requires quiescence, exact target review and rehash before application. No files changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inventory(Path('/home/sasaki/datasets/openloris'))
    atomic_json(args.output, report)
    print({k: v for k, v in report.items() if k != 'groups'})
    print('groups:', len(report['groups']))


if __name__ == '__main__':
    main()
