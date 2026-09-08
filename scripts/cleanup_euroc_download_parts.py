#!/usr/bin/env python3
"""Verify EuRoC download parts against the retained archive before removal."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--remove-verified-parts', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    archive = root / 'machine_hall.zip'
    expected = (root / 'machine_hall.zip.sha256').read_text().split()[0]
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError('Retained archive SHA-256 mismatch')
    parts = []
    with archive.open('rb') as stream:
        for part in sorted((root / 'ranges').glob('*.part')):
            match = re.fullmatch(r'(\d+)-(\d+)\.part', part.name)
            if part.is_symlink() or not part.is_file() or not match:
                raise RuntimeError(f'Unexpected part: {part}')
            start, end = map(int, match.groups())
            if end < start or part.stat().st_size != end - start + 1:
                raise RuntimeError(f'Invalid range size: {part}')
            stream.seek(start)
            part_digest = hashlib.sha256()
            with part.open('rb') as incoming:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                    if stream.read(len(chunk)) != chunk:
                        raise RuntimeError(f'Content mismatch: {part}')
                    part_digest.update(chunk)
            parts.append((part, part.stat(), part_digest.hexdigest()))
    # Complete all validation before any deletion. Headers and logs are retained.
    for part, original, _ in parts:
        current = part.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (
                original.st_ino, original.st_size, original.st_mtime_ns):
            raise RuntimeError(f'Part changed during verification: {part}')
    report = {
        'archive': str(archive), 'archive_sha256': expected,
        'part_count': len(parts), 'bytes': sum(s.st_size for _, s, _ in parts),
        'action': 'remove' if args.remove_verified_parts else 'audit_only',
        'parts': [{'name': p.name, 'bytes': s.st_size, 'sha256': h}
                  for p, s, h in parts],
    }
    print(json.dumps(report), flush=True)
    if args.remove_verified_parts:
        for part, _, _ in parts:
            part.unlink()
        print('Removed verified duplicate parts; retained archive and headers.', flush=True)


if __name__ == '__main__':
    main()
