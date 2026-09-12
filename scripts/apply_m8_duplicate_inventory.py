#!/usr/bin/env python3
"""Apply the fixed reviewed inventory, retaining every path and file content.

Run only while M8 experiments are quiescent. All participating files become
frozen shared-inode artifacts; new experiments must use fresh output roots.
"""
import argparse
import json
from pathlib import Path

from benchmark_electro import atomic_json
from deduplicate_atlas_parity import checked_stat, merge_file
from replay_native_candidates import sha

INVENTORY_SHA = 'fe8204077bc0a252229bbb47e2ae790be83b0e242ccac2b10d465d939be42dd0'
ROOT = Path('/home/sasaki/datasets/openloris')


def safe_path(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or len(path.parts) < 2:
        raise ValueError('Unsafe path')
    if not path.parts[0].startswith('corridor1-1-m8-'):
        raise ValueError('Outside M8')
    if path.name not in ('images.txt', 'points3D.txt') and path.suffix != '.vps':
        raise ValueError('Unsupported artifact')
    return ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.audit.exists():
        raise FileExistsError(args.audit)
    inventory_path = ROOT / 'm8-model-match-duplicate-inventory-v1.json'
    if sha(inventory_path) != INVENTORY_SHA:
        raise ValueError('Inventory changed')
    inventory = json.loads(inventory_path.read_text())
    operations = []
    for group in inventory['groups']:
        digest = group['sha256']
        source = safe_path(group['files'][0]['path'])
        for row in group['files']:
            path = safe_path(row['path'])
            current = checked_stat(path, digest)
            if (current.st_dev, current.st_ino, current.st_nlink) != (
                    row['device'], row['inode'], row['links']):
                raise ValueError('Inventory identity changed: ' + str(path))
            if path != source and row['links'] == 1:
                operations.append(dict(source=str(source), target=str(path), sha256=digest,
                                       inode=current.st_ino, device=current.st_dev))
    audit = dict(status='validated', inventory_sha256=INVENTORY_SHA, applied=args.apply,
                 operations=operations, processed=0, freed_allocated_bytes=0,
                 caveat='All participating paths are frozen shared-inode artifacts; never overwrite.')
    atomic_json(args.audit, audit)
    if args.apply:
        try:
            for row in operations:
                target = Path(row['target'])
                current = target.stat()
                if (current.st_dev, current.st_ino, current.st_nlink) != (
                        row['device'], row['inode'], 1):
                    raise ValueError('Target identity changed')
                audit['freed_allocated_bytes'] += merge_file(Path(row['source']), target, row['sha256'])
                audit['processed'] += 1
                atomic_json(args.audit, audit)
            for row in operations:
                checked_stat(Path(row['target']), row['sha256'])
                if not Path(row['source']).samefile(row['target']):
                    raise ValueError('Postflight inode mismatch')
            audit['status'] = 'completed'
        except BaseException as error:
            audit.update(status='failed-partial-preserved', error=str(error))
            raise
        finally:
            atomic_json(args.audit, audit)
    print({k: v for k, v in audit.items() if k != 'operations'})


if __name__ == '__main__':
    main()
