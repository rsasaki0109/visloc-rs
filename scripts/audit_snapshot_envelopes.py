#!/usr/bin/env python3
"""Measure repeated v1 snapshot envelopes, without claiming payload validation."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

MAGIC = b'VISLOC-VERIFIED-PAIR-SNAPSHOT\0'


def inspect(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        def read(count):
            if count < 0 or count > size - stream.tell():
                raise ValueError('Truncated snapshot')
            value = stream.read(count)
            if len(value) != count:
                raise ValueError('Truncated snapshot')
            return value

        def integer(fmt):
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

        if read(len(MAGIC)) != MAGIC or integer('<I') != 1:
            raise ValueError('Unsupported snapshot header')
        payload_size = integer('<Q')
        start = stream.tell()
        if start + payload_size + 8 != size:
            raise ValueError('Snapshot length mismatch')
        if integer('<I') != 1:
            raise ValueError('Unsupported payload schema')
        count = integer('<Q')
        if count > payload_size // 8:
            raise ValueError('Invalid image count')
        name_bytes = 0
        for _ in range(count):
            length = integer('<Q')
            read(length)
            name_bytes += 8 + length
        read(16)  # image and feature manifest hashes
        features = integer('<Q')
        if features != count:
            raise ValueError('Image/feature count mismatch')
        read(8 * features)
        read(48)  # width, height, intrinsics
        for _ in range(2):
            read(8)  # configuration hash
            read(integer('<Q'))
        end = stream.tell()
        read(24)  # shard-specific pair/edge hashes and match count
        pairs = integer('<Q')
        if stream.tell() > start + payload_size:
            raise ValueError('Envelope exceeds payload')
        stream.seek(start)
        digest = hashlib.sha256(read(end - start)).hexdigest()
    return {'images': count, 'pairs': pairs, 'file_bytes': size,
            'shared_envelope_bytes': end - start,
            'image_dependent_bytes': name_bytes + 8 * features,
            'envelope_sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.snapshots.glob('*.vps'))
    if not paths:
        parser.error('No snapshot files')
    rows = [inspect(path) for path in paths]
    unique = {}
    for row in rows:
        unique[row['envelope_sha256']] = row['shared_envelope_bytes']
    total = sum(row['shared_envelope_bytes'] for row in rows)
    print(json.dumps({
        'scope': 'Structural envelope size audit only; pair payload/checksum not validated.',
        'snapshots': str(args.snapshots.resolve()), 'shards': len(rows),
        'image_counts': sorted({row['images'] for row in rows}),
        'pairs': sum(row['pairs'] for row in rows),
        'file_bytes': sum(row['file_bytes'] for row in rows),
        'shared_envelope_bytes_total': total,
        'unique_envelopes': unique,
        'duplicate_envelope_bytes': total - sum(unique.values()),
        'image_dependent_bytes_total': sum(row['image_dependent_bytes'] for row in rows),
        'scaling': 'With O(N) fixed-pair-count shards and full N-image envelopes, serialized image metadata is O(N^2). A streaming merger alone does not change this.'
    }, indent=2))


if __name__ == '__main__':
    main()
