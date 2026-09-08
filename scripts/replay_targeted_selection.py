#!/usr/bin/env python3
"""Replay the recovered repair-prefix mapper and derive unregistered targets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_electro import parse_gnu_time
from replay_native_candidates import sha, bind_feature_input
from select_rig_sift_supplements import parse_frames


def unregistered_frames(frames, components):
    known = {name for members in frames.values() for name in members}
    registered = set()
    if '# retrieval-component-manifest-v1' not in components.splitlines():
        raise ValueError('Missing component manifest schema')
    for line in components.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        if len(fields) != 3 or fields[0] != 'C' or int(fields[1]) < 0:
            raise ValueError('Malformed component row')
        name = fields[2]
        if name not in known or name in registered:
            raise ValueError('Unknown or repeated registered image')
        registered.add(name)
    missing = []
    for frame, members in sorted(frames.items()):
        count = sum(name in registered for name in members)
        if count not in (0, len(members)):
            raise ValueError('Partially registered rig frame is ambiguous')
        if count == 0:
            missing.append(frame)
    return missing


def main():
    base = Path('/home/sasaki/datasets/openloris')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path,
                        default=base / 'corridor1-1-m8-adaptive-bank-publication-v1/features')
    parser.add_argument('--snapshot', type=Path,
                        default=base / 'corridor1-1-m8-repair19-admission-replay-v1/output.vps')
    parser.add_argument('--output', type=Path,
                        default=base / 'corridor1-1-m8-targeted-selection-replay-v1')
    args = parser.parse_args()
    rig = base / 'corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt'
    features = args.features_dir.resolve(strict=True)
    snapshot = args.snapshot.resolve(strict=True)
    binary = base / 'corridor1-1-m8-retry-reuse-1k-v1/rig-0466499'
    if sha(binary) != '3d744ced8b2b09cbaba26bf963e9ac31c5293538b1a8bbbee62621ccfdd70c34':
        raise RuntimeError('Saved mapper binary changed')
    if sha(snapshot) != 'ac93b28daf92b9bb4a621b9687087abbf906a1bc78cca0d8f4ea9688a1ba7a8d':
        raise RuntimeError('Reproduced repair19 snapshot changed')
    _, features = bind_feature_input(['mapper', '--features-dir', str(features)],
        base / 'corridor1-1-m8-adaptive32-halo8-10k-v1/pipeline/features.json')
    frames = parse_frames(rig.read_bytes())
    reference = base / 'corridor1-1-m8-visloc-rig/tier-10000-deferred-repair19-pair-confidence-finalfix32-v1/model'
    output = args.output.resolve()
    output.mkdir()
    command = [str(binary), '--manifest', str(rig), '--features-dir', str(features),
               '--snapshot', str(snapshot), '--out-colmap', str(output / 'model'),
               '--max-models', '10', '--pair-confidence-tracks',
               '--final-ba-min-pose-observations', '32',
               '--deferred-registration-pair-prefix', '59961']
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--signal=TERM', '--kill-after=10s', '600s', *command]
    report = {'status': 'running', 'command': invocation,
              'binary_sha256': sha(binary), 'snapshot_sha256': sha(snapshot),
              'rig_manifest_sha256': sha(rig), 'features_resolved': str(features.resolve()),
              'rayon_num_threads': 8, 'malloc_arena_max': '1',
              'recovered_command_event_utc': '2026-09-02T23:34:18.633Z'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, env=dict(os.environ, RAYON_NUM_THREADS='8', MALLOC_ARENA_MAX='1'),
                                stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode, wall_seconds=time.monotonic() - started,
                  measurement=parse_gnu_time(output / 'time.txt'))
    report['status'] = 'fail'
    if result.returncode == 0:
        model = output / 'model'
        targets = unregistered_frames(frames, (model / 'retrieval-components.txt').read_text())
        report['target_frames'] = targets
        report['target_frames_match'] = targets == [1999, 3264, 3266, 3267, 4493, 4494, 4495]
        actual = {p.relative_to(model).as_posix(): sha(p) for p in model.rglob('*') if p.is_file()}
        expected = {p.relative_to(reference).as_posix(): sha(p) for p in reference.rglob('*') if p.is_file()}
        report['all_model_files_identical'] = actual == expected
        report['different_model_files'] = sorted(name for name in actual.keys() | expected.keys()
                                                 if actual.get(name) != expected.get(name))
        if report['target_frames_match']:
            report['status'] = 'complete-selection-reproduction-pass'
            (output / 'selection.json').write_text(json.dumps({'target_frames': targets}, indent=2) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'complete-selection-reproduction-pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
