#!/usr/bin/env python3
"""Replay prefix registration or repair-prefix mapping for downstream selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_electro import parse_gnu_time
from replay_native_candidates import sha, bind_feature_input
from select_rig_sift_supplements import parse_frames


def mapper_command(stage, binary, rig, features, output, snapshot):
    if stage not in ('prefix', 'targeted'):
        raise ValueError('Unknown registration stage')
    command = [str(binary), '--manifest', str(rig), '--features-dir', str(features),
               '--snapshot', str(snapshot), '--out-colmap', str(output / 'model'),
               '--max-models', '10', '--pair-confidence-tracks',
               '--final-ba-min-pose-observations', '32']
    if stage == 'targeted':
        command += ['--deferred-registration-pair-prefix', '59961']
    return command, 1 if stage == 'prefix' else 8


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
    parser.add_argument('--stage', choices=['targeted', 'prefix'], default='targeted')
    parser.add_argument('--features-dir', type=Path,
                        default=base / 'corridor1-1-m8-adaptive-bank-publication-v1/features')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rig = base / 'corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt'
    features = args.features_dir.resolve(strict=True)
    prefix = args.stage == 'prefix'
    default_snapshot = ('corridor1-1-m8-native-prefix-supplement-replay-v1/output.vps' if prefix
                        else 'corridor1-1-m8-repair19-admission-replay-v1/output.vps')
    snapshot = (args.snapshot or base / default_snapshot).resolve(strict=True)
    binary = base / 'corridor1-1-m8-retry-reuse-1k-v1/rig-0466499'
    if sha(binary) != '3d744ced8b2b09cbaba26bf963e9ac31c5293538b1a8bbbee62621ccfdd70c34':
        raise RuntimeError('Saved mapper binary changed')
    expected_snapshot = ('fbcd111cd193fd59a3311c99208de6fe30c2c334f0b088ba7798aeff69dffe6f' if prefix
                         else 'ac93b28daf92b9bb4a621b9687087abbf906a1bc78cca0d8f4ea9688a1ba7a8d')
    if sha(snapshot) != expected_snapshot:
        raise RuntimeError('Reproduced input snapshot changed')
    _, features = bind_feature_input(['mapper', '--features-dir', str(features)],
        base / 'corridor1-1-m8-adaptive32-halo8-10k-v1/pipeline/features.json')
    frames = parse_frames(rig.read_bytes())
    reference = base / 'corridor1-1-m8-visloc-rig/tier-10000-deferred-repair19-pair-confidence-finalfix32-v1/model'
    if prefix:
        reference = base / 'corridor1-1-m8-repair-registration-replay-v1/model'
    default_output = ('corridor1-1-m8-prefix-registration-bound-v1' if prefix
                      else 'corridor1-1-m8-targeted-selection-replay-v1')
    output = (args.output or base / default_output).resolve()
    output.mkdir()
    command, threads = mapper_command(args.stage, binary, rig, features, output, snapshot)
    invocation = ['/usr/bin/time', '-v', '-o', str(output / 'time.txt'),
                  'timeout', '--foreground', '--signal=TERM', '--kill-after=10s', '600s', *command]
    report = {'status': 'running', 'stage': args.stage, 'command': invocation,
              'binary_sha256': sha(binary), 'snapshot_sha256': sha(snapshot),
              'rig_manifest_sha256': sha(rig), 'features_resolved': str(features.resolve()),
              'rayon_num_threads': threads, 'malloc_arena_max': '1'}
    if not prefix:
        report['recovered_command_event_utc'] = '2026-09-02T23:34:18.633Z'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    with (output / 'run.log').open('w') as log:
        result = subprocess.run(invocation, env=dict(os.environ, RAYON_NUM_THREADS=str(threads), MALLOC_ARENA_MAX='1'),
                                stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode, wall_seconds=time.monotonic() - started,
                  measurement=parse_gnu_time(output / 'time.txt'))
    report['status'] = 'fail'
    if result.returncode == 0:
        model = output / 'model'
        targets = unregistered_frames(frames, (model / 'retrieval-components.txt').read_text())
        report['target_frames'] = targets
        expected_targets = (unregistered_frames(frames, (reference / 'retrieval-components.txt').read_text())
                            if prefix else [1999, 3264, 3266, 3267, 4493, 4494, 4495])
        report['target_frames_match'] = targets == expected_targets
        actual = {p.relative_to(model).as_posix(): sha(p) for p in model.rglob('*') if p.is_file()}
        expected = {p.relative_to(reference).as_posix(): sha(p) for p in reference.rglob('*') if p.is_file()}
        report['all_model_files_identical'] = actual == expected
        report['different_model_files'] = sorted(name for name in actual.keys() | expected.keys()
                                                 if actual.get(name) != expected.get(name))
        if report['target_frames_match'] and (not prefix or actual == expected):
            report['status'] = 'complete-selection-reproduction-pass'
            (output / 'selection.json').write_text(json.dumps({'target_frames': targets}, indent=2) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'complete-selection-reproduction-pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
