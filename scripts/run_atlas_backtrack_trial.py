#!/usr/bin/env python3
"""Experimental ON integration suffix; completion is not a quality PASS.

Reuses hash-verified OFF stitch outputs. Compare integration stage times only,
not this suffix's total wall time against a control including stitch.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from benchmark_electro import atomic_json
from native_atlas_recipe import build
from replay_native_candidates import sha
from run_native_pipeline import require_memory_scope


def validate_outputs(output, expected):
    required = {f'{phase}/{name}.txt' for phase in ('pre-ba', 'model')
                for name in ('cameras', 'images', 'points3D')}
    if set(expected) != required:
        raise ValueError('Require all six integration reference files')
    actual = {}
    for name in sorted(required):
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing, empty or linked output: ' + name)
        actual[name] = sha(path)
        if (name.startswith('pre-ba/') or name == 'model/cameras.txt') and actual[name] != expected[name]:
            raise ValueError('Pre-BA input or fixed camera changed: ' + name)
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_memory_scope()
    root = args.output.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if shutil.disk_usage(root.parent).free < 500 * 1024**2:
        raise RuntimeError('Require 500 MiB free for one bounded integration trial')
    evidence = Path(__file__).resolve().parents[1] / 'benchmarks/electro'
    control_path = evidence / 'm8-feasible-backtrack-off-v1.json'
    control = json.loads(control_path.read_text())
    if control['status'] != 'off-reference-parity-pass':
        raise ValueError('Require a completed exact OFF control')
    mapping_path = evidence / 'm8-native-mapping-bound-v2.json'
    mapping = json.loads(mapping_path.read_text())
    input_path = evidence / 'm8-native-mapping-bound-inputs-v1.json'
    inputs = json.loads(input_path.read_text())
    dataset = Path('/home/sasaki/datasets/openloris')
    binary = dataset / 'schur-probe-binaries.4WAfOy/integrate-37298b3'
    source = Path(mapping['run_root'])
    control_root = Path(control['run_root'])
    rig = dataset / 'corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt'
    pins = {str(binary): control['binary_sha256'],
            str(source / 'nodes.tsv'): mapping['nodes_sha256'],
            str(rig): inputs['rig_manifest']['sha256']}
    pins.update({str(source / name): digest for name, digest in mapping['reference_file_hashes'].items()
                 if name.startswith('sources/')})
    pins.update({str(control_root / name): digest
                 for name, digest in control['reference_file_sha256'].items()})
    for path in (control_path, mapping_path, input_path, Path(__file__),
                 evidence / 'm8-openloris-regenerated-atlas-v1.json',
                 evidence / 'm8-openloris-regenerated-integration-v1.json',
                 Path(__file__).with_name('native_atlas_recipe.py')):
        pins[str(path.resolve())] = sha(path)

    def verify():
        for name, digest in pins.items():
            if sha(Path(name)) != digest:
                raise ValueError('Pinned input changed: ' + name)

    verify()
    stages = build(evidence)[1:]
    root.mkdir()
    report = {'status': 'running', 'quality_status': 'not-evaluated',
              'scope': 'Integration suffix only; retained OFF stitch and sources; not cold E2E',
              'pinned_files': pins, 'stages': []}
    report_path = root / 'trial-report.json'
    atomic_json(report_path, report)
    try:
        for stage, component in zip(stages, ('component-001', 'component-000')):
            name = stage['id'].removeprefix('integrate-')
            output = root / 'integrated' / name
            output.mkdir(parents=True)
            values = dict(integration_binary=str(binary), rig_manifest=str(rig),
                          nodes_tsv=str(source / 'nodes.tsv'), run_root=str(root))
            command = [arg.format_map(values) for arg in stage['argv']]
            command[command.index('--atlas-dir') + 1] = str(control_root / 'atlas' / component)
            env = {k: v for k, v in os.environ.items() if not k.startswith('VISLOC_')}
            env.update(stage['environment'], MALLOC_ARENA_MAX='1', VISLOC_SFM_BA_FEASIBLE_BACKTRACK='1')
            row = {'id': stage['id'], 'status': 'running', 'command': command,
                   'environment_overrides': dict(stage['environment'], MALLOC_ARENA_MAX='1',
                                                 VISLOC_SFM_BA_FEASIBLE_BACKTRACK='1')}
            report['stages'].append(row)
            atomic_json(report_path, report)
            verify()
            started = time.monotonic()
            with (output / 'execution.log').open('x') as log:
                child = subprocess.run(['timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                                        str(stage['timeout_seconds']) + 's', *command],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            row.update(exit_code=child.returncode, wall_seconds=time.monotonic() - started)
            if child.returncode:
                raise RuntimeError('Integration failed: ' + name)
            verify()
            row['file_hashes'] = validate_outputs(output, stage['expected_files'])
            row['status'] = 'completed-not-quality-evaluated'
            atomic_json(report_path, report)
        report['status'] = 'completed-not-quality-evaluated'
    except BaseException as error:
        report.update(status='fail', error=str(error))
        raise
    finally:
        atomic_json(report_path, report)


if __name__ == '__main__':
    main()
