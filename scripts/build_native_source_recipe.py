#!/usr/bin/env python3
"""Compile reproduced source-mapper evidence into argv templates; never execute it."""
import hashlib
import json
from pathlib import Path
import shlex


INPUTS = {
    '--manifest': '{rig_manifest}',
    '--features-dir': '{adaptive_features}',
    '--append-features-dir': '{dense_features}',
    '--snapshot': '{targeted_admission}',
    '--deferred-overlay-snapshot': '{dense_snapshot}',
}


def render_nodes_tsv(recipe, run_root):
    """Bind atlas nodes to this run's source outputs, never retained models."""
    root = Path(run_root).resolve()
    executions = {row['id'] for row in recipe['executions']}
    lines = ['# node_id\twindow_start\timages_txt']
    seen = set()
    for row in recipe['nodes']:
        if row['node'] in seen or row['execution'] not in executions:
            raise ValueError('Duplicate node or unknown source execution')
        seen.add(row['node'])
        for value in (row['execution'], row['component']):
            if not value or Path(value).name != value or value in ('.', '..'):
                raise ValueError('Invalid source path component')
        path = root / 'sources' / row['execution'] / 'model' / row['component'] / 'images.txt'
        if any(char in str(path) for char in '\t\r\n'):
            raise ValueError('Path cannot be represented in TSV')
        lines.append(f"{row['node']}\t{row['offset']}\t{path}")
    return '\n'.join(lines) + '\n'


def compile_execution(path):
    raw = path.read_bytes()
    evidence = json.loads(raw)
    if 'RAYON_NUM_THREADS=1' not in shlex.split(evidence['command']):
        raise ValueError('Missing recorded single-thread environment')
    line = evidence['audit']['time'].splitlines()[0]
    command = shlex.split(shlex.split(line.split('Command being timed: ', 1)[1])[0])
    if command[:4] != ['timeout', '--signal=TERM', '--kill-after=10s', '180s']:
        raise ValueError('Unexpected timeout wrapper')
    command = command[4:]
    environment = {'RAYON_NUM_THREADS': '1'}
    if command[:2] == ['env', 'VISLOC_DEFERRED_DEBUG=1']:
        environment['VISLOC_DEFERRED_DEBUG'] = '1'
        command = command[2:]
    if Path(command[0]).name != 'rig-0466499':
        raise ValueError('Unexpected mapper executable')
    command[0] = '{mapper_binary}'
    identity = path.stem.removeprefix('m8-openloris-').removesuffix('-v1')
    substitutions = dict(INPUTS, **{'--out-colmap': '{run_root}/sources/' + identity + '/model'})
    for flag, value in substitutions.items():
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError(f'Missing/duplicate path flag: {flag}')
        command[command.index(flag) + 1] = value
    if any(value.startswith('/') for value in command):
        raise ValueError('Unbound absolute path remains')
    expected = evidence['audit']['new_hashes']
    if not expected or not evidence['audit']['exact']:
        raise ValueError('Source lacks exact reproduction evidence')
    return {'id': identity, 'argv': command, 'environment': environment,
            'unset_environment': ['VISLOC_SFM_DEBUG', 'VISLOC_SFM_DEBUG_BA',
                                  'VISLOC_BA_TRACE_PHASE_TIMING', 'VISLOC_DEFERRED_DEBUG'],
            'environment_order': 'unset listed variables, then apply environment',
            'timeout_seconds': 180, 'binary_sha256': evidence['binary_sha256'],
            'source_commit': evidence['build_commit'],
            'evidence': path.name, 'evidence_sha256': hashlib.sha256(raw).hexdigest(),
            'expected_model_hashes': expected}


def build(evidence_root):
    paths = sorted([*evidence_root.glob('m8-openloris-source-spec-*-v1.json'),
                    *evidence_root.glob('m8-openloris-source-replay-*-v1.json')])
    if len(paths) != 21:
        raise ValueError('Expected exactly 21 source executions')
    executions = [compile_execution(path) for path in paths]
    binding_bytes = (evidence_root / 'm8-openloris-regenerated-nodes-v1.json').read_bytes()
    bindings = json.loads(binding_bytes)['audit']['rows']
    by_evidence = {row['evidence']: row for row in executions}
    nodes = []
    for row in bindings:
        execution = by_evidence[Path(row['evidence']).name]
        component = Path(row['images']).parent.name
        for name, digest in row['hashes'].items():
            if execution['expected_model_hashes'].get(component + '/' + name) != digest:
                raise ValueError('Node/source hash mismatch')
        nodes.append({'node': row['node'], 'offset': row['offset'],
                      'execution': execution['id'], 'component': component})
    if len(nodes) != 23 or len({row['node'] for row in nodes}) != 23:
        raise ValueError('Expected 23 unique atlas nodes')
    return {'schema': 'native-source-recipe-v1', 'executions': executions, 'nodes': nodes,
            'node_evidence_sha256': hashlib.sha256(binding_bytes).hexdigest(),
            'scope': 'Non-executing source-mapping templates only. Bind and validate newly generated inputs, binary hash, environment and resource limits before execution. Not a complete E2E DAG or a passing quality result.'}


if __name__ == '__main__':
    print(json.dumps(build(Path(__file__).resolve().parents[1] / 'benchmarks/electro'), indent=2))
