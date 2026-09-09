"""Bind the frozen stitch/integration recipe to a new pipeline's artifacts."""
import json
from pathlib import Path
import shlex


def timed_argv(audit, seconds):
    line = audit['time'].splitlines()[0].split('Command being timed: ', 1)[1]
    command = shlex.split(shlex.split(line)[0])
    if command[:4] != ['timeout', '--signal=TERM', '--kill-after=10s', f'{seconds}s']:
        raise ValueError('Unexpected atlas timeout wrapper')
    return command[4:]


def bind(command, replacements):
    for flag, value in replacements.items():
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError(f'Missing or duplicate atlas flag: {flag}')
        command[command.index(flag) + 1] = value
    if any(arg.startswith('/') for arg in command):
        raise ValueError('Unbound atlas path')
    return command


def build(evidence_root):
    root = Path(evidence_root)
    stitch = json.loads((root / 'm8-openloris-regenerated-atlas-v1.json').read_text())
    integration = json.loads((root / 'm8-openloris-regenerated-integration-v1.json').read_text())
    command = timed_argv(stitch['audit'], 180)
    command[0] = '{stitch_binary}'
    common = {'--rig-manifest': '{rig_manifest}', '--nodes-tsv': '{nodes_tsv}'}
    stages = [{'id': 'stitch', 'binary_sha256': stitch['binary_sha256'],
               'argv': bind(command, dict(common, **{'--out-dir': '{run_root}/atlas'})),
               'environment': {'RAYON_NUM_THREADS': '1'}, 'timeout_seconds': 180}]
    for name, component in [('tail', 'component-001'), ('main', 'component-000')]:
        audit = integration['audits'][name]
        required = {phase + '/' + filename for phase in ('model', 'pre-ba')
                    for filename in ('cameras.txt', 'images.txt', 'points3D.txt')}
        if (set(audit['files']) != required or not audit['all_exact'] or
                not all(row['exact'] and row['new'] == row['old'] for row in audit['files'].values())):
            raise ValueError('Integration lacks exact reference evidence')
        command = timed_argv(audit, 360)
        command[0] = '{integration_binary}'
        replacements = dict(common, **{'--atlas-dir': '{run_root}/atlas/' + component,
                                      '--out-dir': '{run_root}/integrated/' + name + '/model',
                                      '--pre-ba-out-dir': '{run_root}/integrated/' + name + '/pre-ba'})
        stages.append({'id': 'integrate-' + name, 'argv': bind(command, replacements),
                       'binary_sha256': integration['binary_sha256'], 'timeout_seconds': 360,
                       'environment': {'RAYON_NUM_THREADS': '1', 'VISLOC_BA_TRACE_PHASE_TIMING': '1',
                                       'VISLOC_ATLAS_TRACE_WINDOW_TIMING': '1'},
                       'expected_files': {key: row['new'] for key, row in audit['files'].items()}})
    return stages
