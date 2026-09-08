#!/usr/bin/env python3
"""Compile three reproduced admission stages without executing retained paths."""
import hashlib
import json
from pathlib import Path
import shlex

from replay_native_candidates import replace_path


STAGES = [
    ('native-prefix-supplement', {
        '--base-snapshot': '{native_snapshot}', '--augmented-snapshot': '{adaptive_snapshot}',
        '--base-features-dir': '{base_features}'}),
    ('repair19-admission', {
        '--base-snapshot': '{prefix_snapshot}', '--augmented-snapshot': '{adaptive_snapshot}',
        '--base-features-dir': '{adaptive_features}',
        '--repair-registered-components': '{prefix_registration_components}'}),
    ('targeted7-admission', {
        '--base-snapshot': '{repair_snapshot}', '--augmented-snapshot': '{targeted_snapshot}',
        '--base-features-dir': '{adaptive_features}'}),
]


def build(root):
    stages = []
    for name, inputs in STAGES:
        path = root / f'm8-openloris-{name}-replay-v1.json'
        raw = path.read_bytes()
        evidence = json.loads(raw)
        if evidence['exit_status'] != 0:
            raise ValueError('Admission evidence did not exit zero')
        argv = shlex.split(evidence['command'])
        if Path(argv[0]).name != 'admit_verified_bridge_snapshot':
            raise ValueError('Unexpected admission executable')
        argv[0] = '{admission_binary}'
        inputs = dict(inputs, **{'--rig-manifest': '{rig_manifest}'})
        for flag, value in inputs.items():
            argv = replace_path(argv, flag, value)
        argv = replace_path(argv, '--output', '{run_root}/admissions/' + name + '.vps')
        if any(value.startswith('/') for value in argv):
            raise ValueError('Unbound admission input remains')
        stages.append({'id': name, 'argv': argv, 'input_bindings': inputs,
                       'binary_sha256': evidence['binary_sha256'],
                       'evidence_sha256': hashlib.sha256(raw).hexdigest(),
                       'expected_snapshot_sha256': evidence['output_and_reference_sha256']})
    return {'schema': 'native-admission-recipe-v1', 'stages': stages,
            'scope': 'Non-executing admission templates only. Registration components must be produced by the intermediate mapper, not supplied from retained runs. Does not implement selection, extraction, execution, resume or E2E.'}


if __name__ == '__main__':
    print(json.dumps(build(Path(__file__).resolve().parents[1] / 'benchmarks/electro'), indent=2))
