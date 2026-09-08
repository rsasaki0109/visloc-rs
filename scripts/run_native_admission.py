#!/usr/bin/env python3
"""Execute one frozen admission recipe with explicit regenerated input bindings."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from build_native_admission_recipe import build
from replay_native_candidates import bind_feature_input, sha


def render(stage, bindings, binary, output):
    required = {value[1:-1] for value in stage['input_bindings'].values()}
    if set(bindings) != required:
        raise ValueError('Bindings must exactly match required stage inputs')
    values = {name: str(Path(path).resolve(strict=True)) for name, path in bindings.items()}
    values.update(admission_binary=str(binary), run_root=str(output))
    return [arg.format_map(values) for arg in stage['argv']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=['native-prefix-supplement', 'repair19-admission', 'targeted7-admission'])
    parser.add_argument('--bindings', type=Path, required=True,
                        help='JSON object mapping every recipe input placeholder to an existing path')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate inputs and print the bound command without creating output')
    args = parser.parse_args()
    evidence_root = Path(__file__).resolve().parents[1] / 'benchmarks/electro'
    stage = next(item for item in build(evidence_root)['stages'] if item['id'] == args.stage)
    binary = args.binary.resolve(strict=True)
    if sha(binary) != stage['binary_sha256']:
        raise ValueError('Admission binary differs from frozen recipe')
    bindings_raw = args.bindings.read_bytes()
    bindings = json.loads(bindings_raw)
    output = args.output.resolve()
    command = render(stage, bindings, binary, output)
    evidence = json.loads((evidence_root / f'm8-openloris-{args.stage}-replay-v1.json').read_text())
    reference_command = shlex.split(evidence['command'])
    inputs = {}
    base = Path('/home/sasaki/datasets/openloris')
    for flag, placeholder in stage['input_bindings'].items():
        path = Path(command[command.index(flag) + 1])
        if flag == '--base-features-dir':
            manifest = (base / 'corridor1-1-m8-native-rig-runner-10k-v1/features.json'
                        if args.stage == 'native-prefix-supplement' else
                        base / 'corridor1-1-m8-adaptive32-halo8-10k-v1/pipeline/features.json')
            bind_feature_input(['admit', '--features-dir', str(path)], manifest)
            inputs[placeholder] = {'path': str(path), 'feature_manifest_sha256': sha(manifest)}
        else:
            reference = Path(reference_command[reference_command.index(flag) + 1])
            digest = sha(path)
            if digest != sha(reference):
                raise ValueError(f'Input differs from frozen reference: {flag}')
            inputs[placeholder] = {'path': str(path), 'sha256': digest}
    report = {'status': 'running', 'stage': args.stage, 'command': command,
              'inputs': inputs, 'recipe': stage, 'scope': 'Single bound admission only; not native E2E.'}
    if output.exists():
        raise ValueError('Output must be a new path')
    if args.validate_only:
        report['status'] = 'input-validation-pass-not-executed'
        print(json.dumps(report, indent=2))
        return 0
    output.mkdir()
    (output / 'admissions').mkdir()
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    try:
        with (output / 'run.log').open('w') as log:
            result = subprocess.run(['timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                                     '600s', *command], stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, RAYON_NUM_THREADS='1', MALLOC_ARENA_MAX='1'))
        report['exit_code'] = result.returncode
        snapshot = output / 'admissions' / (args.stage + '.vps')
        report['snapshot_sha256'] = sha(snapshot) if snapshot.is_file() else None
        report['status'] = ('pass' if result.returncode == 0 and
                            report['snapshot_sha256'] == stage['expected_snapshot_sha256'] else 'fail')
    except BaseException as error:
        report.update(status='fail', error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
