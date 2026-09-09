#!/usr/bin/env python3
"""Extraction-through-atlas diagnostic executor; no restart or quality promotion.

Run inside launch_native_measurement.py. All generated dependencies stay under
one fresh root. Retained artifacts are comparison or calibration inputs only.
The conservative 16 GiB free-space guard is not a measured lifetime bound.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from replay_native_candidates import sha


def plan(root, binaries, dataset):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    scripts = Path(__file__).resolve().parent
    rig = dataset / 'corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt'
    stages = []

    def add(name, script, arguments, payload=None, capture=None):
        stages.append({'id': name, 'argv': [sys.executable, str(scripts / script),
                                            *map(str, arguments)],
                       'payload': payload, 'capture': str(capture) if capture else None})

    for variant in ('base', 'dense'):
        add('extract-' + variant, 'probe_openloris_base_extraction.py',
            ['--variant', variant, '--all-images', '--workers', 6, '--output', root / variant])
    base, dense, adaptive = root / 'base/features', root / 'dense/features', root / 'adaptive'
    selection = root / 'adaptive-selection.json'
    add('adaptive-selection', 'select_rig_sift_supplements.py',
        ['--rig-manifest', rig, '--features-dir', base, '--min-sensor-rows-lt', 32, '--frame-halo', 8],
        capture=selection)
    add('adaptive-bank', 'merge_sift_supplements.py',
        ['--base', base, '--supplement', dense, '--selection', selection,
         '--output', adaptive, '--hardlink-unselected'])
    for variant, features in [('native', base), ('dense', dense)]:
        candidate = 'candidate_' + variant
        add('candidates-' + variant, 'replay_native_candidates.py',
            ['--variant', variant, '--binary', binaries[candidate],
             '--binary-sha256', binaries[candidate + '_sha256'],
             '--source-commit', 'a7ff5ff47b561556ba20a0afdb40417ec9413120' if variant == 'native'
             else '3ae253a0af909042fe0cfbdb08e8d50733b79375',
             '--features-dir', features, '--output', root / ('candidates-' + variant)])

    def matching(variant, features, candidates):
        add('match-' + variant, 'replay_full_shared_runner.py',
            ['--variant', variant, '--binary', binaries['sfm'], '--merge-binary', binaries['merge'],
             '--compare-binary', binaries['compare'], '--features-dir', features,
             '--candidate-manifest', candidates, '--output', root / ('match-' + variant)])

    def snapshot(variant):
        return str(root / ('match-' + variant) / 'run/mapping/verified-merged.vps')

    for variant, features in [('native', base), ('adaptive', adaptive), ('dense', dense)]:
        matching(variant, features, root / ('candidates-dense' if variant == 'dense' else 'candidates-native') / 'candidates.txt')

    def admission(name, bindings):
        bindings = {key: str(value) for key, value in dict(bindings, rig_manifest=rig).items()}
        path = root / (name + '-inputs.json')
        add(name, 'run_native_admission.py',
            ['--stage', name, '--bindings', path, '--binary', binaries['admission'],
             '--output', root / name], payload={'path': str(path), 'data': bindings})
        return root / name / 'admissions' / (name + '.vps')

    prefix = admission('native-prefix-supplement', dict(native_snapshot=snapshot('native'),
                       adaptive_snapshot=snapshot('adaptive'), base_features=base))
    add('prefix-registration', 'replay_targeted_selection.py',
        ['--stage', 'prefix', '--features-dir', adaptive, '--snapshot', prefix,
         '--output', root / 'prefix-registration'])
    repair = admission('repair19-admission', dict(prefix_snapshot=prefix,
                       adaptive_snapshot=snapshot('adaptive'), adaptive_features=adaptive,
                       prefix_registration_components=root / 'prefix-registration/model/retrieval-components.txt'))
    add('target-selection', 'replay_targeted_selection.py',
        ['--stage', 'targeted', '--features-dir', adaptive, '--snapshot', repair,
         '--output', root / 'target-selection'])
    add('target-candidates', 'build_targeted_rig_candidates.py',
        ['--rig-manifest', rig, '--selection-json', root / 'target-selection/selection.json',
         '--max-frame-gap', 256, '--output-directory', root / 'target-candidates'])
    matching('targeted7', adaptive, root / 'target-candidates/candidates.txt')
    final = admission('targeted7-admission', dict(repair_snapshot=repair,
                      targeted_snapshot=snapshot('targeted7'), adaptive_features=adaptive))
    # The mapping CLI validates these records against the supplied pinned spec.
    spec_path = Path(__file__).resolve().parents[1] / 'benchmarks/electro/m8-native-mapping-bound-inputs-v1.json'
    spec = json.loads(spec_path.read_text())
    for key, value in dict(adaptive_features=adaptive, dense_features=dense,
                           targeted_admission=final, dense_snapshot=snapshot('dense')).items():
        spec[key]['path'] = str(value)
    path = root / 'mapping-inputs.json'
    add('mapping', 'run_native_mapping.py',
        ['--inputs', path, '--mapper-binary', binaries['mapper'], '--stitch-binary', binaries['stitch'],
         '--integration-binary', binaries['integration'], '--output', root / 'mapping'],
        payload={'path': str(path), 'data': spec})
    return stages


def pinned_files(repository, binary_spec):
    """Record code/evidence/binary identities; this is not filesystem isolation."""
    repository = Path(repository)
    paths = set((repository / 'scripts').glob('*.py'))
    paths.update((repository / 'benchmarks/electro').glob('*.json'))
    paths.update((repository / 'benchmarks/electro').glob('*.tsv'))
    paths.update(Path(row['path']) for row in binary_spec.values())
    return {str(path.resolve(strict=True)): sha(path) for path in sorted(paths)}


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError('Pipeline dependency changed: ' + name)


def execute(stages, root, pins=None):
    pins = pins or {}
    verify_pins(pins)
    root = Path(root).resolve()
    if shutil.disk_usage(root.parent).free < 16 * 1024**3:
        raise RuntimeError('Require 16 GiB free for retained-stage diagnostic pipeline; no automatic cleanup')
    root.mkdir()  # No implicit reuse of interrupted or completed outputs.
    report = {'status': 'running', 'stages': [], 'plan': stages, 'dependency_sha256': pins,
              'scope': 'Extraction-through-atlas diagnostic including reference validation; no restart or quality promotion.'}
    def save():
        (root / 'pipeline-report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    started = time.monotonic()
    try:
        for stage in stages:
            verify_pins(pins)
            if stage['payload']:
                with Path(stage['payload']['path']).open('x') as stream:
                    json.dump(stage['payload']['data'], stream, indent=2)
            log = root / (stage['id'] + '.log')
            begin = time.monotonic()
            with log.open('x') as stream:
                result = subprocess.run(stage['argv'], stdout=stream, stderr=subprocess.STDOUT)
            report['stages'].append({'id': stage['id'], 'exit_code': result.returncode,
                                     'wall_seconds': time.monotonic() - begin, 'log_sha256': sha(log)})
            save()
            verify_pins(pins)
            if result.returncode:
                raise RuntimeError('Pipeline stage failed: ' + stage['id'])
            if stage['capture']:
                data = json.loads(log.read_text())
                with Path(stage['capture']).open('x') as stream:
                    json.dump(data, stream, indent=2)
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='fail', error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, required=True,
                        help='JSON name -> {path, sha256} for candidate_native/candidate_dense/sfm/merge/compare/admission/mapper/stitch/integration')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.binaries.read_text())
    required = {'candidate_native', 'candidate_dense', 'sfm', 'merge', 'compare', 'admission', 'mapper', 'stitch', 'integration'}
    if set(spec) != required:
        raise ValueError('Require exactly all pipeline binaries')
    binaries = {}
    for key, row in spec.items():
        path = Path(row['path']).resolve(strict=True)
        if sha(path) != row['sha256']:
            raise ValueError('Binary hash mismatch: ' + key)
        binaries[key] = str(path)
    for variant, expected in [('native', '8eeee5c2f8b39de6575c2308e89cba11b96d38d66d1ae3d51ab6ce0d61d43e79'),
                              ('dense', '8cfa9c53fcaea5d6305dbdd3018b8381ae214806751e6ee3dc85bb8692a8da34')]:
        key = 'candidate_' + variant
        if spec[key]['sha256'] != expected:
            raise ValueError('Candidate binary differs from reproduced recipe: ' + variant)
        binaries[key + '_sha256'] = expected
    stages = plan(args.output, binaries, Path('/home/sasaki/datasets/openloris'))
    if args.plan_only:
        print(json.dumps(stages, indent=2))
    else:
        pins = pinned_files(Path(__file__).resolve().parents[1], spec)
        print(json.dumps(execute(stages, args.output, pins), indent=2))


if __name__ == '__main__':
    main()
