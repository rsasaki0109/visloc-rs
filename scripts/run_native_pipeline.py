#!/usr/bin/env python3
"""Extraction-through-atlas diagnostic executor with reported-failure resume.

Run inside launch_native_measurement.py. All generated dependencies stay under
one fresh root. Retained artifacts are comparison or calibration inputs only.
The conservative 16 GiB free-space guard is not a measured lifetime bound.
Resume requires a normal failed child exit and unchanged completed artifacts;
unobserved interruption/SIGKILL recovery is not supported yet.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from replay_native_candidates import sha
from measure_native_scope import validate_limits
from benchmark_electro import atomic_json
from native_pipeline_checkpoint import stage_checkpoint
from native_pipeline_resume import validate_resume, quarantine_failed_stage


def require_memory_scope(proc_cgroup=Path('/proc/self/cgroup'), cgroup_root=Path('/sys/fs/cgroup')):
    entries = proc_cgroup.read_text().splitlines()
    if len(entries) != 1 or not entries[0].startswith('0::/'):
        raise ValueError('Require unified cgroup v2 measurement scope')
    relative = Path(entries[0][4:])
    if '..' in relative.parts or relative.is_absolute():
        raise ValueError('Invalid cgroup path')
    scope = cgroup_root / relative
    validate_limits(scope)
    return str(scope)


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


def artifact_lifetimes(stages):
    """Derive last path consumers for this serial plan; never delete artifacts.

    This describes namespace lifetime, not physical allocation: adaptive files
    can hardlink base inodes, so dropping base paths does not free shared bytes.
    Checkpoint/restart retention can require keeping artifacts beyond this point.
    """
    produced = {}
    for index, stage in enumerate(stages):
        argv = stage['argv']
        outputs = [argv[i + 1] for i, arg in enumerate(argv[:-1])
                   if arg in ('--output', '--output-directory')]
        if stage['capture']:
            outputs.append(stage['capture'])
        for value in outputs:
            path = Path(value)
            if path in produced:
                raise ValueError('Repeated artifact producer: ' + value)
            produced[path] = {'producer': stage['id'], 'produced_at': index,
                              'last_consumer': stage['id'], 'last_use_at': index, 'consumers': []}
    for index, stage in enumerate(stages):
        reads = list(stage['argv'])
        if stage['payload']:
            def paths(value):
                if isinstance(value, dict):
                    return [p for item in value.values() for p in paths(item)]
                return [value] if isinstance(value, str) else []
            reads.extend(paths(stage['payload']['data']))
        for output, row in produced.items():
            if index <= row['produced_at']:
                continue
            if any(Path(value).is_relative_to(output) for value in reads if value.startswith('/')):
                row['consumers'].append(stage['id'])
                row.update(last_consumer=stage['id'], last_use_at=index)
    return {str(path): row for path, row in produced.items()}


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


def execute(stages, root, pins=None, resume=False):
    pins = pins or {}
    verify_pins(pins)
    root = Path(root).resolve()
    if shutil.disk_usage(root.parent).free < 16 * 1024**3:
        raise RuntimeError('Require 16 GiB free for retained-stage diagnostic pipeline; no automatic cleanup')
    if not resume:
        root.mkdir()  # Reuse is only permitted through explicit validated resume.
    elif not root.is_dir():
        raise ValueError('Resume root missing')
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_locked(stages, root, pins, resume)


def execute_locked(stages, root, pins, resume):
    start_index = 0
    if resume:
        report = json.loads((root / 'pipeline-report.json').read_text())
        start_index = validate_resume(report, stages, pins)
        archive = quarantine_failed_stage(root, stages[start_index], report)
        report.setdefault('resumed_attempts', []).append(archive)
        report['stages'] = report['stages'][:start_index]
        report.update(status='running', active_stage=None)
        report.pop('error', None)
        report.pop('failure_kind', None)
    else:
        report = {'status': 'running', 'stages': [], 'plan': stages, 'dependency_sha256': pins,
              'artifact_lifetimes': artifact_lifetimes(stages),
              'scope': 'Diagnostic pipeline; normal child-failure resume only, no SIGKILL recovery or quality promotion.'}
    def save():
        atomic_json(root / 'pipeline-report.json', report)
    save()
    started = time.monotonic()
    try:
        for stage in stages[start_index:]:
            verify_pins(pins)
            report['active_stage'] = stage['id']
            save()
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
                if result.returncode > 0:
                    report['failure_kind'] = 'child-exit'
                raise RuntimeError('Pipeline stage failed: ' + stage['id'])
            if stage['capture']:
                data = json.loads(log.read_text())
                with Path(stage['capture']).open('x') as stream:
                    json.dump(data, stream, indent=2)
            # Persist content identities only after successful execution and
            # capture publication. This is prerequisite evidence for resume,
            # not permission to reuse an interrupted output directory yet.
            report['stages'][-1]['artifact_checkpoint'] = stage_checkpoint(stage, log)
            report['stages'][-1]['completed'] = True
            report['active_stage'] = None
            save()
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
    parser.add_argument('--resume', action='store_true',
                        help='Resume a recorded normal child failure; unobserved or signal interruption is refused')
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
        # Refuse a direct uncontained CLI run before hashing or extracting.
        # This verifies limits, not that an independent monitor is present.
        require_memory_scope()
        pins = pinned_files(Path(__file__).resolve().parents[1], spec)
        print(json.dumps(execute(stages, args.output, pins, resume=args.resume), indent=2))


if __name__ == '__main__':
    main()
