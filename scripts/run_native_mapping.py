#!/usr/bin/env python3
"""Run all source mappers and atlas integration from validated frontend artifacts.

Use inside launch_native_measurement.py for resource containment. This suffix
uses existing frontend artifacts and must not be reported as cold native E2E.
"""
import argparse
import json
from pathlib import Path
import time

from build_native_source_recipe import build as source_recipe, INPUTS
from native_atlas_recipe import build as atlas_recipe
from execute_native_source import execute_sources
from execute_native_atlas import execute_atlas
from replay_native_candidates import bind_feature_input, sha


def validate_inputs(spec):
    required = {value[1:-1] for value in INPUTS.values()}
    if set(spec) != required:
        raise ValueError('Require all five mapping inputs')
    bindings, provenance = {}, {}
    for name, row in spec.items():
        path = Path(row['path']).resolve(strict=True)
        if name.endswith('_features'):
            manifest = Path(row['manifest']).resolve(strict=True)
            if sha(manifest) != row['manifest_sha256']:
                raise ValueError('Feature manifest hash mismatch: ' + name)
            bind_feature_input(['validate', '--features-dir', str(path)], manifest)
        elif sha(path) != row['sha256']:
            raise ValueError('Mapping input hash mismatch: ' + name)
        bindings[name] = str(path)
        provenance[name] = dict(row, path=str(path))
    return bindings, provenance


def run(sources, atlas, bindings, binaries, output, provenance):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'inputs': provenance, 'source_recipe': sources,
              'atlas_recipe': atlas, 'scope': 'Mapping suffix from existing frontend; not cold native E2E.'}
    report_path = output / 'mapping-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    try:
        report['sources'] = execute_sources(sources, bindings, binaries['mapper_binary'], output)
        report['atlas'] = execute_atlas(atlas, {'rig_manifest': bindings['rig_manifest'],
                                               'nodes_tsv': report['sources']['nodes_tsv']},
                                       {key: binaries[key] for key in ('stitch_binary', 'integration_binary')}, output)
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='fail', error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='Five input records: path plus SHA256 or feature manifest and its SHA256')
    for name in ('mapper', 'stitch', 'integration'):
        parser.add_argument('--' + name + '-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parents[1] / 'benchmarks/electro'
    sources, atlas = source_recipe(evidence), atlas_recipe(evidence)
    binaries = {name + '_binary': getattr(args, name + '_binary').resolve(strict=True)
                for name in ('mapper', 'stitch', 'integration')}
    for stage, key in [(s, 'mapper_binary') for s in sources['executions']] + [
            (s, 'stitch_binary' if s['id'] == 'stitch' else 'integration_binary') for s in atlas]:
        if sha(binaries[key]) != stage['binary_sha256']:
            raise ValueError('Mapping binary hash mismatch: ' + key)
    if args.output.exists():
        raise FileExistsError(args.output)
    bindings, provenance = validate_inputs(json.loads(args.inputs.read_text()))
    if args.validate_only:
        print(json.dumps({'status': 'validated-not-executed', 'inputs': provenance}, indent=2))
    else:
        print(json.dumps(run(sources, atlas, bindings, binaries, args.output, provenance), indent=2))


if __name__ == '__main__':
    main()
