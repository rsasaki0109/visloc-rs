#!/usr/bin/env python3
"""Compare one new integration binary with frozen atlas model references.

Uses retained validated source nodes. Run under launch_native_measurement.py;
this is a diagnostic parity probe, not cold E2E or a quality improvement.
"""
import argparse
import json
from pathlib import Path

from execute_native_atlas import execute_atlas
from native_atlas_recipe import build
from replay_native_candidates import sha
from run_native_pipeline import require_memory_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration-binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    require_memory_scope()
    binary = args.integration_binary.resolve(strict=True)
    if sha(binary) != args.binary_sha256:
        raise ValueError('Integration binary changed')
    dataset = Path('/home/sasaki/datasets/openloris')
    evidence = Path(__file__).resolve().parents[1] / 'benchmarks/electro'
    reference = json.loads((evidence / 'm8-native-mapping-bound-v2.json').read_text())
    source_root = Path(reference['run_root'])
    if sha(source_root / 'nodes.tsv') != reference['nodes_sha256']:
        raise ValueError('Source node manifest changed')
    for name, expected in reference['reference_file_hashes'].items():
        if name.startswith('sources/') and sha(source_root / name) != expected:
            raise ValueError('Source model changed: ' + name)
    rig = dataset / 'corridor1-1-m8-visloc-rig/tier-10000-champion/rig-manifest.txt'
    inputs = json.loads((evidence / 'm8-native-mapping-bound-inputs-v1.json').read_text())
    if sha(rig) != inputs['rig_manifest']['sha256']:
        raise ValueError('Rig manifest changed')
    stages = build(evidence)
    for stage in stages[1:]:
        # Explicit behavioral comparison of a new binary, not frozen-binary replay.
        stage['binary_sha256'] = args.binary_sha256
        if args.diagnostic:
            stage['environment'].update(VISLOC_SFM_DEBUG_BA='1', VISLOC_SFM_DEBUG_BA_STEPS='1',
                VISLOC_SFM_DEBUG_BA_SCHUR_SLOT='0', VISLOC_SFM_DEBUG_BA_SPARSE_FIRST_WINDOW='1')
    root = args.output.resolve()
    root.mkdir()  # Fresh diagnostic root, never reuse previous reports/models.
    reports = execute_atlas(stages, {
        'rig_manifest': rig,
        'nodes_tsv': source_root / 'nodes.tsv',
    }, {
        'stitch_binary': dataset / 'corridor1-1-m8-regenerated-atlas-v1/stitch-cfe11c6',
        'integration_binary': binary,
    }, root)
    print(json.dumps({'status': 'reference-parity-pass', 'diagnostic': args.diagnostic,
                      'reports': reports}, indent=2))


if __name__ == '__main__':
    main()
