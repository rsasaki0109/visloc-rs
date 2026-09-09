"""Execute one source stage inside an enclosing measured native pipeline.

The caller owns resource limits and immutable input publication. This is not
a standalone cold pipeline or a resume implementation.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from build_native_source_recipe import INPUTS, render_nodes_tsv
from replay_native_candidates import sha


def execute_sources(recipe, bindings, binary, run_root):
    """Publish atlas inputs only after all frozen source stages pass.

    This deliberately refuses existing source outputs; pipeline-level restart
    must first validate dependencies rather than silently reusing directories.
    """
    root = Path(run_root).resolve()
    nodes = root / 'nodes.tsv'
    if nodes.exists() or (root / 'sources').exists():
        raise FileExistsError('Source phase output already exists')
    if len(recipe['executions']) != 21 or len(recipe['nodes']) != 23:
        raise ValueError('Require all 21 source executions and 23 atlas nodes')
    manifest = render_nodes_tsv(recipe, root)
    reports = []
    for stage in recipe['executions']:
        reports.append(execute_source(stage, bindings, binary, root))
    # Exclusive publication: an incomplete source phase never publishes nodes.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=root, prefix='.nodes-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(manifest)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, nodes)  # Atomic publication, fails if destination exists.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'status': 'pass', 'sources': reports, 'nodes_tsv': str(nodes),
            'nodes_sha256': sha(nodes), 'scope': 'Source phase only; not cold native E2E.'}


def execute_source(stage, bindings, binary, run_root):
    required = {value[1:-1] for value in INPUTS.values()}
    if set(bindings) != required:
        raise ValueError('Require exactly the source input bindings')
    binary = Path(binary).resolve(strict=True)
    if sha(binary) != stage['binary_sha256']:
        raise ValueError('Source binary hash mismatch')
    identity = stage['id']
    if not identity or Path(identity).name != identity or identity in ('.', '..'):
        raise ValueError('Invalid source identity')
    root = Path(run_root).resolve()
    values = {key: str(Path(value).resolve(strict=True)) for key, value in bindings.items()}
    values.update(mapper_binary=str(binary), run_root=str(root))
    command = [arg.format_map(values) for arg in stage['argv']]
    output = root / 'sources' / identity
    model = output / 'model'
    if command.count('--out-colmap') != 1 or command[command.index('--out-colmap') + 1] != str(model):
        raise ValueError('Source output escapes its stage')
    if not stage['expected_model_hashes']:
        raise ValueError('Require nonempty expected model hashes')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in stage['unset_environment']:
        env.pop(key, None)
    env.update(stage['environment'], MALLOC_ARENA_MAX='1')
    report = {'status': 'running', 'command': command, 'input_paths': values,
              'binary_sha256': stage['binary_sha256'],
              'scope': 'Single source stage; caller must validate immutable input provenance and enforce resource limits.'}
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    started = time.monotonic()
    try:
        with (output / 'run.log').open('w') as log:
            result = subprocess.run(['timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                                     str(stage['timeout_seconds']) + 's', *command],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        report['exit_code'] = result.returncode
        actual = {p.relative_to(model).as_posix(): sha(p)
                  for p in model.rglob('*') if p.is_file()}
        report['model_hashes'] = actual
        if result.returncode != 0 or actual != stage['expected_model_hashes']:
            raise RuntimeError('Source failed or model hashes differ')
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='fail', error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report
