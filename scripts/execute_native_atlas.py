"""Execute the atlas suffix; caller owns immutable inputs and resource limits.

This is not a cold frontend executor, nor a restart implementation.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from replay_native_candidates import sha


def execute_atlas(stages, bindings, binaries, run_root):
    root = Path(run_root).resolve()
    if set(bindings) != {'rig_manifest', 'nodes_tsv'}:
        raise ValueError('Require rig and source node bindings')
    if set(binaries) != {'stitch_binary', 'integration_binary'}:
        raise ValueError('Require both atlas binaries')
    if [s['id'] for s in stages] != ['stitch', 'integrate-tail', 'integrate-main']:
        raise ValueError('Require ordered complete atlas stages')
    values = {key: str(Path(path).resolve(strict=True))
              for key, path in dict(bindings, **binaries).items()}
    values['run_root'] = str(root)
    plans = []
    for stage, relative, binary_key in zip(stages, ['atlas', 'integrated/tail', 'integrated/main'],
                                           ['stitch_binary', 'integration_binary', 'integration_binary']):
        output = root / relative
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        if sha(Path(values[binary_key])) != stage['binary_sha256']:
            raise ValueError('Atlas binary hash mismatch')
        command = [arg.format_map(values) for arg in stage['argv']]
        if command[0] != values[binary_key]:
            raise ValueError('Atlas executable binding mismatch')
        flags = {'--out-dir': str(output)} if stage['id'] == 'stitch' else {
            '--out-dir': str(output / 'model'), '--pre-ba-out-dir': str(output / 'pre-ba')}
        flags.update({'--rig-manifest': values['rig_manifest'], '--nodes-tsv': values['nodes_tsv']})
        for flag, expected in flags.items():
            if command.count(flag) != 1 or command.index(flag) + 1 == len(command) or command[command.index(flag) + 1] != expected:
                raise ValueError('Invalid atlas binding: ' + flag)
        if not stage['expected_files'] or any(Path(p).is_absolute() or '..' in Path(p).parts
                                              for p in stage['expected_files']):
            raise ValueError('Invalid expected atlas files')
        if type(stage['timeout_seconds']) is not int or stage['timeout_seconds'] <= 0:
            raise ValueError('Invalid atlas timeout')
        plans.append((stage, output, command))
    reports = []
    for stage, output, command in plans:
        output.mkdir(parents=True, exist_ok=False)
        env = {k: v for k, v in os.environ.items() if not k.startswith('VISLOC_')}
        env.update(stage['environment'], MALLOC_ARENA_MAX='1')
        report = {'status': 'running', 'command': command, 'binary_sha256': stage['binary_sha256'],
                  'input_hashes': {key: sha(Path(values[key])) for key in bindings},
                  'scope': 'Atlas suffix only; not cold native E2E.'}
        report_path = output / 'execution-report.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        started = time.monotonic()
        try:
            with (output / 'execution.log').open('w') as log:
                result = subprocess.run(['timeout', '--foreground', '--signal=TERM', '--kill-after=10s',
                                         str(stage['timeout_seconds']) + 's', *command],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            report['exit_code'] = result.returncode
            report['file_hashes'] = {name: sha(output / name) for name in stage['expected_files']}
            if result.returncode != 0 or report['file_hashes'] != stage['expected_files']:
                raise RuntimeError('Atlas failed or reference hashes differ')
            report['status'] = 'pass'
        except BaseException as error:
            report.update(status='fail', error=str(error))
            raise
        finally:
            report['wall_seconds'] = time.monotonic() - started
            report_path.write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
    return reports
