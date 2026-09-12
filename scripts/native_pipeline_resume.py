"""Recovery of reported normal child failures, not unobserved SIGKILL recovery."""
from pathlib import Path
import uuid

from benchmark_electro import atomic_json
from native_pipeline_checkpoint import verify_checkpoint


def validate_resume(report, stages, pins):
    if report.get('status') != 'fail' or report.get('failure_kind') != 'child-exit':
        raise ValueError('Resume requires a recorded normal child exit failure')
    if report.get('plan') != stages or report.get('dependency_sha256') != pins:
        raise ValueError('Resume plan or dependency identities changed')
    rows = report.get('stages', [])
    if not rows or len(rows) > len(stages):
        raise ValueError('Invalid completed stage prefix')
    for index, row in enumerate(rows):
        if row.get('id') != stages[index]['id']:
            raise ValueError('Stage prefix order changed')
        if index < len(rows) - 1:
            if row.get('completed') is not True or row.get('exit_code') != 0:
                raise ValueError('Missing successful prefix checkpoint')
            verify_checkpoint(row.get('artifact_checkpoint', {}))
        elif row.get('completed') or type(row.get('exit_code')) is not int or row['exit_code'] <= 0:
            raise ValueError('Require a normally exited failed last child')
    if report.get('active_stage') != rows[-1]['id']:
        raise ValueError('Failed active stage mismatch')
    return len(rows) - 1


def quarantine_failed_stage(root, stage, report):
    root = Path(root).resolve()
    paths = {root / (stage['id'] + '.log')}
    for index, arg in enumerate(stage['argv'][:-1]):
        if arg in ('--output', '--output-directory'):
            paths.add(Path(stage['argv'][index + 1]))
    if stage['capture']:
        paths.add(Path(stage['capture']))
    if stage['payload']:
        paths.add(Path(stage['payload']['path']))
    for path in paths:
        if not path.is_absolute() or path == root or not path.resolve().is_relative_to(root):
            raise ValueError('Failed artifact outside pipeline root')
        if path.is_symlink():
            raise ValueError('Refuse linked failed output root')
        for row in report['stages'][:-1]:
            for name in row['artifact_checkpoint']:
                protected = Path(name)
                if path.is_relative_to(protected) or protected.is_relative_to(path):
                    raise ValueError('Failed output overlaps completed artifact')
    selected = [path for path in paths if not any(path != other and path.is_relative_to(other) for other in paths)]
    archive = root / 'failed-attempts' / uuid.uuid4().hex
    archive.mkdir(parents=True)
    atomic_json(archive / 'pipeline-report.json', report)
    for path in sorted(selected):
        if path.exists():
            target = archive / 'artifacts' / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
    return str(archive)
