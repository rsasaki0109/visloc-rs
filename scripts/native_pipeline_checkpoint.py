"""Content checkpoints for completed pipeline stages; not a resume executor."""
from pathlib import Path
import os
import stat

from replay_native_candidates import sha


def snapshot(path):
    path = Path(path)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        if not path.is_file():
            raise ValueError('Only file symlinks are checkpointable: ' + str(path))
        return {'.': {'kind': 'symlink', 'target': os.readlink(path), 'sha256': sha(path)}}
    if stat.S_ISREG(mode):
        return {'.': {'kind': 'file', 'sha256': sha(path)}}
    if not stat.S_ISDIR(mode):
        raise ValueError('Unsupported checkpoint artifact: ' + str(path))
    result = {'.': {'kind': 'directory'}}
    # scandir does not follow directory symlinks; snapshot rejects them.
    with os.scandir(path) as entries:
        for entry in sorted(entries, key=lambda value: value.name):
            for relative, row in snapshot(Path(entry.path)).items():
                name = entry.name if relative == '.' else entry.name + '/' + relative
                result[name] = row
    return result


def stage_checkpoint(stage, log):
    paths = {str(Path(log).absolute())}
    argv = stage['argv']
    for index, arg in enumerate(argv[:-1]):
        if arg in ('--output', '--output-directory'):
            paths.add(str(Path(argv[index + 1]).absolute()))
    if stage['capture']:
        paths.add(str(Path(stage['capture']).absolute()))
    if stage['payload']:
        paths.add(str(Path(stage['payload']['path']).absolute()))
    return {path: snapshot(path) for path in sorted(paths)}


def verify_checkpoint(checkpoint):
    if not checkpoint:
        raise ValueError('Empty stage checkpoint')
    for name, expected in checkpoint.items():
        if snapshot(name) != expected:
            raise ValueError('Completed stage artifact changed: ' + name)
