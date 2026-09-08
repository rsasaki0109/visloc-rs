"""Translate frozen matching argv to explicit Python-runner matcher settings."""
import math
import shlex


def runner_match_flags(command):
    options = {}
    boolean = {'--stream-match-features', '--shared-snapshot-envelope'}
    paths = {'--features-dir', '--input-colmap-calibration',
             '--persistent-match-worker-plan', '--out-colmap'}
    fixed = {'--feature-extractor': 'files', '--verification-mode': 'full',
             '--matcher': 'nn', '--mapper': 'incremental'}
    forwarded = ['--feature-suffix', '--image-suffix', '--min-matches', '--match-ratio']
    i = 1
    while i < len(command):
        flag = command[i]
        if flag in options or flag not in boolean | paths | fixed.keys() | set(forwarded):
            raise ValueError(f'Unknown or repeated matching option: {flag}')
        if flag in boolean:
            options[flag] = True
            i += 1
        else:
            if i + 1 >= len(command) or command[i + 1].startswith('--'):
                raise ValueError(f'Missing matching value: {flag}')
            options[flag] = command[i + 1]
            i += 2
    for flag, value in fixed.items():
        if options.get(flag) != value:
            raise ValueError(f'Runner cannot reproduce {flag}')
    if not (paths | set(forwarded)) <= options.keys():
        raise ValueError('Missing required matching option')
    ratio = float(options['--match-ratio'])
    if int(options['--min-matches']) <= 0 or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('Invalid matching thresholds')
    return [item for flag in forwarded for item in (flag, options[flag])]


def flags_from_timing(path):
    line = path.read_text().splitlines()[0]
    argv = shlex.split(shlex.split(line.split('Command being timed: ', 1)[1])[0])
    return runner_match_flags(argv)
