"""Restore ignored runtime files from an existing, trusted V10 server package."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def install(source):
    source = Path(source).resolve(strict=True)
    if source == ROOT or ROOT in source.parents or source in ROOT.parents:
        raise ValueError('Source package must be outside the repository version directory')
    checks = json.loads((ROOT / 'RUNTIME_SHA256.json').read_text(encoding='utf-8'))
    pending = []
    for name, expected in checks.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe runtime path')
        src = (source / relative).resolve(strict=True)
        dst = (ROOT / relative).resolve()
        if source not in src.parents or ROOT not in dst.parents:
            raise ValueError('Runtime path escapes package')
        if digest(src) != expected:
            raise ValueError('Source checksum mismatch: ' + name)
        if dst.exists():
            if digest(dst) != expected:
                raise ValueError('Existing runtime differs; refusing overwrite: ' + name)
        else:
            pending.append((src, dst, expected))
    # Validate every input before the first copy. Only new files may be created.
    for src, dst, expected in pending:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as inp, dst.open('xb') as out:
            for block in iter(lambda: inp.read(1024 * 1024), b''):
                out.write(block)
        if digest(dst) != expected:
            raise RuntimeError('Copied runtime checksum mismatch')
    print(f'Runtime verified: {len(checks)} files; installed {len(pending)} new files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Extracted original V10 package outside repository')
    install(parser.parse_args().source)
