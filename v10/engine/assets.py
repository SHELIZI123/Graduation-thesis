"""Pinned public assets. No participant data is used by the downloader."""
import argparse
from pathlib import Path
import urllib.request
from common import sha256, save_json

ROOT = Path(__file__).resolve().parent
CBRAMOD_SHA = '0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178'
TABPFN_SHA = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
ASSETS = {
    'cbramod': ('cbramod_pretrained.pth', CBRAMOD_SHA,
        'https://huggingface.co/weighting666/CBraMod/resolve/500543c7e30bda1b22bfd51a49301b238dee21fd/pretrained_weights.pth'),
    'tabpfn': ('tabpfn-v2-regressor.ckpt', TABPFN_SHA,
        'https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt'),
}


def checked_weight(name, path=None):
    filename, expected, _ = ASSETS[name]
    path = Path(path) if path else ROOT/'weights'/filename
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f'Missing or incorrect {name} weights: {path}; run assets.py first')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(ASSETS), default=list(ASSETS))
    args = parser.parse_args()
    destination = ROOT/'weights'; destination.mkdir(exist_ok=True)
    result = {}
    for name in args.models:
        filename, expected, url = ASSETS[name]
        path = destination/filename
        if not path.exists():
            partial = destination/(filename+'.part')
            with urllib.request.urlopen(url, timeout=120) as src, partial.open('xb') as dst:
                while chunk := src.read(1024*1024):
                    dst.write(chunk)
            if sha256(partial) != expected:
                raise ValueError('Downloaded weight hash mismatch; retained .part for diagnosis')
            partial.replace(path)
        checked_weight(name, path)
        result[name] = {'url': url, 'sha256': expected, 'bytes': path.stat().st_size}
        print(name, 'verified', flush=True)
    save_json(destination/'verified_assets.json', result)


if __name__ == '__main__':
    main()
