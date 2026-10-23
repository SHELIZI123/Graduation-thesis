"""Check exact upstream code hashes without accessing any private cohort files."""
from pathlib import Path,PurePosixPath
import hashlib,json

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(root=ROOT):
    mapping=json.loads((root/'PUBLIC_SOURCE_SHA256.json').read_text(encoding='utf-8'))
    for name,digest in mapping.items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe source path')
        if sha(root.joinpath(*p.parts))!=digest:raise ValueError('Published source changed: '+name)
    return mapping

if __name__=='__main__':print('Verified unchanged source files:',len(verify()))
