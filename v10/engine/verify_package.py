"""Verify code-package checksums after download/extraction; no patient access."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
checks=json.loads((root/'CHECKSUMS.json').read_text(encoding='utf-8'))
for name,expected in checks.items():
    p=root/name
    if root not in p.resolve().parents:raise ValueError('Unsafe checksum path')
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('Checksum mismatch: '+name)
print('Verified',len(checks),'package files')
