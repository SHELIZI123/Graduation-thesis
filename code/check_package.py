from pathlib import Path
import sys
from common import read,sha,load_data

root=Path(__file__).resolve().parents[1]
checks=read(root/'PACKAGE_CONTENTS_SHA256.json')
for name,digest in checks.items():
    path=(root/name).resolve()
    if root.resolve() not in path.parents:raise ValueError('Invalid package path')
    if sha(path)!=digest:raise ValueError('Changed package file: '+name)
meta,records,y=load_data(root/'data')
assert len(records)==len(meta['subjects'])==y.shape[0]
print(f'Package verified: {len(checks)} files, {len(records)} signal subjects, {y.shape[1]} target columns.')
