"""Verify portable files, unchanged predecessors, signals, labels and checkpoints."""
from pathlib import Path
import argparse,importlib.metadata
import numpy as np
from common import read,sha
from features_v172 import load_inputs
from prepare_v172 import WEIGHT_SHA
from foundation_v173 import runtime

def main(a):
    root=Path(__file__).resolve().parents[1];p=root/'PACKAGE_MANIFEST.json'
    if p.exists():
        for name,h in read(p)['files'].items():
            if sha(root/name)!=h:raise ValueError('Package file changed: '+name)
    elif not a.allow_unpacked_build:raise FileNotFoundError('Package manifest missing')
    for snapshot in ['V171_SOURCE_SNAPSHOT.json','V172_SOURCE_SNAPSHOT.json']:
        for name,h in read(root/snapshot).items():
            if sha(root/name)!=h:raise ValueError('Inherited source/data changed: '+name)
    meta,records,y=load_inputs(root/'data',root/'extras')
    if sha(root/'weights/cbramod_pretrained.pth')!=WEIGHT_SHA:raise ValueError('CBraMod weights changed')
    if sha(root/'weights/tabpfn-v2-regressor.ckpt')!='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736':raise ValueError('TabPFN weights changed')
    if a.foundation and importlib.metadata.version('tabpfn')!='2.0.9':raise RuntimeError('Install requirements.foundation.txt; TabPFN must be 2.0.9')
    print('Verified:',len(records),'signal subjects;',int(np.isfinite(y).any(1).sum()),'labelled; predecessor code/data unchanged')
    print(runtime())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--foundation',action='store_true');p.add_argument('--allow-unpacked-build',action='store_true');main(p.parse_args())
