"""Participant-only frozen EEG augmentation. Never modify source files or fit labels."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
from torch import nn
from common import read,write,sha
from core_extract import covariances
from vendor_cbramod.cbramod import CBraMod

WEIGHT_SHA='0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178'

def encoder(weights,device):
    if sha(weights)!=WEIGHT_SHA:raise ValueError('Official CBraMod weight hash mismatch')
    model=CBraMod();model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.proj_out=nn.Identity();model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    return model

def encode(model,x,device,batch):
    out=[]
    if x.ndim!=3 or x.shape[1:]!=(16,800) or not np.isfinite(x).all():raise ValueError('Invalid raw EEG')
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            v=torch.as_tensor(x[start:start+batch],dtype=torch.float32,device=device).reshape(-1,16,4,200)
            out.append(model(v).mean((1,2)).cpu().numpy())
    z=np.concatenate(out)
    return np.r_[z.mean(0),z.std(0)].astype(np.float32)

def main(a):
    data=Path(a.data);source=Path(a.signals);out=Path(a.output)
    if out.resolve()==source.resolve() or source.resolve() in out.resolve().parents:raise ValueError('Output overlaps signal source')
    out.mkdir(exist_ok=False,parents=True)
    meta=read(data/'manifest.json');raw=read(source/'manifest.json')
    if sha(source/'manifest.json')!=meta['source_manifest_sha256']:raise ValueError('Wrong upstream signal dataset')
    lookup={r['subject_id']:r for r in raw['subjects']};sources={r['subject_id']:r for r in meta['source_files']}
    protected={str(p):sha(p) for p in [data/'manifest.json',data/'targets.npz',source/'manifest.json']}
    model=encoder(a.weights,a.device);info=dict(state='running',data_manifest_sha256=sha(data/'manifest.json'),
        weight_sha256=WEIGHT_SHA,subjects={},encoder_frozen=True,cohort_fitting=False,labels_used=False,
        units='existing uV/100; no second division',sfreq=200,windows='all accepted 4-second windows',
        embedding='tokens mean(channel,time); window mean+population SD; 400D',
        covariance='5 frequency bands; per-window zero-phase filter; .1 isotropic shrinkage; arithmetic mean per person',
        code_sha256=sha(Path(__file__)))
    write(out/'manifest.json',info);start=time.perf_counter()
    for n,e in enumerate(meta['subjects']):
        sid=e['subject_id'];p=source/lookup[sid]['file'];h=sha(p)
        if h!=sources[sid]['signal_sha256']:raise ValueError('Source hash mismatch: '+sid)
        cov=np.full((5,16,16),np.nan);emb=np.full(400,np.nan,np.float32)
        if e['modality_available']['eeg']:
            with np.load(p,allow_pickle=False) as z:x=z['eeg'].copy()
            cov=covariances(x);emb=encode(model,x,a.device,a.batch_size)
        if sha(p)!=h:raise RuntimeError('Source modified during extraction')
        dest=out/(sid+'.npz');np.savez_compressed(dest,cov=cov,embedding=emb)
        info['subjects'][sid]=dict(file=dest.name,sha256=sha(dest),source_sha256=h,eeg_available=e['modality_available']['eeg'])
        if (n+1)%20==0 or n+1==len(meta['subjects']):
            write(out/'manifest.json',info);print('Prepared',n+1,'/',len(meta['subjects']),'seconds',round(time.perf_counter()-start),flush=True)
    assert protected=={p:sha(p) for p in protected}
    info.update(state='complete',seconds=time.perf_counter()-start,source_files_modified=False)
    write(out/'manifest.json',info)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--signals',required=True)
    p.add_argument('--output',required=True);p.add_argument('--weights',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);main(a)
