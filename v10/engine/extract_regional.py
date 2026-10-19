"""Frozen CBraMod with fixed regional pooling; no cohort-fitted feature extraction."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import load_json,save_json,manifest,sha256,new_output
from benchmark import validate_plan,protected_inputs
from extract_cbramod import load_encoder,chosen_indices,CHANNELS
from improved_models import REGIONS
from assets import CBRAMOD_SHA


def pool_tokens(tokens):
    if tokens.ndim!=4 or tokens.shape[1:]!=(16,4,200):raise ValueError('Unexpected encoder shape')
    time=tokens.mean(dim=2)
    regions=torch.stack([time[:,g,:].mean(dim=1) for g in REGIONS.values()],dim=1)
    return regions.flatten(1),time.mean(dim=1)


def encode(model,windows,device,batch_size):
    x=np.asarray(windows,dtype='float32')
    if x.ndim!=3 or x.shape[1:]!=(16,800) or not np.isfinite(x).all() or len(x)<1:
        raise ValueError('Expected finite [windows,16,800], already uV/100')
    regional=[];global_vectors=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            tokens=model(torch.from_numpy(x[start:start+batch_size]).reshape(-1,16,4,200).to(device))
            r,g=pool_tokens(tokens);regional.append(r.cpu().numpy());global_vectors.append(g.cpu().numpy())
    r=np.concatenate(regional);g=np.concatenate(global_vectors)
    return np.r_[r.mean(0),r.std(0)].astype('float32'),np.r_[g.mean(0),g.std(0)].astype('float32')


def main(args):
    if args.batch_size<1 or args.threads<1 or args.max_windows<0:raise ValueError('Invalid extraction arguments')
    torch.set_num_threads(args.threads)
    root=Path(args.features).resolve(strict=True);people=manifest(root);ids=set();plans={}
    for p in args.plans:
        plan=load_json(p);validate_plan(plan,sha256(root/'manifest.json'));ids.update(plan['subject_ids']);plans[str(p)]=sha256(p)
    if not ids or ids-set(people):raise ValueError('Missing extraction people')
    out=new_output(args.output,*protected_inputs(root));model=load_encoder(args.weights).to(args.device)
    record={'status':'running','feature_manifest_sha256':sha256(root/'manifest.json'),'plans':plans,
        'weight_sha256':CBRAMOD_SHA,'channels':CHANNELS,'regions':REGIONS,'regional_dimension':2000,
        'pooling':'patch mean -> channel means within fixed regions -> window mean + population SD',
        'global_control':'channel/patch mean -> window mean + population SD; 400D',
        'scale':'inherited uV/100; no additional normalization','max_windows':args.max_windows,
        'fitted_on_participant_data':False,'code_sha256':sha256(__file__),'sources':{}}
    save_json(out/'encoder.json',record)
    for i,sid in enumerate(sorted(ids)):
        path=root/(sid+'.npz');before=sha256(path)
        with np.load(path,allow_pickle=False) as z:eeg=z['eeg'].copy()
        keep=chosen_indices(len(eeg),args.max_windows)
        regional,global_control=encode(model,eeg[keep],args.device,args.batch_size)
        if regional.shape!=(2000,) or not np.isfinite(regional).all():raise ValueError('Invalid regional embedding')
        if sha256(path)!=before:raise RuntimeError('Source changed')
        p=out/(sid+'.npz');np.savez_compressed(p,regional=regional,cbramod_global=global_control,window_indices=keep)
        record['sources'][sid]={'input_sha256':before,'output_sha256':sha256(p),'windows_used':len(keep),'windows_available':len(eeg)}
        if (i+1)%10==0 or i+1==len(ids):print('Regional encoded',i+1,'/',len(ids),flush=True)
    record['status']='complete';save_json(out/'encoder.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--plans',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--weights')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-windows',type=int,default=0)
    main(p.parse_args())
