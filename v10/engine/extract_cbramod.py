"""Frozen official CBraMod -> one feature vector per person, without fitting on labels."""
import argparse
import platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from assets import checked_weight, CBRAMOD_SHA
from common import load_json, save_json, sha256, new_output, manifest, validate_folds
from vendor_cbramod.cbramod import CBraMod

CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6']


def chosen_indices(n, limit):
    if n < 1 or limit < 0:
        raise ValueError('Invalid window count/limit')
    return np.arange(n) if limit == 0 or n <= limit else np.unique(np.linspace(0,n-1,limit).round().astype(int))


def load_encoder(weights=None, random_control=False, seed=20261015):
    torch.manual_seed(seed)
    model = CBraMod()
    if not random_control:
        state = torch.load(checked_weight('cbramod',weights), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.proj_out = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode_windows(model, windows, device, batch_size):
    x = np.asarray(windows, dtype='float32')
    if x.ndim != 3 or x.shape[1:] != (16,800) or not np.isfinite(x).all():
        raise ValueError('Expected finite [windows,16,800], already uV/100')
    pieces = []
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            batch = torch.from_numpy(x[start:start+batch_size]).reshape(-1,16,4,200).to(device)
            # Preserve the physical scale already set by the preprocessing. No second /100.
            tokens = model(batch)
            pieces.append(tokens.mean(dim=(1,2)).cpu().numpy())
    z = np.concatenate(pieces)
    vector = np.r_[z.mean(0),z.std(0)].astype('float32')
    if vector.shape != (400,) or not np.isfinite(vector).all():
        raise ValueError('Invalid encoder output')
    return vector


def main(args):
    if args.batch_size < 1: raise ValueError('batch-size must be positive')
    torch.set_num_threads(args.threads)
    root = Path(args.features).resolve(strict=True)
    cfg = load_json(root/'config.json')
    people = manifest(root); plan_hashes = {}
    if args.plans:
        ids = set()
        for path in args.plans:
            plan = load_json(path); validate_folds(plan)
            if sha256(root/'manifest.json') != plan['feature_manifest_sha256']:
                raise ValueError('Plan/feature manifest mismatch')
            ids.update(plan['subject_ids']); plan_hashes[str(path)] = sha256(path)
        ids = sorted(ids)
    else:
        ids = sorted(s for s,r in people.items() if r.get('eeg_windows',0) >= 3)
    if not ids: raise ValueError('No eligible EEG subjects')
    model = load_encoder(args.weights,args.random_control,args.seed).to(args.device)
    out = new_output(args.output,root,cfg['raw_root'],cfg['processed_root'])
    protocol = {'encoder':'CBraMod random control' if args.random_control else 'official pretrained CBraMod',
        'weight_sha256':None if args.random_control else CBRAMOD_SHA, 'seed':args.seed,
        'vendor_provenance':load_json(Path(__file__).parent/'vendor_cbramod/provenance.json'),
        'feature_manifest_sha256':sha256(root/'manifest.json'), 'plans':plan_hashes,
        'subject_ids':ids,'feature_dimension':400,'sample_rate_hz':200,'channels':CHANNELS,
        'units':'uV/100, no additional division', 'max_windows':args.max_windows,
        'pooling':'mean channel/patch tokens -> window 200D -> subject mean+population SD',
        'preprocessing':'inherited V3 1-45 Hz; original reference; an adaptation, not exact official benchmark preprocessing',
        'fitted_on_participant_data':False,'torch':torch.__version__,'platform':platform.platform(),
        'status':'running','sources':{}}
    save_json(out/'encoder.json',protocol)
    for i,sid in enumerate(ids):
        path = root/(sid+'.npz'); before = sha256(path)
        with np.load(path,allow_pickle=False) as z:
            eeg = z['eeg'].copy()
        keep = chosen_indices(len(eeg),args.max_windows)
        vector = encode_windows(model,eeg[keep],args.device,args.batch_size)
        if sha256(path) != before: raise RuntimeError('Input changed during extraction')
        np.savez_compressed(out/(sid+'.npz'),cbramod=vector,window_indices=keep)
        protocol['sources'][sid] = {'input_sha256':before,'output_sha256':sha256(out/(sid+'.npz')),
            'windows_available':len(eeg),'windows_used':len(keep)}
        if (i+1)%10 == 0 or i+1 == len(ids): print('Encoded',i+1,'/',len(ids),flush=True)
    protocol['status']='complete'; save_json(out/'encoder.json',protocol)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True); p.add_argument('--output',required=True)
    p.add_argument('--plans',nargs='+'); p.add_argument('--weights')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-windows',type=int,default=0,help='0=all accepted windows; positive=deterministic temporal subsample')
    p.add_argument('--threads',type=int,default=2); p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--random-control',action='store_true')
    main(p.parse_args())
