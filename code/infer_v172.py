"""Signal-only inference: targets.npz is never opened by this entry point."""
from pathlib import Path
import argparse
import torch,joblib
from threadpoolctl import threadpool_limits
from common import csvwrite
from features_v172 import load_inputs
from engine_v172 import predict_candidate

def main(a):
    meta,records,_=load_inputs(a.data,a.extras,with_targets=False);model=joblib.load(a.model)
    if [t['name'] for t in meta['schema']]!=[t['name'] for t in model['schema']]:raise ValueError('Target schema differs')
    ids=set(a.ids) if a.ids else {r['subject_id'] for r in records}
    if not ids<={r['subject_id'] for r in records}:raise ValueError('Unknown inference ID')
    rr=[r for r in records if r['subject_id'] in ids]
    if not a.allow_development and ({r['subject_id'] for r in rr}&set(model['training_ids']) or {r['group_id'] for r in rr}&set(model['training_groups'])):
        raise ValueError('Training subject/group requested; --allow-development is required for development replay')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    p=predict_candidate(model,rr,a.device)
    csvwrite(out,[dict(subject_id=r['subject_id'],**{t['name']:float(p[i,j]) for j,t in enumerate(model['schema'])}) for i,r in enumerate(rr)])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--extras',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--ids',nargs='+');p.add_argument('--allow-development',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
