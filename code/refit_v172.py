"""Select on full development cohort inner folds and fit; never aggregate outer epochs."""
from pathlib import Path
import argparse
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from common import new_dir,write,sha
from features_v172 import load_inputs
from preprocessing import group_splits
from train_v172 import tune
from engine_v172 import FitCache

def main(a):
    meta,r,y=load_inputs(a.data,a.extras);ix=np.flatnonzero(np.isfinite(y).any(1));r=[r[i] for i in ix];y=y[ix]
    out=new_dir(a.output,[a.data,a.extras,Path(__file__).parent]);inner=group_splits(list(range(len(r))),[v['group_id'] for v in r],3,20261124)
    cache=FitCache(meta['dimensions'],meta['schema']);decisions={}
    for name in a.names:
        model,_,decision=tune(name,r,y,meta['dimensions'],meta['schema'],inner,a.seeds,a.device,a.profile=='smoke',cache)
        model.update(training_ids=[v['subject_id'] for v in r],training_groups=sorted({v['group_id'] for v in r}),name=name,selection=decision)
        joblib.dump(model,out/(name+'.joblib'),compress=3);decisions[name]=decision;print('Refitted',name,decision['config'],flush=True)
    write(out/'manifest.json',dict(state='complete',training_ids=[v['subject_id'] for v in r],decisions=decisions,
        seeds=a.seeds,profile=a.profile,exploratory=True,independent_validation=False,
        refit_rule='New full-development inner CV; no aggregation of outer hyperparameters or epochs',
        files={p.name:sha(p) for p in out.glob('*.joblib')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--extras',required=True);p.add_argument('--output',required=True)
    p.add_argument('--names',nargs='+',default=['ridge_trimodal','residual_mmpareto']);p.add_argument('--seeds',nargs='+',type=int,default=[20261023,20261024,20261025]);p.add_argument('--profile',choices=['full','smoke'],default='full')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
