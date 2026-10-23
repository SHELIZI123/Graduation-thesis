from pathlib import Path
import argparse,time
import numpy as np
import joblib,torch
from threadpoolctl import threadpool_limits
from common import read,write,new_dir,load_data,sha
from engine import fit_neural,fit_ridge,CONFIGS,RIDGE_LAMBDAS
from train import BASELINES

def merge_inputs(paths):
    protocols=[read(Path(p)/'protocol.json') for p in paths]
    base=protocols[0]
    for p in protocols[1:]:
        for k in ['data_manifest_sha256','plan','schema','neural_configurations','baseline_penalties','architectures','seeds','maximum_epochs','cap','code_sha256']:
            if p[k]!=base[k]:raise ValueError('Incompatible runs: '+k)
    decisions=[]
    for path in paths:
        state=read(Path(path)/'status.json')
        if state['state']!='complete':raise ValueError('Run incomplete')
        decisions.extend(read(Path(path)/'decisions.json'))
    if sorted(d['fold'] for d in decisions)!=list(range(5)):raise ValueError('Need exactly five non-overlapping outer folds')
    return base,sorted(decisions,key=lambda x:x['fold'])

def main(a):
    protocol,decisions=merge_inputs(a.runs);meta,records,y=load_data(a.data)
    if sha(Path(a.data)/'manifest.json')!=protocol['data_manifest_sha256']:raise ValueError('Dataset mismatch')
    out=new_dir(a.output,[a.data,Path(__file__).parent]);dims=meta['dimensions'];schema=meta['schema']
    ix=protocol['plan']['eligible_indices'];training=[records[i] for i in ix];yy=y[ix]
    ids=[meta['subjects'][i]['subject_id'] for i in ix];selections={};models={};start=time.perf_counter()
    for name,mods in BASELINES.items():
        candidates={pen:float(np.mean([v for d in decisions for v in d['baselines'][name]['inner_scores'][str(pen)]])) for pen in RIDGE_LAMBDAS}
        penalty=min(candidates,key=lambda p:(candidates[p],-p))
        artifact=fit_ridge(training,yy,dims,schema,penalty,mods,name=='ridge_mask_only')
        artifact.update(training_ids=ids,test_ids=[]);p=out/(name+'.joblib');joblib.dump(artifact,p,compress=3)
        models[name]=sha(p);selections[name]={'penalty':penalty,'inner_only_scores':candidates}
    for routing in protocol['architectures']:
        candidates=[]
        for config in CONFIGS:
            trials=[v for d in decisions for v in d['neural'][routing]['search'] if v['config']['id']==config['id']]
            epochs=[e for t in trials for e in t['inner_best_epochs']]
            candidates.append({'config':config,'score':float(np.mean([e for t in trials for e in t['inner_scores']])),
                               'epochs':int(np.median(epochs))})
        chosen=min(candidates,key=lambda x:(x['score'],x['config']['hidden']))
        print('Refit',routing,chosen,flush=True)
        artifact,_=fit_neural(training,yy,dims,schema,chosen['config'],routing,protocol['seeds'],chosen['epochs'],a.device,cap=protocol['cap'])
        artifact.update(training_ids=ids,test_ids=[]);p=out/('set_'+routing+'.joblib');joblib.dump(artifact,p,compress=3)
        models['set_'+routing]=sha(p);selections['set_'+routing]=chosen
    write(out/'manifest.json',{'state':'complete','training_ids':ids,'data_manifest_sha256':protocol['data_manifest_sha256'],
                              'models_sha256':models,'selections':selections,'selection_uses_outer_scores':False,
                              'independent_test':False,'exploratory':True,'seconds':time.perf_counter()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):main(a)
