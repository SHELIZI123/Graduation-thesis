"""Full development refit from inner decisions only; no outer-score model promotion."""
from pathlib import Path
import argparse,collections
import numpy as np
import joblib,torch
from threadpoolctl import threadpool_limits
from common import load_data,new_dir,read,write,sha
from train import BASELINES
from engine import fit_ridge,fit_neural,bound
from engine_v171 import fit_classical,fit_new
from train_v171 import predict_artifact
from evaluate_v171 import collect_runs

def mode(values):
    counts=collections.Counter(values)
    return min(counts,key=lambda k:(-counts[k],str(k)))

def main(a):
    protocol,folders=collect_runs(a.runs)
    if sorted(folders)!=list(range(5)) or protocol['profile']!='full':raise ValueError('Full fivefold production run required for full refit')
    meta,records,y=load_data(a.data)
    if sha(Path(a.data)/'manifest.json')!=protocol['data_manifest_sha256']:raise ValueError('Wrong data')
    out=new_dir(a.output,[a.data,Path(__file__).parent]);eligible=protocol['plan']['eligible_indices']
    rr=[records[i] for i in eligible];yy=y[eligible];schema=meta['schema'];dims=meta['dimensions']
    decisions=[read(folders[i]/'decisions.json') for i in sorted(folders)]
    models={};selections={};ids=[meta['subjects'][i]['subject_id'] for i in eligible]
    def save(name,art,decision):
        art.update(training_ids=ids,test_ids=[]);models[name]=art;selections[name]=decision
        joblib.dump(art,out/(name+'.joblib'),compress=3)
    for name,fn in [('training_mean',np.mean),('training_median',np.median)]:
        v=np.array([fn(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in yy.T])
        save(name,dict(kind='constant',values=bound(v[None],schema)[0],schema=schema),{})
    for name,modalities in BASELINES.items():
        penalty=mode([d[name]['penalty'] for d in decisions])
        save(name,fit_ridge(rr,yy,dims,schema,penalty,modalities,name=='ridge_mask_only'),dict(penalty=penalty))
    from engine_v171 import classical_configs
    for name in classical_configs(protocol.get('foundation',False)):
        configs=decisions[0][name]['configurations']
        choice=[mode([d[name]['selected_per_target'][j] for d in decisions]) for j in range(len(schema))]
        bank={str(k):fit_classical(rr,yy,dims,schema,configs[k],device=a.device) for k in sorted(set(choice))}
        save(name,dict(kind='bank',models=bank,selection=choice,schema=schema),dict(selected_per_target=choice,configurations=configs))
    for name in protocol['recipes']:
        config_id=mode([d[name]['search'][d[name]['selected']]['config']['id'] for d in decisions])
        chosen=[d[name]['search'][d[name]['selected']] for d in decisions]
        config=next(x['config'] for x in chosen if x['config']['id']==config_id)
        epochs=int(np.median([x['refit_epochs'] for x in chosen]))
        if name=='v17_ple':art,_=fit_neural(rr,yy,dims,schema,config,'ple',protocol['seeds'],epochs,a.device,cap=32)
        else:art,_=fit_new(rr,yy,dims,schema,config,name,protocol['seeds'],epochs,a.device,cap=protocol['cap'])
        save(name,art,dict(config=config,epochs=epochs))
    names=decisions[0]['stack_l1']['names']
    assert all(d['stack_l1']['names']==names for d in decisions)
    weights=np.mean([d['stack_l1']['weights'] for d in decisions],axis=0)
    save('stack_l1',dict(kind='stack',models={n:models[n] for n in names},names=names,weights=weights,schema=schema),
         dict(names=names,weights=weights.tolist(),rule='Mean of outer-training-inner-OOF-fitted weights; no outer-score use'))
    p=predict_artifact(models['stack_l1'],rr[:2],a.device)
    assert p.shape==(min(2,len(rr)),len(schema)) and np.isfinite(p).all()
    write(out/'manifest.json',dict(state='complete',default_candidate='stack_l1',training_ids=ids,
        data_manifest_sha256=protocol['data_manifest_sha256'],selection_uses_outer_scores=False,
        exploratory=True,independent_test=False,models={n:n+'.joblib' for n in models},
        sha256={p.name:sha(p) for p in out.glob('*.joblib')}))
    write(out/'selections.json',selections)
    print('Refit complete:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
