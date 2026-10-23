"""Subject-grouped nested evaluation; fixed epoch configurations; no stacking."""
from pathlib import Path
import argparse,time,sys,platform
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from common import write,read,csvwrite,sha,new_dir
from train import make_plan
from preprocessing import target_stats,normalized_mae
from features_v171 import focus_mae
from features_v172 import load_inputs
from engine_v172 import configs,FitCache,fit_candidate,predict_candidate

CONTROLS=['training_mean','training_median','ridge_trimodal','ridge_eeg','ridge_eye','ridge_heart','ridge_eeg_eye','ridge_mask_only','ridge_fixed']
METHODS=['ridge_summary','ridge_tangent','ridge_frozen','residual_plain','residual_aux','residual_mmpareto','tabpfn_v2']

def tune(name,records,y,dims,schema,inner,seeds,device,smoke=False,cache=None):
    cache=cache or FitCache(dims,schema)
    grid=[{}] if name.startswith('training_') else configs(name,smoke)
    banks=[];search=[]
    for config in grid:
        ip=np.full_like(y,np.nan);scores=[]
        for tr,va in inner:
            model=fit_candidate(name,[records[i] for i in tr],y[tr],dims,schema,config,seeds,device,cache)
            p=predict_candidate(model,[records[i] for i in va],device);ip[va]=p
            criterion=normalized_mae if name in CONTROLS else focus_mae
            scores.append(criterion(y[va],p,target_stats(y[tr])[1],schema))
        banks.append(ip);search.append(dict(config=config,fold_scores=scores,score=float(np.mean(scores))))
    # Prefer simpler/stronger-regularized candidate for exact score ties.
    k=min(range(len(grid)),key=lambda i:(search[i]['score'],grid[i].get('epochs',0),-grid[i].get('penalty',0)))
    art=fit_candidate(name,records,y,dims,schema,grid[k],seeds,device,cache)
    return art,banks[k],dict(selected=k,config=grid[k],search=search,
        selected_oof_is_tuning_diagnostic_not_independent_score=True)

def main(a):
    meta,records,y=load_inputs(a.data,a.extras);out=Path(a.output)
    if not a.resume:out=new_dir(out,[a.data,a.extras,Path(__file__).parent])
    elif not out.is_dir():raise FileNotFoundError(out)
    plan=make_plan(meta,y,a.split_seed);schema=meta['schema'];dims=meta['dimensions']
    names=CONTROLS+[n for n in a.methods if n!='tabpfn_v2' or not a.without_foundation]
    source_hash={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    signature=dict(version='V17.2',data_manifest_sha256=sha(Path(a.data)/'manifest.json'),
        extras_manifest_sha256=sha(Path(a.extras)/'manifest.json'),plan=plan,seeds=a.seeds,profile=a.profile,
        names=names,grids={n:([{}] if n.startswith('training_') else configs(n,a.profile=='smoke')) for n in names},
        code_sha256=source_hash,foundation='tabpfn_v2' in names,
        epoch_selection='Fixed positive epoch grid, evaluated by full inner folds; constants/zero correction separate; no median epochs',
        fusion='Frozen fixed-alpha Ridge plus cross-fitted residual, shrink in {0,.25}; no learned stacking weights',
        residual_crossfit='Fixed Ridge penalty=1, three disjoint subject-group folds within each fitting subset',
        selection='V17 controls preserve all-target criterion; new candidates use six-primary family-balanced normalized MAE',
        exploratory=True,independent_test=False,labels_are_inputs=False,labels_imputed=False,
        hardware=dict(device=a.device,torch=torch.__version__,python=sys.version,platform=platform.platform()))
    if a.resume:
        prior=read(out/'protocol.json')
        for k in signature:
            if k!='hardware' and prior[k]!=signature[k]:raise ValueError('Resume protocol changed: '+k)
    else:write(out/'protocol.json',signature)
    started=time.perf_counter();completed=[]
    for fold in plan['folds']:
        fi=fold['fold']
        if a.folds is not None and fi not in a.folds:continue
        folder=out/f'fold_{fi}'
        if a.resume and (folder/'complete.json').is_file():completed.append(fi);continue
        if folder.exists():raise FileExistsError('Incomplete fold retained; use a new output: '+str(folder))
        folder.mkdir();tr,te=fold['train'],fold['test'];train=[records[i] for i in tr];test=[records[i] for i in te]
        yy=y[tr];pos={i:j for j,i in enumerate(tr)};inner=[([pos[i] for i in s['train']],[pos[i] for i in s['validation']]) for s in fold['inner']]
        cache=FitCache(dims,schema);decisions={};rows=[];inner_predictions={}
        print('Fold',fi,'train',len(tr),'test',len(te),flush=True)
        for name in names:
            model,ip,decision=tune(name,train,yy,dims,schema,inner,a.seeds,a.device,a.profile=='smoke',cache)
            model.update(training_ids=[r['subject_id'] for r in train],training_groups=sorted({r['group_id'] for r in train}),test_ids=[r['subject_id'] for r in test],name=name,selection=decision)
            p,diagnostic=predict_candidate(model,test,a.device,components=True)
            joblib.dump(model,folder/(name+'.joblib'),compress=3)
            decisions[name]={**decision,'outer_prediction_components':diagnostic,'training_logs':model.get('logs',[])}
            inner_predictions[name]=ip
            # Outer labels are read only for reporting after this candidate's predictions.
            for k,i in enumerate(te):
                e=meta['subjects'][i]
                for j,t in enumerate(schema):
                    if np.isfinite(y[i,j]):rows.append(dict(subject_id=e['subject_id'],group_id=e['group_id'],fold=fi,model=name,
                        target=t['name'],family=t['family'],y_true=float(y[i,j]),prediction=float(p[k,j]),
                        complete_three_modality=all(e['modality_available'].values()),eeg_available=e['modality_available']['eeg']))
            print(' ',name,'selected',decision['config'],flush=True)
        csvwrite(folder/'predictions.csv',rows);write(folder/'decisions.json',decisions)
        np.savez_compressed(folder/'inner_oof.npz',**inner_predictions,train_indices=np.array(tr))
        write(folder/'complete.json',dict(fold=fi,train_ids=[r['subject_id'] for r in train],test_ids=[r['subject_id'] for r in test],
            artifact_sha256={p.name:sha(p) for p in folder.glob('*.joblib')},prediction_sha256=sha(folder/'predictions.csv')))
        completed.append(fi);write(out/'status.json',dict(state='running',completed=completed,seconds=time.perf_counter()-started))
    if source_hash!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}:raise RuntimeError('Code changed during training')
    write(out/'status.json',dict(state='complete',completed=completed,full_fivefold=sorted(completed)==list(range(5)),seconds=time.perf_counter()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--extras',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--split-seed',type=int,default=20261023)
    p.add_argument('--seeds',type=int,nargs='+',default=[20261023,20261024,20261025]);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--folds',type=int,nargs='+');p.add_argument('--profile',choices=['full','smoke'],default='full')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);p.add_argument('--without-foundation',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.seeds or len(set(a.methods))!=len(a.methods) or (a.folds is not None and not set(a.folds)<=set(range(5))):raise ValueError('Invalid seeds/methods/folds')
    torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
