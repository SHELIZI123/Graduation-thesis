"""Nested, subject-grouped comparisons. Outer labels are used only after predictions."""
from pathlib import Path
import argparse,time,sys,platform
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from common import load_data,new_dir,write,csvwrite,sha,MODALITIES
from train import make_plan,BASELINES
from preprocessing import target_stats,normalized_mae
from engine import CONFIGS,RIDGE_LAMBDAS,fit_ridge,fit_neural,predict,bound
from features_v171 import FOCUS,focus_mae
from engine_v171 import CONFIGS_NEW,RECIPES,classical_configs,fit_classical,predict_classical,fit_new,predict_new,convex_weights,combine

def predict_artifact(a,records,device='cpu'):
    if a['kind']=='new_neural':return predict_new(a,records,device)
    if a['kind'] in ['new_classical','foundation_context']:return predict_classical(a,records,device)
    if a['kind']=='bank':
        pp={key:predict_artifact(m,records,device) for key,m in a['models'].items()}
        return np.stack([pp[str(k)][:,j] for j,k in enumerate(a['selection'])],1)
    if a['kind']=='stack':
        pp={key:predict_artifact(m,records,device) for key,m in a['models'].items()}
        return combine(pp,a['names'],a['weights'])
    if a['kind']=='constant':return np.broadcast_to(a['values'],(len(records),len(a['values']))).copy()
    return predict(a,records,device)

def target_choice(y,pp):
    errors=[]
    for p in pp:
        col=[]
        for j in range(y.shape[1]):
            ok=np.isfinite(y[:,j]);col.append(float(np.mean(abs(y[ok,j]-p[ok,j]))) if ok.any() else 0.)
        errors.append(col)
    return np.argmin(np.asarray(errors),axis=0).tolist(),errors

def main(a):
    meta,records,y=load_data(a.data);out=Path(a.output)
    if not a.resume:out=new_dir(out,[a.data,Path(__file__).parent])
    elif not out.is_dir():raise FileNotFoundError(out)
    plan=make_plan(meta,y,a.split_seed);schema=meta['schema'];dims=meta['dimensions']
    source_hash={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    signature=dict(version='V17.1',data_manifest_sha256=sha(Path(a.data)/'manifest.json'),
        plan=plan,seeds=a.seeds,maximum_epochs=a.epochs,patience=a.patience,cap=a.cap,
        recipes=a.recipes,profile=a.profile,foundation=a.foundation,focus_targets=list(FOCUS),code_sha256=source_hash,
        exploratory=True,independent_test=False,labels_are_inputs=False,labels_imputed=False,
        split_seed_is_distinct_from_initialization_seeds=True,
        stacking='Inner OOF L1 simplex weights; fixed .02 deviation penalty to V17 Ridge. Outer labels excluded.',
        selection='V17 controls retain original criterion; new neural configs use family-balanced six-primary-target normalized MAE.',
        heart='Separate PPG and NNI feature views. Native units only; no invented sampling rate or visit alignment.',
        hardware=dict(torch=torch.__version__,python=sys.version,device=a.device,platform=platform.platform()))
    if a.resume:
        from common import read
        prior=read(out/'protocol.json')
        for key in ['version','data_manifest_sha256','plan','seeds','maximum_epochs','patience','cap','recipes','profile','foundation','code_sha256']:
            if prior[key]!=signature[key]:raise ValueError('Resume protocol changed: '+key)
    else:write(out/'protocol.json',signature)
    completed=[];started=time.perf_counter()
    folds=[f for f in plan['folds'] if a.folds is None or f['fold'] in a.folds]
    for fold in folds:
        fi=fold['fold'];folder=out/f'fold_{fi}'
        if a.resume and (folder/'complete.json').is_file():completed.append(fi);continue
        if folder.exists():raise FileExistsError('Incomplete fold exists. Use a new output or retain/remove that fold manually: '+str(folder))
        folder.mkdir();tr,te=fold['train'],fold['test'];train=[records[i] for i in tr];test=[records[i] for i in te]
        yy=y[tr];mu,sd=target_stats(yy);pos={i:k for k,i in enumerate(tr)}
        inner_pred={};outer_pred={};artifacts={};decisions={};train_ids=[plan['subject_ids'][i] for i in tr]
        print(f'Fold {fi}: {len(tr)} train, {len(te)} test',flush=True)
        def save(name,artifact,p,ip,decision):
            artifact.update(training_ids=train_ids,test_ids=[plan['subject_ids'][i] for i in te])
            artifacts[name]=artifact;outer_pred[name]=p;inner_pred[name]=ip;decisions[name]=decision
            joblib.dump(artifact,folder/(name+'.joblib'),compress=3)
        for name,fn in [('training_mean',np.mean),('training_median',np.median)]:
            center=np.array([fn(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in yy.T])
            ip=np.zeros_like(yy)
            for inner in fold['inner']:
                it,iv=inner['train'],inner['validation']
                med=np.array([fn(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in y[it].T])
                ip[[pos[i] for i in iv]]=bound(np.broadcast_to(med,(len(iv),len(schema))),schema)
            art=dict(kind='constant',values=bound(center[None],schema)[0],schema=schema)
            save(name,art,predict_artifact(art,test),ip,{})
        # Exact V17 ridge controls, including identical original hyperparameter criterion.
        penalties=RIDGE_LAMBDAS[:1] if a.profile=='smoke' else RIDGE_LAMBDAS
        for name,modalities in BASELINES.items():
            bank=[];scores=[]
            for penalty in penalties:
                ip=np.zeros_like(yy);ss=[]
                for inner in fold['inner']:
                    it,iv=inner['train'],inner['validation'];ir=[records[i] for i in it];vr=[records[i] for i in iv]
                    model=fit_ridge(ir,y[it],dims,schema,penalty,modalities,name=='ridge_mask_only')
                    p=predict(model,vr);ip[[pos[i] for i in iv]]=p
                    ss.append(normalized_mae(y[iv],p,target_stats(y[it])[1],schema))
                bank.append(ip);scores.append(float(np.mean(ss)))
            chosen=min(range(len(penalties)),key=lambda k:(scores[k],-penalties[k]))
            art=fit_ridge(train,yy,dims,schema,penalties[chosen],modalities,name=='ridge_mask_only')
            save(name,art,predict(art,test),bank[chosen],dict(penalty=penalties[chosen],inner_scores=scores))
        for name,configs in classical_configs(a.foundation).items():
            if a.profile=='smoke':configs=configs[:1]
            pp=[]
            for config in configs:
                ip=np.zeros_like(yy)
                for inner in fold['inner']:
                    it,iv=inner['train'],inner['validation']
                    model=fit_classical([records[i] for i in it],y[it],dims,schema,config,device=a.device)
                    ip[[pos[i] for i in iv]]=predict_classical(model,[records[i] for i in iv])
                pp.append(ip)
            choice,errors=target_choice(yy,pp)
            models={str(k):fit_classical(train,yy,dims,schema,configs[k],device=a.device) for k in sorted(set(choice))}
            art=dict(kind='bank',models=models,selection=choice,schema=schema)
            ip=np.stack([pp[k][:,j] for j,k in enumerate(choice)],1)
            save(name,art,predict_artifact(art,test,a.device),ip,dict(configurations=configs,selected_per_target=choice,inner_mae=errors))
            print(' ',name,'done',flush=True)
        for name in a.recipes:
            legacy=name=='v17_ple';configs=CONFIGS if legacy else CONFIGS_NEW
            if a.profile=='smoke':configs=configs[:1]
            searches=[];pp=[]
            for config in configs:
                ip=np.zeros_like(yy);scores=[];epochs=[]
                for inner in fold['inner']:
                    it,iv=inner['train'],inner['validation'];ir=[records[i] for i in it];vr=[records[i] for i in iv]
                    if legacy:
                        model,p=fit_neural(ir,y[it],dims,schema,config,'ple',a.seeds,a.epochs,a.device,(vr,y[iv]),a.patience,32)
                        scores.append(normalized_mae(y[iv],p,target_stats(y[it])[1],schema))
                    else:
                        model,p=fit_new(ir,y[it],dims,schema,config,name,a.seeds,a.epochs,a.device,(vr,y[iv]),a.patience,a.cap)
                        scores.append(focus_mae(y[iv],p,target_stats(y[it])[1],schema))
                    epochs.extend(log['epochs'] for log in model['logs']);ip[[pos[i] for i in iv]]=p
                pp.append(ip);searches.append(dict(config=config,inner_score=float(np.mean(scores)),
                                                   inner_best_epochs=epochs,refit_epochs=int(np.median(epochs))))
            k=min(range(len(configs)),key=lambda k:(searches[k]['inner_score'],configs[k]['hidden']))
            selected=searches[k]
            if legacy:art,_=fit_neural(train,yy,dims,schema,configs[k],'ple',a.seeds,selected['refit_epochs'],a.device,cap=32)
            else:art,_=fit_new(train,yy,dims,schema,configs[k],name,a.seeds,selected['refit_epochs'],a.device,cap=a.cap)
            save(name,art,predict_artifact(art,test,a.device),pp[k],dict(selected=k,search=searches))
            print(' ',name,'inner',round(selected['inner_score'],4),'epochs',selected['refit_epochs'],flush=True)
        # Exclude acquisition-mask-only and single-modality controls from the stack library.
        names=['training_median','ridge_trimodal',*classical_configs(a.foundation).keys(),*a.recipes]
        bank={n:inner_pred[n] for n in names}
        names,weights=convex_weights(yy,bank,sd,'ridge_trimodal')
        stack=dict(kind='stack',names=names,weights=weights,models={n:artifacts[n] for n in names},schema=schema)
        save('stack_l1',stack,combine(outer_pred,names,weights),combine(inner_pred,names,weights),
             dict(names=names,weights=weights.tolist(),penalty=.02,outer_labels_used=False))
        # A discrete selector offers a transparent comparison to continuous stacking.
        choice,errors=target_choice(yy,[inner_pred[n] for n in names])
        art=dict(kind='bank',models={str(k):artifacts[names[k]] for k in sorted(set(choice))},selection=choice,schema=schema)
        save('target_selector',art,predict_artifact(art,test,a.device),
             np.stack([inner_pred[names[k]][:,j] for j,k in enumerate(choice)],1),
             dict(names=names,selected_per_target=choice,inner_mae=errors,outer_labels_used=False))
        # Only now read outer target values to construct evaluation records.
        rows=[]
        for name,p in outer_pred.items():
            if not np.isfinite(p).all():raise FloatingPointError(name)
            for ri,i in enumerate(te):
                entry=meta['subjects'][i]
                for j,t in enumerate(schema):
                    if np.isfinite(y[i,j]):
                        rows.append(dict(subject_id=entry['subject_id'],group_id=entry['group_id'],fold=fi,
                          model=name,target=t['name'],family=t['family'],y_true=float(y[i,j]),prediction=float(p[ri,j]),
                          complete_three_modality=all(entry['modality_available'].values())))
        csvwrite(folder/'predictions.csv',rows);write(folder/'decisions.json',decisions)
        # Inner arrays support auditing the weight selection, but are NOT unbiased scores.
        np.savez_compressed(folder/'inner_oof.npz',**inner_pred,train_indices=np.array(tr))
        write(folder/'complete.json',dict(fold=fi,train_ids=train_ids,test_ids=[plan['subject_ids'][i] for i in te],
            artifact_sha256={p.name:sha(p) for p in folder.glob('*.joblib')},prediction_sha256=sha(folder/'predictions.csv')))
        completed.append(fi);write(out/'status.json',dict(state='running',completed=completed,seconds=time.perf_counter()-started))
    if source_hash!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}:raise ValueError('Code changed during training')
    write(out/'status.json',dict(state='complete',completed=sorted(completed),full_fivefold=sorted(completed)==list(range(5)),
                                seconds=time.perf_counter()-started,profile=a.profile))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=240);p.add_argument('--patience',type=int,default=35)
    p.add_argument('--split-seed',type=int,default=20261023);p.add_argument('--seeds',nargs='+',type=int,default=[20261023,20261024,20261025])
    p.add_argument('--folds',nargs='+',type=int);p.add_argument('--cap',type=int,default=128);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--recipes',nargs='+',choices=[*RECIPES,'v17_ple'],default=[*RECIPES,'v17_ple'])
    p.add_argument('--profile',choices=['full','smoke'],default='full');p.add_argument('--resume',action='store_true')
    p.add_argument('--foundation',action='store_true',help='Official TabPFN-v2 primary-target candidate; needs optional dependencies and checkpoint')
    a=p.parse_args()
    if a.epochs<1 or not a.seeds or a.cap<1:raise ValueError('Positive epochs/cap and at least one seed required')
    torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):main(a)
