"""Fixed person-level nested evaluation of EEG, eye and asynchronous fusion baselines."""
import argparse
import gc
import importlib.metadata
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from common import load_json, save_json, save_csv, sha256, new_output, manifest, metrics
from benchmark_models import SPECS, DEFAULT_MODELS, candidates, fit_bundle, predict_bundle
from assets import checked_weight, TABPFN_SHA


def validate_plan(plan, manifest_hash):
    def require(test,message):
        if not test: raise ValueError(message)
    require(plan['feature_manifest_sha256']==manifest_hash,'Plan/feature manifest mismatch')
    ids=plan['subject_ids'];all_ids=set(ids);groups=plan['groups'];seen=[]
    require(len(ids)==len(all_ids),'Duplicate subject IDs')
    require(not all_ids & set(plan.get('reserved_ids',[])),'Reserved ID entered training')
    require(set(groups)==all_ids,'Incomplete group mapping')
    require(len({f['fold'] for f in plan['folds']})==len(plan['folds']),'Duplicate outer folds')
    for fold in plan['folds']:
        a,b=set(fold['development']),set(fold['test'])
        require(a and b and a|b==all_ids and not a&b,'Invalid outer partition')
        require(len(a)==len(fold['development']) and len(b)==len(fold['test']),'Duplicate outer rows')
        require(not {groups[s] for s in a}&{groups[s] for s in b},'Outer person leakage')
        inner_seen=[]
        for inner in fold['inner']:
            c,d=set(inner['train']),set(inner['validation'])
            require(c and d and c|d==a and not c&d,'Invalid inner partition')
            require(len(c)==len(inner['train']) and len(d)==len(inner['validation']),'Duplicate inner rows')
            require(not {groups[s] for s in c}&{groups[s] for s in d},'Inner person leakage')
            inner_seen.extend(inner['validation'])
        require(sorted(inner_seen)==sorted(a),'Incomplete inner coverage')
        seen.extend(fold['test'])
    require(sorted(seen)==sorted(ids),'Incomplete outer coverage')


def protected_inputs(root):
    cfg=load_json(root/'config.json')
    paths=[root]
    for key in ['raw_root','processed_root']:
        value=Path(cfg[key])
        if not value.is_absolute():
            raise ValueError(f'{key} is not absolute on this OS; regenerate features on server or relocate the copied config')
        paths.append(value.resolve())
    return paths


def read_blocks(root, ids, people, names, embedding_root=None):
    need={SPECS[n][0] for n in names}; result={}; sources={}
    require_embedding=bool(need & {'cbramod','fusion','fusion_target'})
    encoder=None
    if require_embedding:
        if embedding_root is None: raise ValueError('CBraMod/fusion requires --embeddings')
        encoder=load_json(Path(embedding_root)/'encoder.json')
        if encoder['status']!='complete' or encoder['feature_manifest_sha256']!=sha256(root/'manifest.json'):
            raise ValueError('Incomplete/mismatched encoder features')
    for sid in ids:
        path=root/(sid+'.npz'); digest=sha256(path);sources[sid]=digest
        with np.load(path,allow_pickle=False) as z:
            blocks={k:np.asarray(z[k],float) for k in ['eeg_power','eeg_cov','eye_base','eye_target']}
        if require_embedding:
            p=Path(embedding_root)/(sid+'.npz');record=encoder['sources'].get(sid,{})
            if record.get('input_sha256')!=digest or record.get('output_sha256')!=sha256(p):
                raise ValueError(f'{sid}: encoder source/output hash mismatch')
            with np.load(p,allow_pickle=False) as z: blocks['cbramod']=np.asarray(z['cbramod'],float)
            if blocks['cbramod'].shape!=(400,):raise ValueError('Expected 400D CBraMod features')
            blocks['fusion']=np.r_[blocks['cbramod'],blocks['eye_base']]
            blocks['fusion_target']=np.r_[blocks['cbramod'],blocks['eye_base'],blocks['eye_target']]
        blocks['eye_target']=np.r_[blocks['eye_base'],blocks['eye_target']]
        blocks['demographics']=np.array([people[sid].get('age') or np.nan,
            {'男':1.,'女':0.}.get(people[sid].get('sex'),np.nan)])
        for key in need:
            value=blocks[key]
            if np.isinf(value).any() or (key!='demographics' and not np.isfinite(value).all()):
                raise ValueError(f'{sid}: {key} unavailable; confirm task/coordinates and freeze a common target_eye cohort first')
            result.setdefault(key,[]).append(value)
    return {k:np.stack(v) for k,v in result.items()},sources,encoder


def release(bundle):
    bundle.clear();gc.collect()
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()


def save_bundle(folder,bundle,x,y,metadata):
    folder.mkdir(parents=True,exist_ok=False)
    metadata.update(kind=bundle['kind'],parameter=bundle['parameter'],n_features=bundle['n_features'])
    if bundle['kind']=='tabpfn':
        np.savez_compressed(folder/'context.npz',x=x,y=y)
        metadata.update(context_sha256=sha256(folder/'context.npz'),weight_sha256=TABPFN_SHA,tabpfn_version='2.0.9')
    else:
        joblib.dump(bundle,folder/'model.joblib')
        metadata['model_sha256']=sha256(folder/'model.joblib')
    save_json(folder/'metadata.json',metadata)


def reload_bundle(folder,device='cpu',weights=None,threads=2):
    folder=Path(folder);meta=load_json(folder/'metadata.json')
    if meta['kind']=='tabpfn':
        if sha256(folder/'context.npz')!=meta['context_sha256']:raise ValueError('Saved context changed')
        with np.load(folder/'context.npz',allow_pickle=False) as z:
            return fit_bundle('tabpfn',meta['parameter'],z['x'],z['y'],device,threads,meta['seed'],weights)
    if sha256(folder/'model.joblib')!=meta['model_sha256']:raise ValueError('Saved model changed')
    return joblib.load(folder/'model.joblib')


def summarize(rows,plan_hash,models,complete,exploratory):
    results={}
    for target in sorted({r['target'] for r in rows}):
        results[target]={}
        for name in models:
            rr=[r for r in rows if r['target']==target and r['model']==name]
            if not rr: continue
            truth=np.array([r['y_true'] for r in rr]);pred=np.array([r['prediction'] for r in rr])
            result=metrics(truth,pred)
            result.update(n=len(rr),mean_baseline=metrics(truth,[r['mean_prediction'] for r in rr]),
                median_baseline=metrics(truth,[r['median_prediction'] for r in rr]))
            results[target][name]=result
    return {'results':results,'models':models,'plan_sha256':plan_hash,'complete_outer_cv':complete,
        'exploratory':exploratory,'raw_outputs_not_clipped':True,'parameter_selection':'inner MAE per target; TabPFN configuration fixed',
        'note':'MAE/RMSE/R2 in original score units. These are symptom regressions, not disease classification.'}


def main(args):
    started=time.time();root=Path(args.features).resolve(strict=True)
    plan=load_json(args.plan);validate_plan(plan,sha256(root/'manifest.json'))
    names=args.models.split(',') if args.models else list(DEFAULT_MODELS)
    if len(names)!=len(set(names)) or set(names)-set(SPECS):raise ValueError('Unknown/duplicate model names')
    if args.fold is not None and args.fold not in {f['fold'] for f in plan['folds']}:raise ValueError('Unknown fold')
    if args.threads<1:raise ValueError('threads must be positive')
    if any(SPECS[n][1]=='tabpfn' for n in names):checked_weight('tabpfn',args.tabpfn_weights)
    people=manifest(root);ids=plan['subject_ids'];index={s:i for i,s in enumerate(ids)}
    blocks,hashes,encoder=read_blocks(root,ids,people,names,args.embeddings)
    labels={t:np.array([people[s]['targets'][t] for s in ids],float) for t in plan['targets']}
    if not all(np.isfinite(y).all() for y in labels.values()):raise ValueError('Incomplete targets')
    protected=protected_inputs(root)+[Path(args.plan).resolve().parent]
    if args.embeddings:protected.append(Path(args.embeddings).resolve())
    out=new_output(args.output,*protected)
    software={}
    for pkg in ['numpy','scipy','scikit-learn','joblib','torch','tabpfn']:
        try:software[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:software[pkg]=None
    protocol={'features':str(root),'embeddings':str(args.embeddings),'plan_sha256':sha256(args.plan),
        'models':{n:SPECS[n] for n in names},'grids':{n:candidates(SPECS[n][1]) for n in names},
        'software':software,'python':sys.version,'platform':platform.platform(),'source_features':hashes,
        'encoder':encoder,'seed':args.seed,'input_channels':16,'target_transform':'training mean/std only',
        'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'exploratory':plan['exploratory'],'modalities_synchronized':False,
        'tabpfn':'fixed official v2; local inference; 8 estimators; no hyperparameter selection'}
    save_json(out/'protocol.json',protocol)
    save_json(out/'status.json',{'state':'running','complete_outer_cv':False})
    rows=[];decisions=[]
    try:
        for fold in plan['folds']:
            if args.fold is not None and fold['fold']!=args.fold:continue
            tr=[index[s] for s in fold['development']];te=[index[s] for s in fold['test']]
            for target,y in labels.items():
                for name in names:
                    key,kind=SPECS[name];x=blocks[key];records=[]
                    for param in candidates(kind):
                        losses=[]
                        if kind!='tabpfn':
                            for inner in fold['inner']:
                                a=[index[s] for s in inner['train']];b=[index[s] for s in inner['validation']]
                                bundle=fit_bundle(kind,param,x[a],y[a],args.device,args.threads,args.seed,args.tabpfn_weights)
                                losses.append(float(np.mean(abs(predict_bundle(bundle,x[b])-y[b]))));release(bundle)
                        records.append({'parameter':param,'inner_mae':float(np.mean(losses)) if losses else None,'fold_mae':losses})
                    selected=records[0] if kind=='tabpfn' else min(records,key=lambda r:r['inner_mae'])
                    decision={'fold':fold['fold'],'target':target,'model':name,'selected':selected,'candidates':records,
                        'training_ids':fold['development'],'test_ids':fold['test']}
                    decisions.append(decision);save_json(out/'selections.json',decisions)
                    # Decision persisted before outer evaluation; no outer early stopping or parameter choice.
                    bundle=fit_bundle(kind,selected['parameter'],x[tr],y[tr],args.device,args.threads,args.seed,args.tabpfn_weights)
                    pred=predict_bundle(bundle,x[te]);folder=out/'models'/target/name/f"fold_{fold['fold']}"
                    save_bundle(folder,bundle,x[tr],y[tr],{'target':target,'name':name,'feature_block':key,
                        'fold':fold['fold'],'seed':args.seed,'training_ids':fold['development'],
                        'training_groups':sorted({plan['groups'][s] for s in fold['development']}),
                        'test_ids':fold['test'],'plan_sha256':sha256(args.plan)})
                    np.savez_compressed(folder/'replay.npz',x=x[te],prediction=pred)
                    release(bundle)
                    for i,sid in enumerate(fold['test']):
                        rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'fold':fold['fold'],'target':target,
                            'model':name,'y_true':float(y[te[i]]),'prediction':float(pred[i]),
                            'mean_prediction':float(y[tr].mean()),'median_prediction':float(np.median(y[tr]))})
                    save_csv(out/'predictions.csv',rows)
                    save_json(out/'summary.json',summarize(rows,sha256(args.plan),names,False,plan['exploratory']))
                    print(target,name,'fold',fold['fold'],'MAE',round(float(np.mean(abs(pred-y[te]))),4),flush=True)
        if any(sha256(root/(s+'.npz'))!=h for s,h in hashes.items()):raise RuntimeError('Input feature files changed')
        save_json(out/'summary.json',summarize(rows,sha256(args.plan),names,args.fold is None,plan['exploratory']))
        save_json(out/'status.json',{'state':'complete','complete_outer_cv':args.fold is None,
            'input_features_unchanged':True,'elapsed_seconds':time.time()-started})
    except Exception as exc:
        save_json(out/'status.json',{'state':'failed','complete_outer_cv':False,'error':str(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--embeddings');p.add_argument('--output',required=True)
    p.add_argument('--models',help=','.join(SPECS));p.add_argument('--fold',type=int)
    p.add_argument('--device',default='cpu');p.add_argument('--tabpfn-weights')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=20261015)
    args=p.parse_args()
    with threadpool_limits(limits=args.threads):main(args)
