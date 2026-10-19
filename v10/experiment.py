"""V9 development experiments. Full nested person-level selection; no reserve fitting."""
import argparse
import collections
import json
import platform
import sys
from pathlib import Path
import joblib
import numpy as np
import legacy_project as lp
from common import load_json,save_json,save_csv,sha256,new_output,manifest,metrics
from v9_models import KernelContext,Head,catalog,choose,FAMILIES
from attention_model import AttentionRegressor

ALIASES={'selected_eeg':['eeg_sparse','eeg_kernel'],
    'selected_eye':['eye_sparse','eye_kernel','eye_raw_kernel'],
    'selected_fusion':['fusion_additive','fusion_interaction','fusion_gated','fusion_multitask','fusion_attention']}

def load_features(folder,ids):
    info=load_json(folder/'features.json')
    if info['state']!='complete' or set(info['ids'])!=set(lp.lock()['development_ids']):raise ValueError('Incomplete/mismatched V9 features')
    if info['feature_code_sha256']!=sha256(Path(__file__).parent/'feature_engineering.py'):raise ValueError('Feature code changed')
    result=[]
    for sid in ids:
        p=folder/(sid+'.npz')
        if sha256(p)!=info['sources'][sid]['output_sha256']:raise ValueError('V9 feature hash changed: '+sid)
        if info['sources'][sid]['input_sha256']!=lp.lock()['feature_hashes'][sid]:raise ValueError('V9 input lineage changed')
        with np.load(p,allow_pickle=False) as z:result.append({k:z[k].copy() for k in z.files})
    return result,info

def inner_search(records,y,fold,ids,targets,device,threads,include_attention=True):
    index={s:i for i,s in enumerate(ids)}
    specs=[s for s in catalog() if include_attention or s['family']!='fusion_attention']
    losses={t:{s['id']:{'spec':s,'fold_mae':[]} for s in specs} for t in targets}
    neural_spec=next((s for s in specs if s['family']=='fusion_attention'),None)
    for inner in fold['inner']:
        a=[index[s] for s in inner['train']];b=[index[s] for s in inner['validation']]
        tr=[records[i] for i in a];va=[records[i] for i in b]
        if neural_spec:
            net=AttentionRegressor().fit(tr,y[a],neural_spec,device,threads)
            neural=net.predict(va)
        for j,target in enumerate(targets):
            context=KernelContext().fit(tr,y[a],j);query=context.prepare_query(va)
            for spec in specs:
                prediction=neural[:,j] if spec['family']=='fusion_attention' else Head().fit(context,spec).predict_prepared(query)[:,j]
                losses[target][spec['id']]['fold_mae'].append(float(np.mean(abs(prediction-y[b,j]))))
    decisions={}
    for target in targets:
        available=[name for name in FAMILIES if include_attention or name!='fusion_attention']
        choices={name:choose([r for r in losses[target].values() if r['spec']['family']==name]) for name in available}
        for alias,families in ALIASES.items():
            eligible=[choices[f] for f in families if f in choices]
            choices[alias]=dict(min(eligible,key=lambda d:(d['inner_mae'],d['spec']['id'])))
            choices[alias]['rule']='Select family using INNER predictions only; never outer scores'
        decisions[target]=choices
    return decisions,losses

def fit_selected(records,y,targets,decisions,device,threads):
    spec=next(s for s in catalog() if s['family']=='fusion_attention')
    net=AttentionRegressor().fit(records,y,spec,device,threads)
    models={}
    for j,target in enumerate(targets):
        c=KernelContext().fit(records,y,j);models[target]={};cache={}
        for name,chosen in decisions[target].items():
            s=chosen['spec']
            if s['id'] not in cache:
                cache[s['id']]=net if s['family']=='fusion_attention' else Head().fit(c,s)
            models[target][name]=cache[s['id']]
    return models

def predictions(models,records,targets):
    result={};cache={}
    for j,target in enumerate(targets):
        result[target]={}
        for name,model in models[target].items():
            key=id(model)
            if key not in cache:cache[key]=model.predict(records)
            result[target][name]=cache[key][:,j]
    return result

def summary_rows(rows):
    result={}
    for target in sorted({r['target'] for r in rows}):
        result[target]={}
        for name in sorted({r['model'] for r in rows if r['target']==target}):
            rr=[r for r in rows if r['target']==target and r['model']==name]
            m=metrics([r['y_true'] for r in rr],[r['prediction'] for r in rr]);m['n']=len(rr)
            result[target][name]=m
    return result

def consent_protocol():
    return {'date':'2026-10-16','version':'V9','exploratory':True,
        'authorization':'User explicitly requested broad model innovation; supersedes V8 no-new-variants only for this separate V9 project.',
        'primary_family':'fusion_multitask','catalog':catalog(),'adaptive_families':ALIASES,
        'models_predeclared_before_outer_evaluation':True,
        'feature_screening':'Robust scale, clipping, top-k supervised correlations fitted on INNER train; refit on outer development only.',
        'neural':'70 fixed epochs, two fixed seeds, hidden width 8; no outer-label early stopping; person-equal Huber loss.',
        'kernel':'Fixed Fourier distribution map; train-only scalers and bandwidth; additive/product/quality-proxy gated PSD kernel.',
        'target_coupling':'Train-target covariance shrunk 0.75 toward multitask output kernel; no questionnaire supplied as prediction input.',
        'gates':'Within-record split agreement and repeat support; engineering quality proxies, NOT validated clinical reliability.',
        'evaluation':'Same historical folds; development comparison after exploratory feature audit, not independent final test.',
        'reserved_subjects_used_for_fit':0,'scores_fabricated':0,'human_confirmations_fabricated':0,
        'final_parameter_rule':'Deterministic modal INNER-selected configuration over outer folds; fit all development people.',
        'python':sys.version,'platform':platform.platform()}

def main(args):
    original=Path(args.features).resolve(strict=True);_,plans=lp.check_dataset(original)
    folder=Path(args.v9_features).resolve(strict=True);people=manifest(original)
    out=new_output(args.output,*lp.protected_inputs(original),folder,Path(__file__).parent)
    protocol=consent_protocol();protocol['code_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    protocol['feature_info_sha256']=sha256(folder/'features.json');save_json(out/'protocol.json',protocol)
    save_json(out/'status.json',{'state':'running','exploratory':True})
    rows=[];decisions_all=[];outer_verified=0;final_verified=0
    try:
        for task,plan in enumerate(plans):
            ids=plan['subject_ids'];targets=plan['targets'];index={s:i for i,s in enumerate(ids)}
            records,_=load_features(folder,ids)
            y=np.stack([lp.labels_for(people,ids,t) for t in targets],axis=1)
            task_decisions=[]
            for fold in plan['folds']:
                print('Selecting:',task,'fold',fold['fold'],flush=True)
                choices,search=inner_search(records,y,fold,ids,targets,args.device,args.threads)
                decision={'task':task,'fold':fold['fold'],'choices':choices,'search':search}
                decisions_all.append(decision);task_decisions.append(decision)
                # All choices persisted before any outer target score is accessed.
                save_json(out/'selections.json',decisions_all)
                a=[index[s] for s in fold['development']];b=[index[s] for s in fold['test']]
                tr=[records[i] for i in a];te=[records[i] for i in b]
                models=fit_selected(tr,y[a],targets,choices,args.device,args.threads)
                pred=predictions(models,te,targets)
                artifact={'targets':targets,'models':models,'training_ids':fold['development'],
                    'test_ids':fold['test'],'choices':choices,'exploratory':True,'task':task,'fold':fold['fold']}
                dest=out/'outer'/f'task_{task}_fold_{fold["fold"]}.joblib';dest.parent.mkdir(exist_ok=True)
                joblib.dump(artifact,dest,compress=3)
                replay=predictions(joblib.load(dest)['models'],te,targets)
                for j,target in enumerate(targets):
                    for name,values in pred[target].items():
                        np.testing.assert_allclose(values,replay[target][name],rtol=0,atol=1e-6)
                        outer_verified+=1
                        for i,sid in enumerate(fold['test']):
                            rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'task':task,
                                'fold':fold['fold'],'target':target,'model':name,'y_true':float(y[b[i],j]),
                                'prediction':float(values[i]),'effective_spec':choices[target][name]['spec']['id']})
                    for baseline in ['mean','median']:
                        value=float(np.mean(y[a,j]) if baseline=='mean' else np.median(y[a,j]))
                        for i,sid in enumerate(fold['test']):
                            rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'task':task,
                                'fold':fold['fold'],'target':target,'model':baseline,'y_true':float(y[b[i],j]),
                                'prediction':value,'effective_spec':baseline})
                save_csv(out/'predictions.csv',rows);save_json(out/'summary.json',{'results':summary_rows(rows),'complete':False,'exploratory':True})
                print('Verified outer fold:',task,fold['fold'],flush=True)
            # Consensus hyperparameters use inner decisions only, never an outer-MAE winner.
            consensus={}
            specs={s['id']:s for s in catalog()}
            for target in targets:
                consensus[target]={}
                for name in task_decisions[0]['choices'][target]:
                    votes=collections.Counter(d['choices'][target][name]['spec']['id'] for d in task_decisions)
                    winner=min(votes,key=lambda key:(-votes[key],key))
                    consensus[target][name]={'spec':specs[winner],'votes':dict(votes),
                        'rule':'Mode of five INNER-selected specs; tie by fixed spec ID'}
            save_json(out/f'final/task_{task}_choices.json',consensus)
            fitted=fit_selected(records,y,targets,consensus,args.device,args.threads)
            artifact={'targets':targets,'models':fitted,'training_ids':ids,'test_ids':[],
                'choices':consensus,'exploratory':True,'task':task,
                'feature_info_sha256':sha256(folder/'features.json'),'protocol_sha256':sha256(out/'protocol.json')}
            dest=out/f'final/task_{task}.joblib';joblib.dump(artifact,dest,compress=3)
            replay=predictions(joblib.load(dest)['models'],records[:2],targets);pred=predictions(fitted,records[:2],targets)
            for target in targets:
                for name in pred[target]:
                    np.testing.assert_allclose(pred[target][name],replay[target][name],rtol=0,atol=1e-6);final_verified+=1
            save_json(out/f'final/task_{task}_integrity.json',{'model_sha256':sha256(dest),
                'training_ids':ids,'targets':targets,'choices_sha256':sha256(out/f'final/task_{task}_choices.json')})
        results=summary_rows(rows)
        save_json(out/'summary.json',{'results':results,'complete':True,'exploratory':True,'primary_family':'fusion_multitask'})
        save_csv(out/'metrics.csv',[{'target':t,'model':name,**values} for t,models in results.items() for name,values in models.items()])
        lp.check_dataset(original);load_features(folder,lp.lock()['development_ids'])
        if protocol['code_sha256']!={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}:
            raise RuntimeError('Project code changed during experiment')
        save_json(out/'status.json',{'state':'complete','exploratory':True,'outer_model_predictions_replayed':outer_verified,
            'final_model_predictions_replayed':final_verified,'reserved_subjects_used_for_fit':0,'original_feature_hashes_unchanged':True})
        print('Complete:',out,flush=True)
    except Exception as exc:
        save_json(out/'status.json',{'state':'failed','error':str(exc),'exploratory':True});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--v9-features',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.threads<1:p.error('threads must be positive')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):main(args)
