"""Locked nested person-level V10 evaluation; no outer target used to choose weights."""
import argparse, collections, sys
from pathlib import Path
import joblib
import numpy as np
import legacy_project as lp
from common import load_json, save_json, save_csv, sha256, new_output, manifest
from experiment import load_features, summary_rows
from literature_models import Context, Regressor, StackedRegressor, catalog, BASE_IDS, fit_weights
from v9_models import choose


def inner(records,y,fold,ids):
    ix={s:i for i,s in enumerate(ids)};lookup={s['id']:s for s in catalog()}
    losses={s['id']:{'spec':s,'fold_mae':[]} for s in catalog()};cross=np.full((len(ids),1+len(BASE_IDS)),np.nan)
    for split in fold['inner']:
        a=[ix[s] for s in split['train']];b=[ix[s] for s in split['validation']]
        c=Context().fit([records[i] for i in a],y[a]);q=c.prepare([records[i] for i in b]);cross[b,0]=c.mean
        for spec in catalog():
            p=Regressor().fit(c,spec).predict_prepared(q)
            losses[spec['id']]['fold_mae'].append(float(np.mean(abs(p-y[b]))))
            if spec['id'] in BASE_IDS:cross[b,1+BASE_IDS.index(spec['id'])]=p
    a=[ix[s] for s in fold['development']]
    if not np.isfinite(cross[a]).all():raise ValueError('Incomplete inner cross-fitting')
    decisions={f:choose([v for v in losses.values() if v['spec']['family']==f]) for f in sorted({s['family'] for s in catalog()})}
    weights=fit_weights(cross[a],y[a])
    return decisions,weights,losses,cross[a],a


def predict_models(models, records):
    c=next(iter(models.values())).context;q=c.prepare(records)
    return {name:m.predict_prepared(q) for name,m in models.items()}


def main(args):
    root=Path(args.features).resolve(strict=True);folder=Path(args.v9_features).resolve(strict=True)
    _,plans=lp.check_dataset(root);people=manifest(root)
    out=new_output(args.output,*lp.protected_inputs(root),folder,Path(__file__).parent)
    code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    protocol={'version':'V10-literature','exploratory':True,'primary_model':'fusion_stack',
      'predeclared_catalog':catalog(),'stack_base_ids':['fold_training_mean']+BASE_IDS,
      'stack_penalty':.02,'selection':'3 inner folds, one-SE heuristic within each family; convex weights fitted on inner OOF predictions of fixed bases only',
      'set_kernel':'All-window empirical RBF mean embedding; Gaussian outer kernel on biased MMD; train-only person-balanced scaling and bandwidth; train-only kernel centering',
      'cpm':'195 SENSOR features, six blocks; 12 sign-separated summary features; FDR q<0.01 control and top16 predictive adaptation',
      'limitations':'Historical development splits reused, all clinical/protocol confirmations pending; not independent final testing; stack penalty is an adaptation with no oracle guarantee here',
      'code_sha256':code,'features_sha256':sha256(folder/'features.json'),'plan_sha256':[sha256(Path(__file__).parent/f'config/task_{i}.json') for i in range(2)],
      'reserved_subjects_used_for_fit':0,'fabricated_scores':0}
    save_json(out/'protocol.json',protocol);save_json(out/'status.json',{'state':'running'})
    rows=[];selections=[];replayed=0
    try:
      for task,plan in enumerate(plans):
        ids=plan['subject_ids'];index={s:i for i,s in enumerate(ids)};records,_=load_features(folder,ids)
        final_models={};final_weights={}
        for target in plan['targets']:
          y=lp.labels_for(people,ids,target);choices_by_fold=[];cross_all=np.full((len(ids),len(BASE_IDS)+1),np.nan)
          for fold in plan['folds']:
            print('Selecting',task,target,fold['fold'],flush=True)
            decisions,w,search,cross,a=inner(records,y,fold,ids);choices_by_fold.append(decisions)
            selections.append({'task':task,'target':target,'fold':fold['fold'],'choices':decisions,
                               'weights':w.tolist(),'weight_order':['mean']+BASE_IDS,'search':search})
            save_json(out/'selections.json',selections) # persisted before test scores
            b=[index[s] for s in fold['test']];tr=[records[i] for i in a];te=[records[i] for i in b]
            c=Context().fit(tr,y[a]);models={f:Regressor().fit(c,d['spec']) for f,d in decisions.items()}
            models['fusion_stack']=StackedRegressor().fit(c,w)
            pred=predict_models(models,te)
            q=c.prepare(te);lookup={s['id']:s for s in catalog()}
            cross_all[b,0]=c.mean
            for col,key in enumerate(BASE_IDS,1):cross_all[b,col]=Regressor().fit(c,lookup[key]).predict_prepared(q)
            dest=out/f'outer/{target}_fold_{fold["fold"]}.joblib';dest.parent.mkdir(exist_ok=True)
            artifact={'models':models,'training_ids':fold['development'],'test_ids':fold['test'],'target':target,
                      'task':task,'fold':fold['fold'],'exploratory':True,'protocol_sha256':sha256(out/'protocol.json')}
            joblib.dump(artifact,dest,compress=3);replay=predict_models(joblib.load(dest)['models'],te)
            for name,values in pred.items():
              np.testing.assert_allclose(values,replay[name],atol=1e-9,rtol=0);replayed+=1
              for sid,i,value in zip(fold['test'],b,values):
                rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'task':task,'fold':fold['fold'],
                             'target':target,'model':name,'y_true':float(y[i]),'prediction':float(value)})
            for name,value in [('mean',float(y[a].mean())),('median',float(np.median(y[a])))]:
              for sid,i in zip(fold['test'],b):rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'task':task,'fold':fold['fold'],
                             'target':target,'model':name,'y_true':float(y[i]),'prediction':value})
            save_csv(out/'predictions.csv',rows);print('Verified',target,fold['fold'],flush=True)
          if not np.isfinite(cross_all).all():raise ValueError('Missing development OOF bases')
          # Final weights may use development OOF truth, NEVER retroactively rescore them.
          # Deployed stack differs by weights from the 5 evaluated outer stacks.
          weights=fit_weights(cross_all,y);c=Context().fit(records,y);models={}
          for family in choices_by_fold[0]:
            votes=collections.Counter(d[family]['spec']['id'] for d in choices_by_fold)
            winner=min(votes,key=lambda k:(-votes[k],k));spec=next(s for s in catalog() if s['id']==winner)
            models[family]=Regressor().fit(c,spec)
          models['fusion_stack']=StackedRegressor().fit(c,weights)
          dest=out/f'final/{target}.joblib';dest.parent.mkdir(exist_ok=True)
          joblib.dump({'models':models,'target':target,'training_ids':ids,'exploratory':True,
              'protocol_sha256':sha256(out/'protocol.json'),'feature_sha256':sha256(folder/'features.json')},dest,compress=3)
          save_json(dest.with_suffix('.json'),{'model_sha256':sha256(dest),'weights':weights.tolist(),'weight_order':['mean']+BASE_IDS,
            'final_weight_rule':'Fit weights on fixed-base outer cross-fitted development predictions; no performance claim for refit model'})
        
      save_json(out/'summary.json',{'complete':True,'exploratory':True,'primary_model':'fusion_stack','results':summary_rows(rows)})
      lp.check_dataset(root);load_features(folder,lp.lock()['development_ids'])
      if code!={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}:raise RuntimeError('Code changed during run')
      save_json(out/'status.json',{'state':'complete','exploratory':True,'replayed':replayed,'reserved_subjects_used_for_fit':0})
      print('Complete',out,flush=True)
    except Exception as e:
      save_json(out/'status.json',{'state':'failed','error':str(e)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--v9-features',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=2)
    args=p.parse_args()
    if args.threads<1:p.error('threads must be positive')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):main(args)
