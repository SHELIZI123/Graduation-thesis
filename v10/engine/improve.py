"""V8 nested regression: regional PCA, prediction fusion and explicit fallback."""
import argparse
import time
import platform
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from common import load_json,save_json,save_csv,sha256,new_output,manifest
from benchmark import validate_plan,protected_inputs,save_bundle,summarize
from benchmark_models import predict_bundle
from improved_models import MODELS,BLOCKS,REGIONS,compact_power,candidate_catalog,inner_records,choose_all,fit_v8_bundle


def read_inputs(root,embedding_root,ids,people):
    encoder=load_json(embedding_root/'encoder.json')
    if encoder['status']!='complete' or encoder['feature_manifest_sha256']!=sha256(root/'manifest.json'):
        raise ValueError('Incomplete or mismatched regional extraction')
    if encoder.get('regions')!=REGIONS or encoder.get('regional_dimension')!=2000:
        raise ValueError('Regional definition differs')
    rows=[];hashes={}
    for sid in ids:
        p=root/(sid+'.npz');ep=embedding_root/(sid+'.npz');h=sha256(p);record=encoder['sources'].get(sid,{})
        if record.get('input_sha256')!=h or record.get('output_sha256')!=sha256(ep):
            raise ValueError('Feature/embedding hash mismatch: '+sid)
        with np.load(p,allow_pickle=False) as z:
            power=compact_power(z['eeg_power']);eye=np.asarray(z['eye_base'],float)
        with np.load(ep,allow_pickle=False) as z:regional=np.asarray(z['regional'],float)
        if eye.shape!=(20,) or regional.shape!=(2000,) or not np.isfinite(np.r_[power,eye,regional]).all():
            raise ValueError('Missing physiological feature block: '+sid)
        demo=[people[sid].get('age') or np.nan,{'男':1.,'女':0.}.get(people[sid].get('sex'),np.nan)]
        rows.append(np.r_[power,eye,regional,demo]);hashes[sid]={'feature':h,'embedding':sha256(ep)}
    return np.stack(rows),hashes,encoder


def main(args):
    started=time.time();root=Path(args.features).resolve(strict=True);eroot=Path(args.embeddings).resolve(strict=True)
    plan=load_json(args.plan);plan_hash=sha256(args.plan);validate_plan(plan,sha256(root/'manifest.json'))
    if plan['exploratory'] and not args.allow_unreviewed:raise ValueError('Exploratory plan requires --allow-unreviewed')
    if args.fold is not None and args.fold not in {f['fold'] for f in plan['folds']}:raise ValueError('Unknown fold')
    people=manifest(root);ids=plan['subject_ids'];index={s:i for i,s in enumerate(ids)}
    x,hashes,encoder=read_inputs(root,eroot,ids,people)
    labels={t:np.array([people[s]['targets'][t] for s in ids],float) for t in plan['targets']}
    if not all(np.isfinite(y).all() for y in labels.values()):raise ValueError('Missing labels')
    out=new_output(args.output,*protected_inputs(root),eroot,Path(args.plan).resolve().parent)
    protocol={'features':str(root),'embeddings':str(eroot),'plan_sha256':plan_hash,'seed':20261015,
        'exploratory':plan['exploratory'],'models':MODELS,'blocks':BLOCKS,'regions':REGIONS,
        'catalog':candidate_catalog(),'sources':hashes,'encoder':encoder,'selection':'inner raw-unit fold-mean MAE; one-SE heuristic except late_fusion',
        'transforms':'train-only median imputation, z score, PCA whitening, target mean/SD',
        'fallback':'mean/median candidates evaluated inside inner folds; no outer-label selection',
        'fusion':'joint fixed-grid convex mixing of inner held-out personal predictions; no synchronized windows',
        'covariates':'age/sex used only by demographics_control; excluded from five physiological candidates',
        'test_reuse':'V8 developed after V7 outer scores were inspected; development CV, not an untouched final test',
        'python':sys.version,'platform':platform.platform(),
        'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    save_json(out/'protocol.json',protocol);save_json(out/'status.json',{'state':'running','complete_outer_cv':False})
    rows=[];decisions=[]
    def summary(complete):
        s=summarize(rows,plan_hash,MODELS,complete,plan['exploratory'])
        s['parameter_selection']=protocol['selection'];s['test_reuse']=protocol['test_reuse'];return s
    try:
        for fold in plan['folds']:
            if args.fold is not None and fold['fold']!=args.fold:continue
            tr=[index[s] for s in fold['development']];te=[index[s] for s in fold['test']]
            for target,y in labels.items():
                records=inner_records(x,y,fold,index);selected=choose_all(records)
                decisions.append({'fold':fold['fold'],'target':target,'selected':selected,'candidates':records})
                save_json(out/'selections.json',decisions)
                # All six decisions persisted before accessing the outer fold scores.
                for name in MODELS:
                    spec=selected[name]['spec'];bundle=fit_v8_bundle(spec,x[tr],y[tr]);pred=predict_bundle(bundle,x[te])
                    folder=out/'models'/target/name/f"fold_{fold['fold']}"
                    save_bundle(folder,bundle,x[tr],y[tr],{'target':target,'name':name,'fold':fold['fold'],
                        'seed':20261015,'training_ids':fold['development'],
                        'training_groups':sorted({plan['groups'][s] for s in fold['development']}),
                        'test_ids':fold['test'],'plan_sha256':plan_hash,'feature_block':'declared block(s) in V8 matrix'})
                    np.savez_compressed(folder/'replay.npz',x=x[te],prediction=pred)
                    for i,sid in enumerate(fold['test']):
                        rows.append({'subject_id':sid,'group_id':plan['groups'][sid],'fold':fold['fold'],
                            'target':target,'model':name,'y_true':float(y[te[i]]),'prediction':float(pred[i]),
                            'mean_prediction':float(y[tr].mean()),'median_prediction':float(np.median(y[tr]))})
                    save_csv(out/'predictions.csv',rows);save_json(out/'summary.json',summary(False))
                    print(target,name,'fold',fold['fold'],spec['id'],'MAE',round(float(np.mean(abs(pred-y[te]))),4),flush=True)
        for sid,h in hashes.items():
            if sha256(root/(sid+'.npz'))!=h['feature'] or sha256(eroot/(sid+'.npz'))!=h['embedding']:
                raise RuntimeError('Protected input changed')
        save_json(out/'summary.json',summary(args.fold is None))
        save_json(out/'status.json',{'state':'complete','complete_outer_cv':args.fold is None,
            'input_features_unchanged':True,'elapsed_seconds':time.time()-started})
    except Exception as exc:
        save_json(out/'status.json',{'state':'failed','complete_outer_cv':False,'error':str(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--plan',required=True);p.add_argument('--embeddings',required=True)
    p.add_argument('--output',required=True);p.add_argument('--fold',type=int);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--allow-unreviewed',action='store_true')
    args=p.parse_args()
    if args.threads<1:p.error('threads must be positive')
    with threadpool_limits(limits=args.threads):main(args)
