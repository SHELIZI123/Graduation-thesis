from pathlib import Path
import argparse,time,platform,sys,collections
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from common import read,write,csvwrite,sha,new_dir,load_data,MODALITIES
from preprocessing import group_splits,target_stats,normalized_mae
from engine import fit_neural,fit_ridge,predict,CONFIGS,RIDGE_LAMBDAS,bound

BASELINES={'ridge_eeg':('eeg',),'ridge_eye':('eye',),'ridge_heart':('heart',),
           'ridge_eeg_eye':('eeg','eye'),'ridge_trimodal':MODALITIES,'ridge_mask_only':MODALITIES}

def make_plan(meta,y,seed):
    ids=[r['subject_id'] for r in meta['subjects']];groups=[r['group_id'] for r in meta['subjects']]
    eligible=np.where(np.isfinite(y).any(1))[0].tolist();folds=[]
    for i,(tr,te) in enumerate(group_splits(eligible,groups,5,seed)):
        inner=group_splits(tr,groups,3,seed+100+i)
        folds.append({'fold':i,'train':tr,'test':te,'inner':[{'train':a,'validation':b} for a,b in inner]})
    return {'seed':seed,'subject_ids':ids,'groups':groups,'eligible_indices':eligible,'folds':folds,
            'split_uses_label_values':False,'group_disjoint':True}

def run(a):
    meta,records,y=load_data(a.data);out=new_dir(a.output,[a.data,Path(__file__).parent])
    schema=meta['schema'];dims=meta['dimensions'];plan=make_plan(meta,y,a.seed)
    src={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    protocol={'version':'V17','data_manifest_sha256':sha(Path(a.data)/'manifest.json'),'schema':schema,
              'plan':plan,'neural_configurations':CONFIGS,'baseline_penalties':RIDGE_LAMBDAS,
              'architectures':a.architectures,'seeds':a.seeds,'maximum_epochs':a.epochs,
              'patience':a.patience,'cap':a.cap,'code_sha256':src,'selected_folds':a.folds,
              'exploratory':True,'independent_test':False,'all_reserved_candidates_used_are_development':True,
              'neural_selection':'Inner threefold family-balanced MAE normalized by INNER TRAIN target SD. Refit outer train with median INNER epoch; no outer early stopping.',
              'heart_units':'PPG shape in sample coordinates; native interval values and unitless ratios. No BPM, ms or Hz HRV.',
              'hardware':{'device':a.device,'torch':torch.__version__,'python':sys.version,'platform':platform.platform()}}
    write(out/'protocol.json',protocol);write(out/'status.json',{'state':'running','completed':[]})
    begin=time.perf_counter();rows=[];decisions=[];completed=[]
    try:
        for fold in plan['folds']:
            fi=fold['fold']
            if a.folds is not None and fi not in a.folds:continue
            print('Outer fold',fi,'elapsed',round(time.perf_counter()-begin),'s',flush=True)
            tr,te=fold['train'],fold['test'];train=[records[i] for i in tr];test=[records[i] for i in te]
            yy=y[tr];fold_dir=out/f'fold_{fi}';fold_dir.mkdir()
            means,std=target_stats(yy);medians=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in yy.T])
            predictions={'training_mean':bound(np.broadcast_to(means,(len(te),len(schema))),schema),
                         'training_median':bound(np.broadcast_to(medians,(len(te),len(schema))),schema)}
            decisions_fold={'fold':fi,'train_ids':[plan['subject_ids'][i] for i in tr],'test_ids':[plan['subject_ids'][i] for i in te],
                            'baselines':{},'neural':{}}
            for name,modalities in BASELINES.items():
                scores={pen:[] for pen in RIDGE_LAMBDAS}
                for inner in fold['inner']:
                    it,iv=inner['train'],inner['validation'];itrec=[records[i] for i in it];ivrec=[records[i] for i in iv]
                    _,s=target_stats(y[it])
                    for pen in RIDGE_LAMBDAS:
                        model=fit_ridge(itrec,y[it],dims,schema,pen,modalities,name=='ridge_mask_only')
                        scores[pen].append(normalized_mae(y[iv],predict(model,ivrec),s,schema))
                best=min(scores,key=lambda p:(np.mean(scores[p]),-p))
                artifact=fit_ridge(train,yy,dims,schema,best,modalities,name=='ridge_mask_only')
                artifact.update(training_ids=decisions_fold['train_ids'],test_ids=decisions_fold['test_ids'])
                joblib.dump(artifact,fold_dir/(name+'.joblib'),compress=3)
                predictions[name]=predict(artifact,test)
                decisions_fold['baselines'][name]={'penalty':best,'inner_scores':scores}
            print('  Baselines done',fi,'elapsed',round(time.perf_counter()-begin),flush=True)
            for architecture in a.architectures:
                searches=[]
                for config in CONFIGS:
                    scores=[];epochs=[]
                    for inner_i,inner in enumerate(fold['inner']):
                        it,iv=inner['train'],inner['validation'];_,s=target_stats(y[it])
                        artifact,p=fit_neural([records[i] for i in it],y[it],dims,schema,config,architecture,
                            a.seeds,a.epochs,a.device,([records[i] for i in iv],y[iv]),a.patience,a.cap)
                        scores.append(normalized_mae(y[iv],p,s,schema));epochs.extend(r['epochs'] for r in artifact['logs'])
                    searches.append({'config':config,'inner_scores':scores,'inner_best_epochs':epochs,
                                     'score':float(np.mean(scores)),'refit_epochs':int(np.median(epochs))})
                    print(' ',architecture,config['id'],'inner',round(np.mean(scores),4),'epochs',epochs,flush=True)
                decision=min(searches,key=lambda x:(x['score'],x['config']['hidden']))
                artifact,_=fit_neural(train,yy,dims,schema,decision['config'],architecture,a.seeds,decision['refit_epochs'],a.device,cap=a.cap)
                artifact.update(training_ids=decisions_fold['train_ids'],test_ids=decisions_fold['test_ids'])
                joblib.dump(artifact,fold_dir/('set_'+architecture+'.joblib'),compress=3)
                predictions['set_'+architecture]=predict(artifact,test,a.device)
                decisions_fold['neural'][architecture]={'selected':decision,'search':searches}
            # Deliberately erase heart inputs at inference for a sensitivity probe, NOT a retrained ablation.
            if 'set_ple' in predictions:
                dropped=[{**r,'heart':np.empty((0,dims['heart']),np.float32)} for r in test]
                valid=np.array([bool(len(r['eeg']) or len(r['eye'])) for r in test])
                p=np.full_like(predictions['set_ple'],np.nan)
                if valid.any():p[valid]=predict(artifact,[dropped[i] for i in np.where(valid)[0]],a.device) if architecture=='ple' else predict(joblib.load(fold_dir/'set_ple.joblib'),[dropped[i] for i in np.where(valid)[0]],a.device)
                predictions['set_ple_heart_removed_probe']=p
            for name,p in predictions.items():
                for ri,i in enumerate(te):
                    for j,t in enumerate(schema):
                        if np.isfinite(y[i,j]) and np.isfinite(p[ri,j]):
                            entry=meta['subjects'][i]
                            rows.append({'subject_id':entry['subject_id'],'group_id':entry['group_id'],'fold':fi,
                                         'model':name,'target':t['name'],'family':t['family'],'y_true':float(y[i,j]),
                                         'prediction':float(p[ri,j]),'complete_three_modality':all(entry['modality_available'].values())})
            write(fold_dir/'decisions.json',decisions_fold);decisions.append(decisions_fold);completed.append(fi)
            csvwrite(out/'predictions.csv',rows);write(out/'decisions.json',decisions)
            write(out/'status.json',{'state':'running','completed':completed,'seconds':time.perf_counter()-begin})
        if src!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}:raise ValueError('Code changed during training')
        write(out/'status.json',{'state':'complete','completed':completed,'full_fivefold':sorted(completed)==list(range(5)),
                                 'seconds':time.perf_counter()-begin})
    except Exception as exc:
        write(out/'status.json',{'state':'failed','completed':completed,'error':repr(exc)});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=160)
    p.add_argument('--patience',type=int,default=25);p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--seeds',nargs='+',type=int,default=[20261023]);p.add_argument('--folds',nargs='+',type=int)
    p.add_argument('--architectures',nargs='+',choices=['ple','mmoe','shared'],default=['ple'])
    p.add_argument('--cap',type=int,default=32);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(a.threads)
    with threadpool_limits(limits=a.threads):run(a)
