"""Label-free V9 inference. Never writes inferred values into questionnaires."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import legacy_project as lp
from common import load_json,save_json,save_csv,sha256,new_output,manifest
from feature_engineering import eeg_descriptors,eye_features,aggregate
from experiment import predictions

def record_for(root,embeddings,sid):
    source=root/(sid+'.npz');before=sha256(source)
    with np.load(source,allow_pickle=False) as z:
        if z['eeg'].ndim!=3 or z['eeg'].shape[1:]!=(16,800) or len(z['eeg'])<3 or len(z['eye'])<1:
            raise ValueError('Paired QC/input shape requirement failed: '+sid)
        s,d,c,bag=eeg_descriptors(z['eeg'],z['eeg_power_windows'])
        eye,ebag,ew,ea,eb,support=eye_features(z['eye'],z['eye_groups'])
        with np.load(embeddings/(sid+'.npz'),allow_pickle=False) as ez:foundation=ez['cbramod_global'].copy()
        result={'eeg_spectral':aggregate(s),'eeg_dynamics':aggregate(d),
            'eeg_connectivity':np.r_[aggregate(c),z['eeg_cov']],'eeg_foundation':foundation,
            'eeg_bag':bag.astype('float32'),'eye_invariant':eye,'eye_raw':z['eye_base'].copy(),
            'eye_bag':ebag,'eeg_half_a':bag[:len(bag)//2].mean(0),
            'eeg_half_b':bag[len(bag)//2:].mean(0),'eye_half_a':ea,'eye_half_b':eb,
            'eye_repeat_support':np.array(support),'counts':np.array([len(bag),len(z['eye']),len(ebag)])}
    if foundation.shape!=(400,) or not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Invalid feature block')
    if sha256(source)!=before:raise RuntimeError('Source changed')
    return result

def main(args):
    root=Path(args.features).resolve(strict=True);run=Path(args.run).resolve(strict=True)
    lp.verify_engine();lp.schema_check(root,lp.lock())
    protocol=load_json(run/'protocol.json')
    if load_json(run/'status.json')['state']!='complete':raise ValueError('Incomplete model run')
    for name,digest in protocol['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError('Training source changed: '+name)
    people=manifest(root);ids=args.ids
    if len(ids)!=len(set(ids)) or not ids or set(ids)-set(people):raise ValueError('Duplicate/missing/unknown IDs')
    for sid in ids:
        if sid in ('.','..') or Path(sid).name!=sid:raise ValueError('Unsafe ID')
    emb=Path(args.embeddings).resolve(strict=True) if args.embeddings else None
    protected=lp.protected_inputs(root)+[run,Path(__file__).parent]
    if emb:protected.append(emb)
    out=new_output(args.output,*protected)
    emb=emb or lp.extract_for_prediction(root,ids,out/'embeddings',args)
    lp.check_encoder(emb);_,hashes,_=lp.read_inputs(root,emb,ids,people)
    records=[record_for(root,emb,s) for s in ids]
    rows=[];model_hashes={}
    for task in range(2):
        file=run/f'final/task_{task}.joblib';integrity=load_json(run/f'final/task_{task}_integrity.json')
        if sha256(file)!=integrity['model_sha256']:raise ValueError('Saved model changed')
        artifact=joblib.load(file)
        if artifact['protocol_sha256']!=sha256(run/'protocol.json'):raise ValueError('Model protocol changed')
        if any(s in lp.lock()['reserve_ids'] for s in artifact['training_ids']):raise ValueError('Model trained on reserve')
        models={t:{name:all_models[name] for name in args.models} for t,all_models in artifact['models'].items()}
        pred=predictions(models,records,artifact['targets']);model_hashes[file.name]=sha256(file)
        for target,values in pred.items():
            for name,numbers in values.items():
                spec=artifact['choices'][target][name]['spec']['id']
                for sid,value in zip(ids,numbers):
                    rows.append({'subject_id':sid,'target':target,'model':name,
                        'predicted_score':float(value),'is_observed_label':False,
                        'training_overlap':sid in artifact['training_ids'],'effective_spec':spec,'exploratory':True})
    for sid,h in hashes.items():
        if sha256(root/(sid+'.npz'))!=h['feature']:raise RuntimeError('Source changed')
    save_csv(out/'predictions.csv',rows)
    save_json(out/'provenance.json',{'sources':hashes,'models':model_hashes,'target_fields_accessed':False,
        'note':'Predicted scores are not observed scale scores or clinical diagnoses.'})
    print('Complete:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--run',required=True)
    p.add_argument('--ids',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--embeddings')
    p.add_argument('--models',nargs='+',default=['selected_eeg','selected_eye','selected_fusion','fusion_multitask'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.threads<1 or args.batch_size<1:p.error('threads and batch-size must be positive')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):main(args)
