"""V10 future-person inference with no questionnaire input or source modification."""
import argparse
from pathlib import Path
import joblib
import legacy_project as lp
from common import load_json,save_json,save_csv,sha256,new_output,manifest
from predict import record_for
from run_literature import predict_models

def main(args):
    root=Path(args.features).resolve(strict=True);run=Path(args.run).resolve(strict=True)
    lp.verify_engine();lp.schema_check(root,lp.lock());protocol=load_json(run/'protocol.json')
    if load_json(run/'status.json')['state']!='complete':raise ValueError('Incomplete model run')
    for name,h in protocol['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=h:raise ValueError('Training source changed: '+name)
    people=manifest(root);ids=args.ids
    if not ids or len(set(ids))!=len(ids) or set(ids)-set(people):raise ValueError('Invalid or duplicate IDs')
    if any(s in ('.','..') or Path(s).name!=s for s in ids):raise ValueError('Unsafe ID')
    emb=Path(args.embeddings).resolve(strict=True) if args.embeddings else None
    protected=lp.protected_inputs(root)+[run,Path(__file__).parent]
    if emb:protected.append(emb)
    out=new_output(args.output,*protected);emb=emb or lp.extract_for_prediction(root,ids,out/'embeddings',args)
    lp.check_encoder(emb);_,hashes,_=lp.read_inputs(root,emb,ids,people)
    records=[record_for(root,emb,s) for s in ids];rows=[]
    for target in ['depression','anxiety','psqi']:
        path=run/f'final/{target}.joblib';meta=load_json(path.with_suffix('.json'))
        if sha256(path)!=meta['model_sha256']:raise ValueError('Model hash changed')
        a=joblib.load(path)
        if a['protocol_sha256']!=sha256(run/'protocol.json'):raise ValueError('Protocol hash changed')
        if set(a['training_ids'])&set(lp.lock()['reserve_ids']):raise ValueError('Reserve leakage')
        selected={m:a['models'][m] for m in args.models};p=predict_models(selected,records)
        for name,v in p.items():
            for sid,value in zip(ids,v):rows.append({'subject_id':sid,'target':target,'model':name,'predicted_score':float(value),
                'is_observed_label':False,'training_overlap':sid in a['training_ids'],'exploratory':True})
    for sid,h in hashes.items():
        if sha256(root/(sid+'.npz'))!=h['feature']:raise RuntimeError('Source changed')
    save_csv(out/'predictions.csv',rows);save_json(out/'provenance.json',{'sources':hashes,'target_fields_accessed':False,
        'note':'Predictions are not observed questionnaire scores or clinical diagnoses.'})
    print('Complete',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--run',required=True)
    p.add_argument('--ids',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--embeddings')
    p.add_argument('--models',nargs='+',default=['fusion_stack']);p.add_argument('--device',default='cuda:0')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args()
    if args.threads<1 or args.batch_size<1:p.error('threads and batch-size must be positive')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):main(args)
