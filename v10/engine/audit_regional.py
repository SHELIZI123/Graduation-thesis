"""Audit unchanged encoder output and inspect already-trained V8 head fit, without tuning."""
import argparse
from pathlib import Path
import numpy as np
from common import load_json,save_json,save_csv,manifest,metrics,new_output
from improve import read_inputs
from benchmark import reload_bundle,protected_inputs
from benchmark_models import predict_bundle
from threadpoolctl import threadpool_limits


def main(args):
    old,emb=Path(args.old_embeddings),Path(args.embeddings)
    encoder=load_json(emb/'encoder.json');differences=[]
    for sid in encoder['sources']:
        with np.load(emb/(sid+'.npz'),allow_pickle=False) as a,np.load(old/(sid+'.npz'),allow_pickle=False) as b:
            np.testing.assert_array_equal(a['window_indices'],b['window_indices'])
            error=float(np.max(np.abs(a['cbramod_global']-b['cbramod'])))
            np.testing.assert_allclose(a['cbramod_global'],b['cbramod'],rtol=1e-5,atol=2e-6)
            differences.append(error)
    rows=[]
    for runpath,planpath in zip(args.runs,args.plans):
        run,plan=Path(runpath),load_json(planpath);protocol=load_json(run/'protocol.json')
        root=Path(protocol['features']);ids=plan['subject_ids'];people=manifest(root);index={s:i for i,s in enumerate(ids)}
        x,_,_=read_inputs(root,emb,ids,people)
        for fold in plan['folds']:
            tr=[index[s] for s in fold['development']];te=[index[s] for s in fold['test']]
            for target in plan['targets']:
                y=np.array([people[s]['targets'][target] for s in ids],float)
                for name in ['power_pca','regional_cbramod','eye_regularized','late_fusion','guarded_fusion']:
                    bundle=reload_bundle(run/'models'/target/name/f"fold_{fold['fold']}")
                    rows.append({'target':target,'model':name,'fold':fold['fold'],
                        'selected':bundle['parameter']['id'],
                        'train_mae':metrics(y[tr],predict_bundle(bundle,x[tr]))['mae'],
                        'test_mae':metrics(y[te],predict_bundle(bundle,x[te]))['mae']})
    protected=[old.resolve(),emb.resolve()]+[Path(p).resolve() for p in args.runs]
    for runpath in args.runs:
        protected += protected_inputs(Path(load_json(Path(runpath)/'protocol.json')['features']))
    output=new_output(args.output,*protected)
    save_csv(output/'train_test_audit.csv',rows)
    result={'same_window_indices_as_v7':True,'global_control_allclose_v7':True,'subjects':len(differences),
        'max_global_difference':max(differences),'saved_models_inspected':len(rows),
        'note':'Post-evaluation descriptive audit only; no model refit or selection'}
    save_json(output/'audit.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-embeddings',required=True)
    p.add_argument('--embeddings',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--plans',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if len(a.runs)!=len(a.plans):p.error('Match runs and plans')
    with threadpool_limits(limits=2):main(a)
