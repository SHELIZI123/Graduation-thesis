"""Independently recompute predictions/coverage and replay saved local regression models."""
import argparse
from pathlib import Path
import numpy as np
from common import csv_rows, load_json, save_json, sha256, metrics
from benchmark import reload_bundle, release, validate_plan
from benchmark_models import predict_bundle


def main(args):
    root=Path(args.run).resolve(strict=True);protocol=load_json(root/'protocol.json')
    plan=load_json(args.plan)
    if sha256(args.plan)!=protocol['plan_sha256']:raise ValueError('Wrong plan')
    validate_plan(plan,plan['feature_manifest_sha256'])
    features=Path(protocol['features'])
    if sha256(features/'manifest.json')!=plan['feature_manifest_sha256']:raise ValueError('Feature manifest changed')
    people={r['subject_id']:r for r in load_json(features/'manifest.json')['subjects']}
    rows=csv_rows(root/'predictions.csv');summary=load_json(root/'summary.json');keys=set()
    models=summary['models'];folds={str(f['fold']):f for f in plan['folds']}
    for r in rows:
        key=(r['subject_id'],r['target'],r['model'],r['fold'])
        if key in keys:raise ValueError('Duplicate heldout prediction')
        keys.add(key)
        if r['subject_id'] not in folds[r['fold']]['test']:raise ValueError('Wrong heldout ID')
        if r['group_id']!=plan['groups'][r['subject_id']]:raise ValueError('Wrong group ID')
        truth=people[r['subject_id']]['targets'][r['target']]
        training=[people[s]['targets'][r['target']] for s in folds[r['fold']]['development']]
        if not np.isclose(float(r['y_true']),truth,atol=1e-12,rtol=0):raise ValueError('Wrong heldout target')
        if not np.isclose(float(r['mean_prediction']),np.mean(training),atol=1e-12,rtol=0):raise ValueError('Mean baseline not train-only')
        if not np.isclose(float(r['median_prediction']),np.median(training),atol=1e-12,rtol=0):raise ValueError('Median baseline not train-only')
    if summary['complete_outer_cv']:
        expected={(s,t,n,str(f['fold'])) for f in plan['folds'] for s in f['test'] for t in plan['targets'] for n in models}
        if keys!=expected:raise ValueError('Incomplete CV')
    comparisons=0;replays=0
    for target,modelscores in summary['results'].items():
        for name,result in modelscores.items():
            rr=[r for r in rows if r['target']==target and r['model']==name]
            m=metrics([float(r['y_true']) for r in rr],[float(r['prediction']) for r in rr])
            for k,v in m.items():
                if v is not None and not np.isclose(v,result[k],rtol=1e-8,atol=1e-10):raise ValueError('Metric mismatch')
            comparisons+=1
    for folder in sorted((root/'models').glob('*/*/fold_*')):
        meta=load_json(folder/'metadata.json');fold=folds[str(meta['fold'])]
        if meta['training_ids']!=fold['development'] or meta['test_ids']!=fold['test']:raise ValueError('Model training IDs differ')
        if set(meta['training_groups']) & {plan['groups'][s] for s in meta['test_ids']}:raise ValueError('Saved model group leakage')
        if args.skip_tabpfn_replay and meta['kind']=='tabpfn':continue
        bundle=reload_bundle(folder,args.device,args.tabpfn_weights,args.threads)
        with np.load(folder/'replay.npz',allow_pickle=False) as z:
            pred=predict_bundle(bundle,z['x'])
            np.testing.assert_allclose(pred,z['prediction'],atol=1e-5,rtol=1e-5)
        release(bundle);replays+=1
    result={'passed':True,'complete_outer_cv':summary['complete_outer_cv'],'metric_tables_recomputed':comparisons,
        'models_reloaded_and_replayed':replays,'tabpfn_replay_skipped':args.skip_tabpfn_replay,
        'heldout_rows':len(rows),'group_leakage_check':True}
    save_json(root/'verification.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--tabpfn-weights');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--skip-tabpfn-replay',action='store_true')
    main(p.parse_args())
