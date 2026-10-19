"""Independent-process full replay and record-level metric validation."""
import argparse,csv
from pathlib import Path
import joblib,numpy as np
import legacy_project as lp
from common import load_json,save_json,sha256,manifest
from experiment import load_features,summary_rows
from run_literature import predict_models

def main(args):
    root=Path(args.features);folder=Path(args.v9_features);run=Path(args.run)
    _,plans=lp.check_dataset(root);people=manifest(root);summary=load_json(run/'summary.json')
    with (run/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:r['y_true']=float(r['y_true']);r['prediction']=float(r['prediction'])
    computed=summary_rows(rows)
    if computed!=summary['results']:raise AssertionError('Metric mismatch')
    protocol=load_json(run/'protocol.json')
    for name,h in protocol['code_sha256'].items():assert sha256(Path(__file__).parent/name)==h
    replayed=0;audits=[]
    for task,plan in enumerate(plans):
      ids=plan['subject_ids'];records,_=load_features(folder,ids);lookup=dict(zip(ids,records))
      for target in plan['targets']:
        y=dict(zip(ids,lp.labels_for(people,ids,target)))
        for name in computed[target]:
          selected=[r for r in rows if r['target']==target and r['model']==name]
          assert len(selected)==len(ids) and {r['subject_id'] for r in selected}==set(ids)
          assert len({r['subject_id'] for r in selected})==len(ids)
        for fold in plan['folds']:
          a=joblib.load(run/f'outer/{target}_fold_{fold["fold"]}.joblib')
          assert a['training_ids']==fold['development'] and a['test_ids']==fold['test']
          assert not set(a['training_ids'])&set(lp.lock()['reserve_ids'])
          assert not {plan['groups'][s] for s in a['training_ids']}&{plan['groups'][s] for s in a['test_ids']}
          assert a['protocol_sha256']==sha256(run/'protocol.json')
          pred=predict_models(a['models'],[lookup[s] for s in a['test_ids']])
          for name,values in pred.items():
            ref={r['subject_id']:r for r in rows if r['target']==target and r['model']==name and int(r['fold'])==fold['fold']}
            np.testing.assert_allclose(values,[ref[s]['prediction'] for s in a['test_ids']],atol=1e-9,rtol=0)
            for sid in a['test_ids']:assert ref[sid]['y_true']==y[sid] and ref[sid]['group_id']==plan['groups'][sid]
            replayed+=1
          c=next(iter(a['models'].values())).context
          audits.append({'target':target,'fold':fold['fold'],'cpm_fdr_selected':c.cpm['fdr01'].selected_count,
             'cpm_top16_selected':c.cpm['top16'].selected_count,'stack_weights':a['models']['fusion_stack'].weights.tolist(),
             'eeg_kernel_min_eigenvalue':float(np.linalg.eigvalsh(c.eeg.training_kernel).min())})
        path=run/f'final/{target}.joblib';meta=load_json(path.with_suffix('.json'));assert sha256(path)==meta['model_sha256']
        final=joblib.load(path);assert final['training_ids']==ids
        assert final['protocol_sha256']==sha256(run/'protocol.json')
        p=predict_models(final['models'],records[:2]);assert all(np.isfinite(v).all() for v in p.values())
    dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    save_json(dest,{'passed':True,'metrics_recalculated':sum(len(m) for m in computed.values()),
       'outer_model_predictions_replayed':replayed,'reserve_training_count':0,'audits':audits})
    print('Verified',replayed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--v9-features',required=True)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main(p.parse_args())
