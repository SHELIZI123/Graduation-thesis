from pathlib import Path
import argparse,csv,collections
import numpy as np
import joblib,torch
from threadpoolctl import threadpool_limits
from common import read,write,load_data,sha
from preprocessing import target_stats,Robust,BagPreprocessor,normalized_mae
from engine import predict,baseline_matrix,bound

def main(a):
    meta,records,y=load_data(a.data);count=0;maximum=0.;folds=[]
    for directory in a.runs:
        root=Path(directory);protocol=read(root/'protocol.json');plan=protocol['plan'];decisions=read(root/'decisions.json')
        assert sha(Path(a.data)/'manifest.json')==protocol['data_manifest_sha256']
        assert protocol['code_sha256']=={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
        assert read(root/'status.json')['state']=='complete'
        with (root/'predictions.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        keyed={(int(r['fold']),r['model'],r['subject_id'],r['target']):r for r in rows};assert len(keyed)==len(rows)
        for d in decisions:
            fold=d['fold'];folds.append(fold);split=plan['folds'][fold];tr,te=split['train'],split['test']
            assert not set(tr)&set(te)
            assert not {plan['groups'][i] for i in tr}&{plan['groups'][i] for i in te}
            assert [plan['subject_ids'][i] for i in tr]==d['train_ids']
            assert [plan['subject_ids'][i] for i in te]==d['test_ids']
            for inner in split['inner']:
                assert not {plan['groups'][i] for i in inner['train']}&{plan['groups'][i] for i in inner['validation']}
                assert set(inner['train'])|set(inner['validation'])==set(tr)
            for name,item in d['baselines'].items():
                scores={float(k):np.mean(v) for k,v in item['inner_scores'].items()}
                assert item['penalty']==min(scores,key=lambda p:(scores[p],-p))
            for name,item in d['neural'].items():
                expected=min(item['search'],key=lambda x:(x['score'],x['config']['hidden']))
                assert item['selected']==expected
                assert expected['refit_epochs']==int(np.median(expected['inner_best_epochs']))
            train=[records[i] for i in tr];test=[records[i] for i in te];mu,sd=target_stats(y[tr])
            pp={'training_mean':bound(np.broadcast_to(mu,(len(te),len(meta['schema']))),meta['schema']),
                'training_median':bound(np.broadcast_to([np.nanmedian(c) for c in y[tr].T],(len(te),len(meta['schema']))),meta['schema'])}
            for file in sorted((root/f'fold_{fold}').glob('*.joblib')):
                model=joblib.load(file);assert model['training_ids']==d['train_ids'] and model['test_ids']==d['test_ids']
                np.testing.assert_allclose(model['mean'],mu);np.testing.assert_allclose(model['scale'],sd)
                if model['kind']=='ridge':
                    raw=baseline_matrix(train,meta['dimensions'],model['modalities'],model['mask_only']);ref=Robust().fit(raw)
                    np.testing.assert_allclose(ref.center,model['scaler'].center);np.testing.assert_allclose(ref.scale,model['scaler'].scale)
                else:
                    ref=BagPreprocessor().fit(train,meta['dimensions'])
                    for m in ref.scalers:
                        np.testing.assert_allclose(ref.scalers[m].center,model['preprocessor'].scalers[m].center)
                        np.testing.assert_allclose(ref.scalers[m].scale,model['preprocessor'].scalers[m].scale)
                pp[file.stem]=predict(model,test,a.device)
                if file.stem=='set_ple':
                    valid=np.array([bool(len(r['eeg']) or len(r['eye'])) for r in test]);p=np.full_like(pp[file.stem],np.nan)
                    drop=[{**r,'heart':np.empty((0,meta['dimensions']['heart']),np.float32)} for r in test]
                    if valid.any():p[valid]=predict(model,[drop[i] for i in np.where(valid)[0]],a.device)
                    pp['set_ple_heart_removed_probe']=p
            expected_count=0
            for name,p in pp.items():
                for ri,i in enumerate(te):
                    for j,t in enumerate(meta['schema']):
                        if not (np.isfinite(y[i,j]) and np.isfinite(p[ri,j])):continue
                        expected_count+=1;row=keyed[(fold,name,plan['subject_ids'][i],t['name'])]
                        assert float(row['y_true'])==y[i,j]
                        delta=abs(float(row['prediction'])-p[ri,j]);maximum=max(maximum,float(delta))
                        np.testing.assert_allclose(float(row['prediction']),p[ri,j],atol=1e-4,rtol=1e-6);count+=1
            assert expected_count==sum(int(r['fold'])==fold for r in rows)
    assert len(folds)==len(set(folds))
    result={'passed':True,'predictions_replayed':count,'maximum_absolute_prediction_difference':maximum,
            'atol':1e-4,'rtol':1e-6,'training_only_preprocessors_rechecked':True,
            'selection_rules_recalculated':True,'outer_groups_disjoint':True,'completed_folds':sorted(folds),
            'full_fivefold':sorted(folds)==list(range(5)),'independent_test':False}
    write(a.output,result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):main(a)
