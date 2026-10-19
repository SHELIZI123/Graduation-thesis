"""Checks for inference determinism, train-only fitting and subject separation."""
import copy
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from argparse import Namespace
from common import save_json, sha256
from benchmark import validate_plan, summarize
from benchmark_models import BlendedKernel, fit_bundle, predict_bundle
from make_splits import group_partitions


class BaselineTests(unittest.TestCase):
    def test_alias_groups_do_not_cross_folds(self):
        ids=[str(i) for i in range(30)];g={s:str(int(s)//2) for s in ids};folds=[]
        for i,te in enumerate(group_partitions(ids,g,3,42)):
            dev=sorted(set(ids)-set(te))
            inner=[{'train':sorted(set(dev)-set(v)),'validation':v} for v in group_partitions(dev,g,2,7+i)]
            folds.append({'fold':i,'development':dev,'test':te,'inner':inner})
        p={'subject_ids':ids,'groups':g,'feature_manifest_sha256':'hash','folds':folds}
        validate_plan(p,'hash')
        q=copy.deepcopy(p);q['groups'][q['folds'][0]['test'][0]]=q['groups'][q['folds'][0]['development'][0]]
        with self.assertRaises(ValueError):validate_plan(q,'hash')
        with self.assertRaises(ValueError):validate_plan(p,'changed')

    def test_kernel_test_rows_do_not_change_fitted_transforms(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(20,420));y=rng.normal(size=20)
        model=BlendedKernel().fit(x,y);widths=list(model.widths_)
        means=[tr[-1].mean_.copy() for tr in model.transforms_]
        a=model.predict(x[:1]);b=model.predict(np.r_[x[:1],np.full((3,420),1e8)])[:1]
        np.testing.assert_allclose(a,b)
        self.assertEqual(widths,model.widths_)
        for before,tr in zip(means,model.transforms_):np.testing.assert_array_equal(before,tr[-1].mean_)

    def test_kernel_endpoints_match_single_modal_inputs(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(18,420));y=rng.normal(size=18)
        for weight,part in [(1.,slice(400,None)),(0.,slice(0,400))]:
            model=BlendedKernel(weight=weight).fit(x,y);changed=x.copy();changed[:,part]+=1e6
            np.testing.assert_allclose(model.predict(x),model.predict(changed),atol=1e-10)

    def test_target_transform_and_saved_model_replay(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(20,6));y=10+3*x[:,0]
        model=fit_bundle('ridge',{'alpha':1.},x,y)
        self.assertAlmostEqual(model['target_mean'],y.mean())
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'model.joblib';joblib.dump(model,p);other=joblib.load(p)
            np.testing.assert_array_equal(predict_bundle(model,x),predict_bundle(other,x))

    def test_constant_baseline_and_partial_flag(self):
        rows=[{'target':'psqi','model':'ridge','y_true':y,'prediction':5.,'mean_prediction':5.,'median_prediction':5.}
            for y in [3.,5.,7.]]
        result=summarize(rows,'hash',['ridge'],False,True)
        self.assertFalse(result['complete_outer_cv']);self.assertEqual(result['results']['psqi']['ridge']['r2'],0.)

    def test_missing_target_eye_not_replaced_by_zeros(self):
        from benchmark import read_blocks
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez(p/'001.npz',eeg_power=np.ones(160),eeg_cov=np.ones(136),
                eye_base=np.ones(20),eye_target=np.full(12,np.nan))
            with self.assertRaisesRegex(ValueError,'unavailable'):
                read_blocks(p,['001'],{'001':{'age':30,'sex':'男'}},['eye_target_svr'])

    def test_outer_labels_cannot_change_model_selection_or_predictions(self):
        from benchmark import main
        from threadpoolctl import threadpool_limits
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=2):
            p=Path(temp);features=p/'features';features.mkdir();raw=p/'raw';raw.mkdir();processed=p/'processed';processed.mkdir()
            ids=[f'{i:03d}' for i in range(18)];groups={s:s for s in ids};people=[];rng=np.random.default_rng(3)
            for sid in ids:
                x=rng.normal(size=160)
                np.savez(features/(sid+'.npz'),eeg_power=x,eeg_cov=np.ones(136),eye_base=np.ones(20),eye_target=np.full(12,np.nan))
                people.append({'subject_id':sid,'age':30,'sex':'男','targets':{'psqi':float(10+x[0])}})
            save_json(features/'config.json',{'raw_root':str(raw),'processed_root':str(processed)})
            folds=[]
            for i,test in enumerate(group_partitions(ids,groups,3,8)):
                dev=sorted(set(ids)-set(test))
                inner=[{'train':sorted(set(dev)-set(v)),'validation':v} for v in group_partitions(dev,groups,2,9+i)]
                folds.append({'fold':i,'development':dev,'test':test,'inner':inner})
            (p/'plans').mkdir()
            plan={'subject_ids':ids,'groups':groups,'folds':folds,'targets':['psqi'],'exploratory':True}
            predictions=[];selections=[]
            for turn in range(2):
                if turn:
                    for person in people:
                        if person['subject_id'] in folds[0]['test']:person['targets']['psqi']+=10000
                save_json(features/'manifest.json',{'format_version':3,'subjects':people})
                plan['feature_manifest_sha256']=sha256(features/'manifest.json');save_json(p/'plans/plan.json',plan)
                args=Namespace(features=str(features),plan=str(p/'plans/plan.json'),models='power_ridge',embeddings=None,
                    output=str(p/f'run{turn}'),fold=0,threads=2,device='cpu',seed=7,tabpfn_weights=None)
                main(args)
                bundle=joblib.load(p/f'run{turn}'/'models/psqi/power_ridge/fold_0/model.joblib')
                probe=np.ones((1,160));predictions.append(predict_bundle(bundle,probe))
                import json
                selections.append(json.loads((p/f'run{turn}'/'selections.json').read_text())[-1]['selected'])
            self.assertEqual(selections[0],selections[1]);np.testing.assert_array_equal(*predictions)


class EncoderTests(unittest.TestCase):
    def test_frozen_inference_batch_size_and_window_permutation(self):
        import torch
        from extract_cbramod import load_encoder, encode_windows
        torch.set_num_threads(2)
        model=load_encoder(random_control=True)
        x=np.random.default_rng(7).normal(size=(3,16,800)).astype('float32')*.1
        a=encode_windows(model,x,'cpu',1);b=encode_windows(model,x[[2,0,1]],'cpu',3)
        np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__=='__main__':unittest.main(verbosity=2)
