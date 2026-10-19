"""Regression safeguards for small-sample selection, transforms and regional pooling."""
import copy
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from improved_models import REGIONS,BLOCKS,V8Regressor,base_candidates,inner_records,choose_all,choose,compact_power


class ImprovedTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(42)
        self.x=self.rng.normal(size=(30,2072));self.y=3+self.rng.normal(size=30)

    def test_regions_cover_exact_electrodes(self):
        self.assertEqual(sorted(sum(REGIONS.values(),[])),list(range(16)))
        v=np.r_[np.arange(80),np.arange(80)+100]
        p=compact_power(v)
        np.testing.assert_allclose(p[:5],v[:80].reshape(16,5)[REGIONS['frontal']].mean(0))
        self.assertEqual(p.shape,(50,))

    def test_token_pool_preserves_region_information(self):
        import torch
        from extract_regional import pool_tokens
        z=torch.zeros(2,16,4,200)
        z[:,REGIONS['frontal']]=2
        regional,glob=pool_tokens(z)
        np.testing.assert_allclose(regional.numpy()[:,:200],2)
        np.testing.assert_allclose(regional.numpy()[:,200:],0)
        np.testing.assert_allclose(glob.numpy(),.75)
        perm=[1,0,3,2]
        np.testing.assert_allclose(pool_tokens(z[:,:,perm])[0],regional)

    def test_pca_is_train_only_and_reload_exact(self):
        spec=next(s for s in base_candidates() if s.get('block')=='regional')
        m=V8Regressor(spec).fit(self.x[:22],self.y[:22]);before=m.pipeline_[1].mean_.copy()
        p=m.predict(self.x[22:23]);q=m.predict(np.r_[self.x[22:23],np.full((3,2072),1e8)])[:1]
        np.testing.assert_allclose(p,q,rtol=1e-8,atol=1e-8)
        np.testing.assert_array_equal(m.pipeline_[1].mean_,before)
        np.testing.assert_allclose(m.pipeline_[1].mean_,self.x[:22,70:2070].mean(0))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'model.joblib';joblib.dump(m,path)
            np.testing.assert_array_equal(m.predict(self.x[22:]),joblib.load(path).predict(self.x[22:]))

    def test_no_demographics_in_physiological_predictions(self):
        specs=[s for s in base_candidates() if s.get('block') in {'power','eye','regional'}]
        changed=self.x.copy();changed[:,2070:]=1e10
        for s in specs:
            model=V8Regressor(s).fit(self.x,self.y)
            np.testing.assert_array_equal(model.predict(self.x),model.predict(changed))

    def test_fusion_endpoints_and_score_units(self):
        eeg=next(s for s in base_candidates() if s.get('block')=='power')
        eye=next(s for s in base_candidates() if s.get('block')=='eye')
        for weight,base in [(0.,eye),(1.,eeg)]:
            m=V8Regressor({'kind':'blend','eeg':eeg,'eye':eye,'eeg_weight':weight}).fit(self.x,self.y)
            np.testing.assert_allclose(m.predict(self.x),V8Regressor(base).fit(self.x,self.y).predict(self.x))
        y=10+4*self.y
        p=V8Regressor(eeg).fit(self.x,self.y).predict(self.x)
        np.testing.assert_allclose(V8Regressor(eeg).fit(self.x,y).predict(self.x),10+4*p)

    def test_conservative_rule_can_select_constant(self):
        records=[{'spec':{'id':'mean','kind':'constant','statistic':'mean'},'inner_mae':1.02,'fold_mae':[1.,1.02,1.04]},
            {'spec':{'id':'eye','kind':'svr','block':'eye','C':.1},'inner_mae':1.,'fold_mae':[.7,1.,1.3]}]
        s=choose(records,lambda s:True)
        self.assertEqual(s['spec']['kind'],'constant')
        self.assertEqual(choose(records,lambda s:True,False)['spec']['kind'],'svr')

    def test_outer_targets_cannot_affect_inner_selection_or_predictions(self):
        ids=[str(i) for i in range(30)];index={s:i for i,s in enumerate(ids)}
        dev=ids[:24];fold={'development':dev,'test':ids[24:],'inner':[]}
        for i in range(3):
            val=dev[8*i:8*(i+1)]
            fold['inner'].append({'train':[s for s in dev if s not in val],'validation':val})
        changed=self.y.copy();changed[24:]+=1000
        with threadpool_limits(limits=1):
            a=choose_all(inner_records(self.x,self.y,fold,index))
            b=choose_all(inner_records(self.x,changed,fold,index))
        self.assertEqual(a,b)
        for name in a:
            p=V8Regressor(a[name]['spec']).fit(self.x[:24],self.y[:24]).predict(self.x[24:])
            q=V8Regressor(b[name]['spec']).fit(self.x[:24],changed[:24]).predict(self.x[24:])
            np.testing.assert_array_equal(p,q)

    def test_relocation_copies_without_changing_source(self):
        from argparse import Namespace
        from common import save_json,load_json,sha256
        from relocate_features import main
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=root/'source';source.mkdir()
            raw=root/'raw';raw.mkdir();processed=root/'processed';processed.mkdir()
            save_json(source/'config.json',{'raw_root':'D:\\original','processed_root':'D:\\processed'})
            save_json(source/'manifest.json',{'subjects':[]});save_json(source/'review_snapshot.json',{})
            np.savez_compressed(source/'001.npz',x=[1.,2.])
            before={p.name:sha256(p) for p in source.iterdir()}
            main(Namespace(source=source,output=root/'copied',raw_root=raw,processed_root=processed))
            self.assertEqual(before,{p.name:sha256(p) for p in source.iterdir()})
            for name in ['manifest.json','review_snapshot.json','001.npz']:
                self.assertEqual(before[name],sha256(root/'copied'/name))
            self.assertEqual(load_json(root/'copied'/'config.json')['raw_root'],str(raw.resolve()))
            with self.assertRaises(ValueError):
                main(Namespace(source=source,output=source/'unsafe',raw_root=raw,processed_root=processed))

    def test_mismatched_regional_source_is_rejected(self):
        from common import save_json,sha256
        from improve import read_inputs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);features=root/'features';features.mkdir();emb=root/'emb';emb.mkdir()
            save_json(features/'manifest.json',{})
            np.savez_compressed(features/'001.npz',eeg_power=np.zeros(160),eye_base=np.zeros(20))
            np.savez_compressed(emb/'001.npz',regional=np.zeros(2000))
            record={'status':'complete','feature_manifest_sha256':sha256(features/'manifest.json'),
                'regions':REGIONS,'regional_dimension':2000,'sources':{'001':{
                    'input_sha256':'deliberately wrong','output_sha256':sha256(emb/'001.npz')}}}
            save_json(emb/'encoder.json',record)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                read_inputs(features,emb,['001'],{'001':{'age':20,'sex':'男'}})


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
