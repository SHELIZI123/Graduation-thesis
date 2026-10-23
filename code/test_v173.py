"""Synthetic engineering tests; fixtures never replace or augment clinical labels."""
from pathlib import Path
import unittest,copy
import numpy as np,torch
from threadpoolctl import threadpool_limits
from test_v172 import fixture
from engine_v173 import FitCache,fit_candidate,predict_candidate,configs
from features_v173 import EEGOnlyPrep,BalancedPrep
from engine import predict_ridge
from foundation_v173 import digest,project
from preprocessing import group_splits
from train_v173 import tune

class TestV173(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r,cls.y,cls.d,cls.s=fixture(36);cls.cache=FitCache(cls.d,cls.s)
    def test_anchor_nested_groups_and_strong_candidates(self):
        a=self.cache.anchor(self.r,self.y);sel=a['anchor_selection']
        self.assertEqual([v['penalty'] for v in sel['search']],[1.,10.,100.])
        for f in sel['folds']:
            self.assertFalse({self.r[int(i)]['group_id'] for i in f['train_ids']}&{self.r[int(i)]['group_id'] for i in f['validation_ids']})
    def test_own_labels_excluded_from_anchor_tuning_in_crossfit(self):
        a=self.cache.residual(self.r,self.y);first=a['crossfit'][0];ix=list(map(int,first['validation_ids']))
        changed=self.y.copy();changed[ix]+=100
        b=FitCache(self.d,self.s).residual(self.r,changed)
        np.testing.assert_allclose((self.y-a['residual'])[ix],(changed-b['residual'])[ix],atol=1e-10)
        self.assertEqual(first['selected_penalty'],b['crossfit'][0]['selected_penalty'])
        self.assertEqual(set(first['anchor_selection']['fit_ids']),set(first['train_ids']))
    def test_zero_correction_exactly_matches_matched_anchor(self):
        a=fit_candidate('residual_aux',self.r,self.y,self.d,self.s,dict(epochs=0,shrink=0),[1],'cpu',self.cache)
        np.testing.assert_array_equal(predict_candidate(a,self.r[:3]),predict_ridge(self.cache.anchor(self.r,self.y),self.r[:3]))
        self.assertEqual(a['penalty'],self.cache.anchor(self.r,self.y)['penalty'])
    def test_all_three_eeg_representations_matched_dimension(self):
        records=copy.deepcopy(self.r);d=dict(self.d,eeg=20);rng=np.random.default_rng(51)
        for r in records:r['eeg']=rng.normal(size=(4,20))
        dims=[]
        for kind in ['summary','tangent','frozen']:
            a=EEGOnlyPrep().fit(records,d,kind);x=a.matrix(records[:3]);dims.append(x.shape[1]);self.assertTrue(np.isfinite(x).all())
        self.assertEqual(dims,[33,33,33])
    def test_eeg_only_excludes_missing_eeg_training_people(self):
        r=copy.deepcopy(self.r);r[0]['eeg']=np.empty((0,6));r[0]['cov'][:]=np.nan;r[0]['embedding'][:]=np.nan
        a=fit_candidate('eeg_frozen',r,self.y,self.d,self.s,dict(penalty=1.),[1],'cpu')
        self.assertNotIn('0',a['effective_fit_ids']);self.assertNotIn('0',a['prep'].training_ids)
        np.testing.assert_allclose(predict_candidate(a,r[:1])[0],a['mean'],atol=1e-10)
    def test_feature_transforms_do_not_refit_on_query(self):
        a=BalancedPrep().fit(self.r[:24],self.d,True);before=copy.deepcopy(a.scales)
        x=a.matrix(self.r[24:]);self.assertEqual(a.scales,before);self.assertTrue(np.isfinite(x).all())
        self.assertEqual(a.training_ids,[r['subject_id'] for r in self.r[:24]])
    def test_block_energies_match_prespecified_weights(self):
        for augmented in [False,True]:
            a=BalancedPrep().fit(self.r,self.d,augmented);raw=a.blocks(self.r)
            for k,x in raw.items():
                energy=np.mean(np.sum(((x-a.centers[k])/a.scales[k]*np.sqrt(a.weights[k]))**2,1))
                self.assertAlmostEqual(float(energy),a.weights[k],places=5)
            self.assertAlmostEqual(sum(a.weights.values()),3.)
    def test_distribution_bank_has_complete_predictions(self):
        inner=group_splits(list(range(len(self.r))),[r['group_id'] for r in self.r],3,33)
        a,p,d=tune('ridge_distribution',self.r,self.y,self.d,self.s,inner,[1],'cpu',True,self.cache)
        a.update(tuning_decision=d)
        self.assertEqual(a['kind'],'v173_bank');self.assertEqual(p.shape,self.y.shape)
        self.assertTrue(np.isfinite(predict_candidate(a,self.r[:3])).all())
    def test_positive_epoch_residual_has_tuned_anchor(self):
        a=fit_candidate('residual_plain',self.r,self.y,self.d,self.s,dict(epochs=2,shrink=.25),[3],'cpu',self.cache)
        self.assertEqual(a['base_penalty'],self.cache.anchor(self.r,self.y)['penalty'])
        self.assertEqual(a['logs'][0]['optimizer_steps'],4)
        self.assertTrue(np.isfinite(predict_candidate(a,self.r[:3])).all())
    def test_query_hash_sensitive_to_tensor_values(self):
        x=np.zeros((2,3),np.float32);before=digest(x);x[0,0]=1;self.assertNotEqual(before,digest(x))
    def test_new_epoch_grid_prespecified(self):
        self.assertEqual([c['epochs'] for c in configs('residual_aux')],[0,2,8,24])

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):unittest.main(verbosity=2)
