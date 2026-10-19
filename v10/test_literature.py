"""Meaningful numerical and leakage-boundary checks for new estimators."""
import unittest
import numpy as np
from literature_models import ExactSetKernel,CPMSummary,Context,Regressor,fit_weights

def records(n=18):
    rng=np.random.default_rng(10)
    return [{k:rng.normal(size=s) for k,s in {'eeg_bag':(8,115),'eye_bag':(3,50),
      'eeg_spectral':(180,),'eeg_dynamics':(50,),'eeg_connectivity':(286,),
      'eeg_foundation':(400,),'eye_invariant':(75,)}.items()} for _ in range(n)]

class Tests(unittest.TestCase):
    def test_empirical_kernel_against_direct(self):
        bags=[np.array([[0.,1],[1,2]]),np.array([[2.,3],[3,4],[4,5]])]
        m=ExactSetKernel().fit(bags);z=[m.standardize(b) for b in bags]
        direct=np.array([[np.mean([np.exp(-sum((a-b)**2)/m.width) for a in x for b in y]) for y in z] for x in z])
        np.testing.assert_allclose(m.mean_gram(z),direct,atol=1e-13)
        np.testing.assert_allclose(m.transform(bags),m.training_kernel,atol=1e-13)
    def test_psd(self):
        rng=np.random.default_rng(7);bags=[rng.normal(size=(i+2,4)) for i in range(12)]
        m=ExactSetKernel().fit(bags)
        self.assertGreater(np.linalg.eigvalsh(m.training_kernel).min(),-1e-10)
    def test_query_does_not_fit_or_depend_on_other_queries(self):
        rng=np.random.default_rng(1);bags=[rng.normal(size=(5,3)) for _ in range(7)]
        m=ExactSetKernel().fit(bags);before=m.scaler.center_.copy()
        p=m.transform(bags[:1]);q=m.transform([bags[0],bags[1]*1000])
        np.testing.assert_allclose(p[0],q[0],atol=1e-13);np.testing.assert_array_equal(before,m.scaler.center_)
    def test_permutation_and_replication_invariance(self):
        rng=np.random.default_rng(3);bags=[rng.normal(size=(5,3)) for _ in range(6)]
        m=ExactSetKernel().fit(bags)
        np.testing.assert_allclose(m.transform(bags[:1]),m.transform([np.repeat(bags[0][::-1],2,axis=0)]),atol=1e-13)
    def test_no_cpm_connections_returns_train_mean(self):
        r=records();y=np.ones(len(r))*2.3;c=Context().fit(r,y)
        self.assertEqual(c.cpm['fdr01'].selected_count,0)
        model=Regressor().fit(c,{'family':'cpm_fdr01','rule':'fdr01','alpha':1.})
        np.testing.assert_allclose(model.predict(r[:2]),2.3)
    def test_cpm_only_training_changes_selection(self):
        r=records();y=np.linspace(-1,1,len(r));m=CPMSummary().fit(r,y,'top16')
        masks=[v.copy() for v in m.masks];m.transform(r[::-1]);self.assertEqual(m.selected_count,16)
        for a,b in zip(masks,m.masks):np.testing.assert_array_equal(a,b)
    def test_convex_stacking(self):
        y=np.arange(10,dtype=float);p=np.column_stack([np.full(10,4.5),y,9-y])
        w=fit_weights(p,y);self.assertAlmostEqual(w.sum(),1,places=7);self.assertTrue((w>=0).all())
        self.assertGreater(w[1],.95)

if __name__=='__main__':unittest.main()
