"""Meaningful correctness tests; no access to real patient data."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from baselines import SafeImputer
from common import new_output,validate_folds
from features import balanced_stats,target_features,nni_features,eeg_features
from make_splits import group_partitions


class CoreTests(unittest.TestCase):
    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'raw';root.mkdir()
            for p in [root,root/'child',root.parent]:
                with self.assertRaises(ValueError):new_output(p,root)
            result=new_output(Path(folder)/'derived',root)
            with self.assertRaises(FileExistsError):new_output(result,root)

    def test_group_splits_and_leakage(self):
        ids=[str(i) for i in range(24)];groups={s:str(int(s)//2) for s in ids};folds=[]
        for i,test in enumerate(group_partitions(ids,groups,3,7)):
            dev=sorted(set(ids)-set(test))
            inner=[{'train':sorted(set(dev)-set(v)),'validation':v} for v in group_partitions(dev,groups,2,8+i)]
            folds.append({'development':dev,'test':test,'inner':inner})
        plan={'groups':groups,'subject_ids':ids,'folds':folds}
        validate_folds(plan)
        plan['folds'][0]['inner'][0]['train'].append(plan['folds'][0]['test'][0])
        with self.assertRaises(AssertionError):validate_folds(plan)

    def test_imputation_is_train_only(self):
        fit=SafeImputer().fit([[1,np.nan],[3,np.nan]])
        before=fit.fill_.copy()
        transformed=fit.transform([[100000,5],[np.nan,np.nan]])
        np.testing.assert_array_equal(before,fit.fill_)
        np.testing.assert_array_equal(transformed[-1],[2,0])

    def test_equal_measurement_weight(self):
        v=np.array([[0.],[0.],[0.],[10.]])
        np.testing.assert_allclose(balanced_stats(v,np.array([0,0,0,1])),[5,5])

    def test_invalid_eye_samples_do_not_enter_target_error(self):
        t=np.arange(100)/60;gaze=np.zeros((100,3));stim=np.zeros((100,3));gaze[10]=1e9
        valid=np.ones(100,bool);valid[10]=False
        native={'time_s':t,'channel1_xyz':gaze,'stimulus_xyz':stim,'channel1_valid':valid}
        np.testing.assert_array_equal(target_features(native,0,2),np.zeros(6))

    def test_nni_units_and_multiple_visits(self):
        records=[{'record_sequence':1,'nni':[.8]*200}]
        self.assertTrue(np.isnan(nni_features(records,{})).all())
        self.assertAlmostEqual(nni_features(records,{'nni_unit':'seconds'})[0],75.)
        self.assertTrue(np.isnan(nni_features(records*2,{'nni_unit':'seconds'})).all())

    def test_spectral_covariance_features(self):
        x=np.random.default_rng(4).normal(size=(3,16,800)).astype('float32')
        spectral,cov=eeg_features(x)
        self.assertEqual(spectral.shape,(3,160));self.assertEqual(cov.shape,(3,136))
        self.assertTrue(np.isfinite(cov).all())
        np.testing.assert_allclose(spectral[:,80:].reshape(3,16,5).sum(-1),1,atol=1e-6)


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch=torch;torch.set_num_threads(2)
        except ImportError:raise unittest.SkipTest('torch not installed; run neural tests in training environment')

    def batch(self):
        t=self.torch
        return {'eeg':t.randn(2,3,16,800),'power':t.randn(2,3,160),'eye':t.randn(2,4,2,4,250),
                'eye_stats':t.randn(2,4,2,10),'measurement_mask':t.tensor([[1,1,0,0],[1,1,1,0]],dtype=t.bool)}

    def test_independent_bag_permutation(self):
        from models import SubjectModel
        t=self.torch;t.manual_seed(1);m=SubjectModel().eval()
        t.nn.init.normal_(m.head[-1].weight,std=.1)
        b=self.batch();c={k:v.clone() for k,v in b.items()}
        c['eeg']=c['eeg'][:,[2,0,1]];c['power']=c['power'][:,[2,0,1]]
        for k in ['eye','eye_stats','measurement_mask']:c[k]=c[k][:,[2,0,3,1]]
        c['eye']=c['eye'][:,:,[1,0]];c['eye_stats']=c['eye_stats'][:,:,[1,0]]
        with t.no_grad():np.testing.assert_allclose(m(b).numpy(),m(c).numpy(),atol=1e-5)

    def test_each_branch_learns(self):
        from models import SubjectModel
        t=self.torch;t.manual_seed(3);m=SubjectModel();opt=t.optim.Adam(m.parameters(),lr=.001)
        b=self.batch();target=t.tensor([[1.,-1.],[-1.,1.]])
        for _ in range(3):
            opt.zero_grad();loss=((m(b)-target)**2).mean();loss.backward()
            self.assertTrue(t.isfinite(loss));opt.step()
        for encoder in [m.eeg,m.eye]:
            grads=[p.grad for p in encoder.parameters() if p.grad is not None]
            self.assertTrue(all(t.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)


if __name__=='__main__':unittest.main(verbosity=2)
