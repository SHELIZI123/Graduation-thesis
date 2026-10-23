"""Synthetic fixtures only: leakage boundaries, geometry, optimization, replay."""
from pathlib import Path
import unittest,copy,tempfile
import numpy as np
import torch,joblib
from threadpoolctl import threadpool_limits
from common import read
from features_v172 import EEGGeometry,FeaturePrep,matrix_function,geometric_mean
from engine_v172 import FitCache,fit_candidate,predict_candidate
from models_v172 import ResidualFusion,masked_loss,pareto_pair
from engine import fit_ridge,predict_ridge

def fixture(n=36):
    rng=np.random.default_rng(71);schema=read(Path(__file__).resolve().parents[1]/'data/manifest.json')['schema']
    records=[];dims=dict(eeg=6,eye=4,heart=73)
    for i in range(n):
        a=rng.normal(size=(5,16,16));c=a@a.transpose(0,2,1)+2*np.eye(16)
        p=np.full((2,73),np.nan,np.float32);p[0,:50]=rng.normal(size=50);p[1,50:71]=rng.normal(size=21);p[:,-2:]=[[1,0],[0,1]]
        records.append(dict(subject_id=str(i),group_id=str(i//2),eeg=rng.normal(size=(4,6)),eye=rng.normal(size=(2,4)),heart=p,
                            cov=c,embedding=rng.normal(size=400)))
    y=np.empty((n,len(schema)))
    for j,t in enumerate(schema):
        low=0 if t['low'] is None else t['low'];high=low+10 if t['high'] is None else t['high']
        y[:,j]=rng.uniform(low+.1*(high-low),high-.1*(high-low),n)
    return records,y,dims,schema

class TestV172(unittest.TestCase):
    def test_geometry_train_only_and_finite(self):
        r,y,d,s=fixture(8);g=EEGGeometry().fit(r[:6]);before=np.stack(g.references).copy()
        v=g.transform(r[6:]);self.assertEqual(v.shape,(2,680));self.assertTrue(np.isfinite(v).all())
        np.testing.assert_array_equal(before,np.stack(g.references))
        self.assertTrue(all(a['residual_norm']<1e-5 for a in g.convergence))
    def test_geometric_mean_diagonal(self):
        a=np.stack([np.diag([1.,4.]),np.diag([9.,16.])]);g,_=geometric_mean(a)
        np.testing.assert_allclose(g,np.diag([3.,8.]),atol=1e-6)
    def test_missing_eeg_transform(self):
        r,y,d,s=fixture(12);prep=FeaturePrep().fit(r[:8],d)
        q=copy.deepcopy(r[8]);q['eeg']=np.empty((0,6));q['cov'][:]=np.nan;q['embedding'][:]=np.nan
        x,m=prep.transform([q]);self.assertTrue(all(np.isfinite(z).all() for z in x));self.assertFalse(m[0,0])
    def test_unimodal_and_fusion_gradients(self):
        r,y,d,s=fixture(8);torch.manual_seed(4);model=ResidualFusion([4,5,6],s)
        p,aux=model([torch.randn(8,k) for k in [4,5,6]],torch.ones(8,3,dtype=torch.bool))
        loss=masked_loss(p,torch.randn_like(p),torch.ones_like(p,dtype=torch.bool),s);loss.backward()
        self.assertTrue(all(sum(float(p.grad.norm()) for p in e.parameters())>0 for e in model.encoders))
    def test_pareto_conflict_and_zero(self):
        g=torch.tensor([1.,0.]);h=torch.tensor([-.5,1.]);v,c,ratio=pareto_pair(g,h)
        self.assertLess(c,0);self.assertGreaterEqual(float(v@g),-1e-6);self.assertGreaterEqual(float(v@h),-1e-6)
        z=pareto_pair(g,torch.zeros_like(g))[0];torch.testing.assert_close(g,z)
    def test_grouped_residual_crossfit_excludes_own_labels(self):
        r,y,d,s=fixture();a=FitCache(d,s).residual(r,y)
        for f in a['crossfit']:
            tr={r[int(i)]['group_id'] for i in f['train_ids']};va={r[int(i)]['group_id'] for i in f['validation_ids']}
            self.assertFalse(tr&va)
        ix=list(map(int,a['crossfit'][0]['validation_ids']));changed=y.copy();changed[ix]+=100
        b=FitCache(d,s).residual(r,changed)
        np.testing.assert_allclose((y-a['residual'])[ix],(changed-b['residual'])[ix],atol=1e-10)
    def test_positive_epochs_and_artifact_replay(self):
        r,y,d,s=fixture();a=fit_candidate('residual_mmpareto',r,y,d,s,dict(epochs=2,shrink=.25),[7],'cpu')
        self.assertEqual(a['logs'][0]['optimizer_steps'],4)
        self.assertTrue(all(v>0 for v in a['logs'][0]['first_encoder_grad_norm']))
        p,parts=predict_candidate(a,r[:3],components=True);self.assertGreater(max(parts['correction_sd']),0)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'a.joblib';joblib.dump(a,path);b=joblib.load(path)
            np.testing.assert_array_equal(p,predict_candidate(b,r[:3]))
    def test_ridge_control_exact(self):
        r,y,d,s=fixture();a=fit_candidate('ridge_trimodal',r,y,d,s,dict(penalty=1.),[1],'cpu')
        b=fit_ridge(r,y,d,s,1.);np.testing.assert_array_equal(predict_candidate(a,r[:3]),predict_ridge(b,r[:3]))
    def test_zero_correction_is_explicit_baseline(self):
        r,y,d,s=fixture();a=fit_candidate('residual_aux',r,y,d,s,dict(epochs=0,shrink=0.),[1],'cpu')
        self.assertEqual(a['kind'],'ridge');self.assertTrue(a['explicit_residual_fallback'])
    def test_missing_labels_do_not_affect_loss(self):
        _,_,_,s=fixture(2);p=torch.randn(2,len(s));truth=torch.zeros_like(p);mask=torch.ones_like(p,dtype=torch.bool);mask[:,0]=False
        a=masked_loss(p,truth,mask,s);truth[:,0]=10000;b=masked_loss(p,truth,mask,s);torch.testing.assert_close(a,b)

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):unittest.main(verbosity=2)
