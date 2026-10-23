import unittest,copy,tempfile
import numpy as np
import torch
from features_v171 import TypedPreprocessor,split_views,FOCUS,focus_mae
from models_v171 import BagEncoder,TypedMIL,TabEnsemble,family_losses,pcgrad_backward
from engine_v171 import convex_weights,combine,fit_classical,predict_classical,fit_new,predict_new,CONFIGS_NEW
from targets import SCHEMA
from heart import HEART_DIM
from train import make_plan

def synthetic(n=15):
    rng=np.random.default_rng(7);records=[]
    for i in range(n):
        h=np.full((2,HEART_DIM),np.nan,np.float32);h[0,:50]=rng.normal(size=50);h[0,-2:]=[1,0]
        h[1,50:-2]=rng.normal(size=21);h[1,-2:]=[0,1]
        records.append(dict(eeg=rng.normal(size=(5,4)).astype('float32'),eye=rng.normal(size=(3,3)).astype('float32'),heart=h))
    y=np.array([[2. if t['low']==1 else 5. for t in SCHEMA] for _ in records])+rng.normal(0,.1,(n,len(SCHEMA)))
    y[:,1:8]=1.;y[::3,2]=np.nan
    return records,y,dict(eeg=4,eye=3,heart=HEART_DIM)

class Tests(unittest.TestCase):
    def test_typed_heart_no_alignment(self):
        r,_,_=synthetic();v=split_views(r[0]);self.assertEqual(v['ppg'].shape,(1,50));self.assertEqual(v['nni'].shape,(1,21))
    def test_attention_permutation_padding_and_absence(self):
        net=BagEncoder(6,8,0,True).eval();x=torch.randn(3,5,6);m=torch.ones(3,5,dtype=torch.bool);m[:,-2:]=False
        ix=torch.tensor([4,1,0,3,2]);torch.testing.assert_close(net(x,m),net(x[:,ix],m[:,ix]))
        q=x.clone();q[:,-2:]+=1000;torch.testing.assert_close(net(x,m),net(q,m))
        torch.testing.assert_close(net(x,torch.zeros_like(m)),torch.zeros(3,8))
    def test_individual_member_loss_not_mean_prediction_loss(self):
        schema=[dict(family='PSQI',weight=1.)];y=torch.zeros(1,1);obs=torch.ones(1,1,dtype=torch.bool)
        p=torch.tensor([[[-1.],[1.]]],requires_grad=True)
        loss=family_losses(p,y,obs,schema)[0];self.assertGreater(float(loss),.5);loss.backward()
        self.assertTrue((p.grad.abs()>0).all())
    def test_pcgrad_conflict_and_private_updates(self):
        class Small(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor([1.,1.]));self.p=torch.nn.Parameter(torch.tensor([2.]))
            def shared_parameters(self):return [self.w]
        net=Small();a=net.w[0]+net.p[0];b=-net.w[0]+net.w[1]
        conflict=pcgrad_backward([a,b],net);self.assertGreater(conflict,0);self.assertAlmostEqual(float(net.p.grad),.5)
        self.assertTrue(torch.isfinite(net.w.grad).all())
    def test_preprocessing_cannot_refit_from_queries(self):
        r,y,d=synthetic();prep=TypedPreprocessor().fit(r[:10],d);center=prep.summary_scaler.center.copy()
        q=copy.deepcopy(r[10:]);q[0]['eeg']*=1e8;prep.transform(q);np.testing.assert_array_equal(center,prep.summary_scaler.center)
    def test_missing_modality_cannot_change_prediction(self):
        r,y,d=synthetic();r[0]['heart']=np.empty((0,HEART_DIM),np.float32)
        prep=TypedPreprocessor().fit(r,d);x,m,s=prep.transform(r)
        x={k:torch.tensor(v) for k,v in x.items()};m={k:torch.tensor(v) for k,v in m.items()};s=torch.tensor(s)
        net=TypedMIL(prep.dims,s.shape[1],SCHEMA,width=8,hidden=12,dropout=0).eval()
        for h in net.heads:torch.nn.init.normal_(h.weight)
        a=net(x,m,s);xx=dict(x);xx['ppg']=x['ppg'].clone();xx['ppg'][0]+=1000
        torch.testing.assert_close(a,net(xx,m,s))
    def test_simplex_and_no_label_imputation(self):
        rng=np.random.default_rng(4);y=rng.normal(size=(35,2));y[:5,1]=np.nan
        predictions=dict(ridge_trimodal=np.zeros((35,2)),good=np.nan_to_num(y)+.1)
        names,w=convex_weights(y,predictions,np.ones(2),'ridge_trimodal')
        np.testing.assert_allclose(w.sum(1),1);self.assertTrue((w>=0).all())
        p=combine(predictions,names,w);self.assertTrue(np.isfinite(p).all());self.assertTrue(np.isnan(y[:5,1]).all())
    def test_outer_labels_do_not_change_plan_or_training(self):
        r,y,d=synthetic();meta=dict(subjects=[dict(subject_id=str(i),group_id=str(i)) for i in range(len(r))])
        plan=make_plan(meta,y,7);q=y.copy();te=plan['folds'][0]['test'];q[te]+=100
        self.assertEqual(plan,make_plan(meta,q,7))
        tr=plan['folds'][0]['train'];rec=[r[i] for i in tr]
        a=fit_classical(rec,y[tr],d,SCHEMA,dict(kind='ridge',alpha=1))
        b=fit_classical(rec,q[tr],d,SCHEMA,dict(kind='ridge',alpha=1))
        np.testing.assert_allclose(predict_classical(a,r),predict_classical(b,r))
    def test_neural_roundtrip_all_four_variants(self):
        r,y,d=synthetic(10)
        for name in ['mil_mean','mil_attention','mil_attention_pcgrad','tab_ensemble']:
            model,_=fit_new(r,y,d,SCHEMA,CONFIGS_NEW[0],name,[5],1,cap=8)
            p=predict_new(model,r);self.assertEqual(p.shape,y.shape);self.assertTrue(np.isfinite(p).all())
        p=torch.randn(3,27,requires_grad=True);truth=torch.randn_like(p);mask=torch.ones_like(p,dtype=torch.bool);mask[0]=False
        torch.stack(family_losses(p,truth,mask,SCHEMA)).mean().backward();self.assertEqual(float(p.grad[0].abs().sum()),0.)
    def test_kernel_is_psd_with_missing_views(self):
        from engine_v171 import KernelViews
        r,_,d=synthetic(10);r[0]['heart']=np.empty((0,HEART_DIM),np.float32)
        kernels=KernelViews().fit(r,d).transform(r)
        for K in kernels:self.assertGreaterEqual(float(np.linalg.eigvalsh(K).min()),-1e-5)
    def test_classical_single_output_and_missing_label_groups(self):
        r,y,d=synthetic(12);y[0,0]=np.nan;y[1,3]=np.nan
        for c in [dict(kind='trees',leaf=3),dict(kind='svr',C=1,epsilon=.1),dict(kind='kernel',weights=[.25]*4,alpha=.1)]:
            model=fit_classical(r,y,d,SCHEMA,c);p=predict_classical(model,r[:4])
            self.assertEqual(p.shape,(4,len(SCHEMA)));self.assertTrue(np.isfinite(p).all())

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
