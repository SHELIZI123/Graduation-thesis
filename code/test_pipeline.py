import unittest,copy
import numpy as np
import torch
from common import MODALITIES
from heart import ppg_record,nni_record,HEART_DIM
from targets import SCHEMA,parse
from preprocessing import subset,Robust,group_splits
from model import SetEncoder,TriScale,masked_group_loss

class Tests(unittest.TestCase):
    def test_ppg_amplitude_invariance(self):
        x=np.sin(np.arange(2048)*.1)+.2*np.sin(np.arange(2048)*.2)
        np.testing.assert_allclose(ppg_record(x),ppg_record(7*x+20),atol=1e-8)
    def test_interval_relative_features_and_native_units(self):
        x=.8+.05*np.sin(np.arange(100)*.3);a=nni_record(x);b=nni_record(x*1000)
        np.testing.assert_allclose(a[2:],b[2:],atol=1e-8)
        np.testing.assert_allclose(a[:2]*1000,b[:2],atol=1e-8)
    def test_masked_label_loss_and_gradients(self):
        p=torch.randn(4,len(SCHEMA),requires_grad=True);y=torch.randn_like(p);m=torch.ones_like(p,dtype=torch.bool);m[0]=False
        a=masked_group_loss(p,y,m,SCHEMA);z=y.clone();z[0]=float('nan');b=masked_group_loss(p,z,m,SCHEMA)
        torch.testing.assert_close(a,b);b.backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertEqual(float(p.grad[0].abs().sum()),0)
    def test_set_permutation_and_padding(self):
        e=SetEncoder(5,8,0).eval();x=torch.randn(3,7,5);m=torch.ones(3,7,dtype=torch.bool);m[:,5:]=False
        order=torch.tensor([5,2,0,6,1,4,3]);torch.testing.assert_close(e(x,m),e(x[:,order],m[:,order]))
        z=x.clone();z[:,5:]=1000;torch.testing.assert_close(e(x,m),e(z,m))
    def test_hash_sampling_invariance(self):
        x=np.random.default_rng(1).normal(size=(100,4)).astype(np.float32)
        a=subset(x,8);b=subset(x[::-1],8);np.testing.assert_array_equal(a,b)
    def test_modality_absence_and_heart_gradients(self):
        dims={'eeg':5,'eye':4,'heart':HEART_DIM};net=TriScale(dims,SCHEMA,width=8,hidden=12,dropout=0,modality_dropout=0)
        x={m:torch.randn(3,2,2*dims[m],requires_grad=True) for m in MODALITIES}
        masks={m:torch.ones(3,2,dtype=torch.bool) for m in MODALITIES};net.eval()
        for head in net.heads:torch.nn.init.normal_(head.weight,std=.1)
        out=net(x,masks);out.square().sum().backward();self.assertGreater(float(x['heart'].grad.abs().sum()),0)
        masks['heart'][:]=False;a=net(x,masks);xx=dict(x);xx['heart']=x['heart']+1000;b=net(xx,masks)
        torch.testing.assert_close(a,b)
        for m in masks:masks[m][:]=False
        with self.assertRaises(ValueError):net(x,masks)
    def test_query_transform_cannot_refit_scaler(self):
        scaler=Robust().fit(np.arange(20).reshape(10,2));c=scaler.center.copy();s=scaler.scale.copy()
        scaler.transform(np.full((3,2),1e12));np.testing.assert_array_equal(c,scaler.center);np.testing.assert_array_equal(s,scaler.scale)
    def test_repeated_scales_and_observed_raw_epq(self):
        source={'scales':{'PSQI':[],'SCL90':[],'EPQ':[{'scores':{'nw_ys':7,'nw_bz':65},'factors':[],'observed_targets':{}}]}}
        y,e,issues=parse(source);j=next(i for i,t in enumerate(SCHEMA) if t['name']=='epq_E');self.assertEqual(y[j],7)
        self.assertTrue(all(v is None for i,v in enumerate(y) if i!=j))
        source['scales']['EPQ']*=2;y,_,_=parse(source);self.assertTrue(all(v is None for v in y))
    def test_positive_negative_alias_mapping(self):
        pos=next(t for t in SCHEMA if t['name']=='scl_positive_count');neg=next(t for t in SCHEMA if t['name']=='scl_negative_count')
        self.assertEqual(pos['legacy'],'yaxs_ys');self.assertEqual(neg['legacy'],'yixs_ys')
    def test_identity_groups_never_cross(self):
        groups=['a','a','b','c','d','e','f','g'];splits=group_splits(list(range(8)),groups,3,3)
        self.assertEqual(sorted(i for a,b in splits for i in b),list(range(8)))
        for a,b in splits:self.assertFalse({groups[i] for i in a}&{groups[i] for i in b})
    def test_all_reference_routes(self):
        dims={'eeg':5,'eye':4,'heart':HEART_DIM}
        x={m:torch.randn(2,3,2*dims[m]) for m in MODALITIES}
        mask={m:torch.ones(2,3,dtype=torch.bool) for m in MODALITIES}
        for route in ['ple','mmoe','shared']:
            net=TriScale(dims,SCHEMA,width=8,hidden=12,routing=route,modality_dropout=0)
            p=net(x,mask);self.assertEqual(tuple(p.shape),(2,len(SCHEMA)));self.assertTrue(torch.isfinite(p).all())

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
